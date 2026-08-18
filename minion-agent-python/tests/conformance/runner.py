"""Executes declarative conformance scenarios against the runtime.

The scenario vocabulary describes plugins by what they do — the services they
provide, the labelled effects they create, the listeners they register — so
the cases stay language-neutral. This module is the Python half of the runner
contract in conformance/schema/README.md.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from minion_agent.runtime import Context, DispatchMode, PluginSpec, Scope, ScopeKey
from minion_agent.runtime.errors import RuntimeError_


@dataclass
class TraceRecorder:
    """Collects ordered trace entries as the scenario runs."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)


@dataclass
class RunOutcome:
    """What a scenario produced."""

    trace: list[dict[str, Any]]
    result: Any = None
    error: Exception | None = None


class ScopeTable:
    """Scope keys and live scopes, keyed by the names scenarios use.

    Scenarios name scopes with plain strings and declare parents by name; this
    resolves those into real `ScopeKey` chains and keeps one live `Scope` per
    name so a `dispose_scope` step can find it.
    """

    def __init__(self, recorder: TraceRecorder) -> None:
        self._recorder = recorder
        self.keys: dict[str, ScopeKey] = {}
        self.live: dict[str, Scope] = {}

    def key_for(self, name: str, parent_name: str | None) -> ScopeKey:
        if name not in self.keys:
            parent = self.keys.get(parent_name) if parent_name else None
            self.keys[name] = ScopeKey(name, parent=parent)
        return self.keys[name]

    def open(self, ctx: Context, name: str, parent_name: str | None) -> Scope:
        if name not in self.live:
            self.live[name] = ctx.scope(self.key_for(name, parent_name))
        return self.live[name]

    async def dispose(self, name: str) -> None:
        """Dispose `name` and every descendant, deepest first."""
        target = self.keys[name]
        descendants = [
            other
            for other, key in self.keys.items()
            if other in self.live and target in key.chain()
        ]
        descendants.sort(key=lambda other: -len(self.keys[other].chain()))
        for other in descendants:
            await self.live.pop(other).dispose()
            self._recorder.record({"event": "scope_disposed", "scope": other})


def _make_listener(
    spec: dict[str, Any],
    plugin_id: str,
    recorder: TraceRecorder,
) -> Callable[..., Awaitable[Any]]:
    action = spec["action"]
    tag = spec["tag"]
    returns = spec.get("returns")
    replacement = spec.get("replacement")

    async def listener(*args: Any) -> Any:
        recorder.record({"event": "listener_entered", "plugin": plugin_id, "tag": tag})
        next_ = args[-1] if args and callable(args[-1]) else None

        if action == "raise":
            raise ValueError(f"{tag} raised")
        if action in ("short_circuit", "observe"):
            return returns
        if next_ is None:
            return returns
        if action == "transform":
            return await next_(replacement)
        if action == "delegate_twice":
            await next_()
            return await next_()
        return await next_()

    return listener


def _make_effect(
    label: str, plugin_id: str, recorder: TraceRecorder
) -> Callable[[], Callable[[], None]]:
    """An effect that records its own creation and disposal.

    Self-recording keeps fiber-owned and scope-owned effects uniform: a
    scope-owned effect never reaches the fiber's `on_effect` hook, because
    ownership follows the registration context by design.
    """

    def execute() -> Callable[[], None]:
        recorder.record({"event": "effect_created", "plugin": plugin_id, "label": label})

        def dispose() -> None:
            recorder.record({"event": "effect_disposed", "plugin": plugin_id, "label": label})

        return dispose

    return execute


def build_plugin(
    entry: dict[str, Any],
    recorder: TraceRecorder,
    scopes: ScopeTable,
    run_step: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> PluginSpec:
    """Turn one declarative plugin entry into a mountable PluginSpec."""
    plugin_id = entry["id"]
    provides = entry.get("provides")
    effects = entry.get("effects", [])
    listeners = entry.get("listeners", [])
    fails = entry.get("fails", False)
    scope_name = entry.get("scope")
    scope_parent = entry.get("scope_parent")
    during_load = entry.get("during_load", [])

    async def apply(ctx: Context, config: Any) -> None:
        # A plugin declaring a scope registers through it, so visibility and
        # ownership both follow the scope rather than this fiber.
        target = ctx
        if scope_name is not None:
            target = scopes.open(ctx, scope_name, scope_parent).ctx

        if provides is not None:
            ctx.provide(provides, {"service": provides})
            recorder.record({"event": "service_provided", "plugin": plugin_id, "service": provides})

        for effect in effects:
            label = effect["label"]
            target.effect(_make_effect(label, plugin_id, recorder), label)

        for listener in listeners:
            target.on(listener["event"], _make_listener(listener, plugin_id, recorder))

        # Run before the failure check: a scenario may need both.
        if during_load and run_step is not None:
            for step in during_load:
                await run_step(step)

        if fails:
            raise ValueError(f"{plugin_id} failed on purpose")

    return PluginSpec(
        name=plugin_id,
        apply=apply,
        inject=tuple(entry.get("inject", ())),
        config_model=None,
        provides=provides,
    )


def _attach_recording(fiber: Any, plugin_id: str, recorder: TraceRecorder) -> None:
    """Record fiber state transitions.

    Declared effects record themselves (see `_make_effect`), which keeps
    fiber-owned and scope-owned effects uniform. Service revocation is not
    recorded: withdrawing a service cascades into dependents while the
    provider is still unloading, so a revocation marker's position relative to
    that cascade reflects how disposal is reported rather than observable
    behavior, and the runner contract admits only observable output.
    """
    fiber.on_state_change = lambda _fiber, state: recorder.record(
        {"event": "fiber_state", "plugin": plugin_id, "state": state.value}
    )


async def run_runtime_scenario(document: dict[str, Any]) -> RunOutcome:
    """Mount the scenario's plugins, run its steps, and return the trace."""
    recorder = TraceRecorder()
    root = Context()
    scopes = ScopeTable(recorder)
    configs = {entry["id"]: entry.get("config") for entry in document["plugins"]}
    fibers: dict[str, Any] = {}
    result: Any = None

    async def execute_step(step: dict[str, Any]) -> None:
        """Apply one scenario step.

        Shared by the top-level loop and by a plugin's `during_load`, so a step
        means the same thing wherever it appears. `specs` is late-bound: it is
        assigned below, before this ever runs.
        """
        nonlocal result

        if "mount" in step:
            plugin_id = step["mount"]
            fiber = root.plugins.mount(specs[plugin_id], configs[plugin_id], root)
            _attach_recording(fiber, plugin_id, recorder)
            fibers[plugin_id] = fiber
            await root.plugins.reconcile()

        elif "unmount" in step:
            await root.plugins.unmount(fibers[step["unmount"]])

        elif "dispose_scope" in step:
            await scopes.dispose(step["dispose_scope"])

        elif "dispatch" in step:
            dispatch = step["dispatch"]
            name = dispatch["event"]
            mode = DispatchMode(dispatch["mode"])
            args = dispatch.get("args", [])
            scope_key = scopes.keys.get(dispatch["scope"]) if "scope" in dispatch else None
            root.events.declare(name, mode)
            if mode is DispatchMode.EMIT:
                root.events.emit(name, *args, scope=scope_key)
            elif mode is DispatchMode.PARALLEL:
                await root.events.parallel(name, *args, scope=scope_key)
            elif mode is DispatchMode.SERIAL:
                result = await root.events.serial(name, *args, scope=scope_key)
            else:
                result = await root.events.waterfall(
                    name,
                    *args,
                    scope=scope_key,
                    terminal=dispatch.get("terminal"),
                )

    specs = {
        entry["id"]: build_plugin(entry, recorder, scopes, execute_step)
        for entry in document["plugins"]
    }

    # Declare every dispatched event up front. A listener cannot register for
    # an undeclared event -- mode is part of the contract -- so declaration has
    # to precede the first mount, not the first dispatch.
    for step in document["steps"]:
        dispatch = step.get("dispatch")
        if dispatch is not None:
            root.events.declare(dispatch["event"], DispatchMode(dispatch["mode"]))

    for step in document["steps"]:
        try:
            await execute_step(step)

        except RuntimeError_ as error:
            return RunOutcome(trace=recorder.entries, result=result, error=error)

    return RunOutcome(trace=recorder.entries, result=result)
