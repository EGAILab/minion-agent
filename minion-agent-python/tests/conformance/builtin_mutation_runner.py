"""Runner for `builtin_mutation` scenarios
(`conformance/schema/builtin-mutation-scenario.schema.json`).

Thin by design: it builds the fixture on disk, wraps the REAL `LocalFileSystem` in each declared
provider (scripted errors and gates for named paths, every call recorded), constructs the REAL
`write`/`edit` tools and runs every call through the REAL Layer 06 `execute_call` pipeline. It
never implements queueing, abort checks or tool logic itself: a gate only holds a provider call,
and an abort step only aborts the call's own signal.
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minion_agent.execution import Err, FsError, FsErrorCode, LocalFileSystem
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin import create_edit_tool, create_write_tool
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

from .builtin_tool_runner import build_fixture

_OPERATIONS = (
    "canonical_path",
    "absolute_path",
    "create_dir",
    "write_file",
    "check_read_write",
    "check_readable",
    "read_binary_file",
)


@dataclass
class _Gate:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None
    held: bool = False


class ScriptedProvider:
    """The real local `ctx.fs` under one provider name: scripted errors, gates, a call record."""

    def __init__(
        self,
        name: str,
        root: Path,
        provider: dict[str, Any],
        log: list[str],
        gates: dict[tuple[str, str, int], _Gate],
        abort_after: str | None = None,
        controller: RunAbortController | None = None,
    ) -> None:
        self._name = name
        self._root = root
        self._local = LocalFileSystem(str(root))
        self._provider = provider
        self._log = log
        self._gates = gates
        self._abort_after = abort_after
        self._controller = controller
        self._counts: dict[tuple[str, str], int] = {}
        self.calls: list[str] = []
        self.inflight = 0

    def recorded(self, path: str) -> str:
        """A relative path verbatim; an absolute one relative to the fixture root ('.' for it)."""
        if not os.path.isabs(path):
            return path
        rel = os.path.relpath(path, self._root)
        return "." if rel == "." else rel.replace(os.sep, "/")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._local, name)

    async def _call(self, operation: str, path: str, real: Any) -> Any:
        rel = self.recorded(path)
        self.calls.append(f"{operation} {rel}")
        n = self._counts[(operation, rel)] = self._counts.get((operation, rel), 0) + 1
        prefix = f"{self._name} {operation} {rel} #{n}"
        self._log.append(prefix + " start")
        answer: Any = None
        gate = self._gates.get((operation, rel, n))
        if gate is not None:
            gate.held = True
            await gate.event.wait()
            gate.held = False
            if gate.error is not None:
                answer = Err(FsError(FsErrorCode(gate.error), "gated", path))
        if answer is None:
            answer = self._scripted(operation, rel, path)
        if answer is None:
            self.inflight += 1
            try:
                answer = await real()
            finally:
                self.inflight -= 1
        self._log.append(
            prefix + (" " + answer.error.code.value if isinstance(answer, Err) else " ok")
        )
        if operation == self._abort_after and self._controller is not None:
            self._abort_after = None  # the FIRST invocation's answer
            self._controller.abort()
        return answer

    def _scripted(self, operation: str, rel: str, path: str) -> Any:
        unsupported = (
            operation == "check_read_write" and self._provider.get("without_exec_009")
        ) or (operation == "check_readable" and self._provider.get("without_exec_008"))
        if unsupported:
            return Err(FsError(FsErrorCode.NOT_SUPPORTED, "not supported", path))
        for entry in self._provider.get(operation, []):
            if entry["path"] == rel:
                return Err(FsError(FsErrorCode(entry["error"]), "scripted", path))
        return None

    async def canonical_path(self, path: str, signal: Any = None) -> Any:
        return await self._call(
            "canonical_path", path, lambda: self._local.canonical_path(path, signal)
        )

    async def absolute_path(self, path: str, signal: Any = None) -> Any:
        return await self._call(
            "absolute_path", path, lambda: self._local.absolute_path(path, signal)
        )

    async def create_dir(self, path: str, recursive: bool = True, signal: Any = None) -> Any:
        return await self._call(
            "create_dir", path, lambda: self._local.create_dir(path, recursive, signal)
        )

    async def write_file(self, path: str, content: str | bytes, signal: Any = None) -> Any:
        return await self._call(
            "write_file", path, lambda: self._local.write_file(path, content, signal)
        )

    async def check_read_write(self, path: str, signal: Any = None) -> Any:
        return await self._call(
            "check_read_write", path, lambda: self._local.check_read_write(path, signal)
        )

    async def check_readable(self, path: str, signal: Any = None) -> Any:
        return await self._call(
            "check_readable", path, lambda: self._local.check_readable(path, signal)
        )

    async def read_binary_file(self, path: str, signal: Any = None) -> Any:
        return await self._call(
            "read_binary_file", path, lambda: self._local.read_binary_file(path, signal)
        )


def _registry(provider: ScriptedProvider) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(create_write_tool(provider))  # type: ignore[arg-type]
    registry.register(create_edit_tool(provider))  # type: ignore[arg-type]
    return registry


def _context() -> Context:
    ctx = Context()
    declare_tools_events(ctx.events)
    return ctx


def files_after(root: Path, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The observed state of each named fixture file, in the expectation's own form."""
    observed: list[dict[str, Any]] = []
    for entry in expected:
        target = root / entry["path"]
        if not target.is_file():
            observed.append({"path": entry["path"], "absent": True})
            continue
        data = target.read_bytes()
        if "text" in entry:
            observed.append({"path": entry["path"], "text": data.decode("utf-8", "replace")})
        else:
            observed.append({"path": entry["path"], "base64": base64.b64encode(data).decode()})
    return observed


def _observe(result: Any) -> dict[str, Any]:
    first = result.content[0]
    assert isinstance(first, TextBlock)
    return {"is_error": result.is_error, "text": first.text, "details": result.details}


async def run_cases(document: dict[str, Any]) -> list[dict[str, Any]]:
    """One `{id, observed, expected}` per case, each on a fresh fixture root."""
    spec = document["builtin_mutation"]
    outcomes: list[dict[str, Any]] = []
    for case in spec["cases"]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            build_fixture(root, spec.get("fixture", []) + case.get("fixture", []))
            controller = RunAbortController()
            provider = ScriptedProvider(
                "p", root, spec.get("provider", {}), [], {}, case.get("abort_after"), controller
            )
            if case.get("signal") == "pre_aborted":
                controller.abort()
            result = await execute_call(
                ToolCallBlock(id="call-1", name=case["tool"], arguments=case["arguments"]),
                registry=_registry(provider),
                ctx=_context(),
                signal=controller.signal,
            )
            observed = _observe(result) | {"fs_calls": provider.calls}
            expected = case["expect"]
            observed["files_after"] = files_after(root, expected.get("files_after", []))
            outcomes.append({"id": case.get("id", ""), "observed": observed, "expected": expected})
    return outcomes


_SETTLE_S = 0.05


async def _quiesce(providers: list[ScriptedProvider], log: list[str]) -> None:
    """Run the scheduler until no task can progress without a gate: no real provider call in
    flight, and no new event across a settle window that spans many scheduler turns AND wall-clock
    time -- so a task reacting on a short timer (a signal poller, say) gets to act before the next
    step, while a task that only polls without acting does not hold the runner forever."""
    while True:
        before = len(log)
        for _ in range(200):
            await asyncio.sleep(0)
        if any(p.inflight for p in providers):
            await asyncio.sleep(0.005)
            continue
        await asyncio.sleep(_SETTLE_S)
        for _ in range(200):
            await asyncio.sleep(0)
        if len(log) == before and not any(p.inflight for p in providers):
            return


async def run_queue(document: dict[str, Any]) -> dict[str, Any]:
    """Run one queue scenario; returns `{log, results, files_after, pending}`."""
    spec = document["builtin_mutation"]
    queue = spec["queue"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        build_fixture(root, spec.get("fixture", []))
        log: list[str] = []
        gates: dict[str, _Gate] = {}
        by_provider: dict[str, dict[tuple[str, str, int], _Gate]] = {}
        names = queue.get("providers", ["p"])
        for gate_spec in queue.get("gates", []):
            gate = gates[gate_spec["id"]] = _Gate()
            key = (gate_spec["operation"], gate_spec["path"], gate_spec.get("occurrence", 1))
            by_provider.setdefault(gate_spec.get("provider", names[0]), {})[key] = gate
        providers = {
            name: ScriptedProvider(
                name, root, spec.get("provider", {}), log, by_provider.get(name, {})
            )
            for name in names
        }
        registries = {name: _registry(provider) for name, provider in providers.items()}
        ctx = _context()
        controllers: dict[str, RunAbortController] = {}
        results: dict[str, Any] = {}
        tasks: dict[str, asyncio.Task[Any]] = {}
        for call in queue["calls"]:
            controller = controllers[call["id"]] = RunAbortController()
            registry = registries[call.get("provider", names[0])]

            async def run(
                call: dict[str, Any] = call,
                registry: ToolRegistry = registry,
                controller: RunAbortController = controller,
            ) -> None:
                result = await execute_call(
                    ToolCallBlock(id=call["id"], name=call["tool"], arguments=call["arguments"]),
                    registry=registry,
                    ctx=ctx,
                    signal=controller.signal,
                )
                log.append(f"result {call['id']}")
                results[call["id"]] = _observe(result)

            tasks[call["id"]] = asyncio.ensure_future(run())
        provider_list = list(providers.values())
        await _quiesce(provider_list, log)
        for step in queue.get("steps", []):
            if "abort" in step:
                controllers[step["abort"]].abort()
            else:
                gate = gates[step["release"]]
                gate.error = step.get("error")
                gate.event.set()
            await _quiesce(provider_list, log)
        pending = sorted(call_id for call_id, task in tasks.items() if not task.done())
        for call_id in pending:
            tasks[call_id].cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            "log": log,
            "results": results,
            "pending": pending,
            "files_after": files_after(root, queue["expect"].get("files_after", [])),
        }
