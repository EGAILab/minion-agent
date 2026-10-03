"""`L0506-D004` at the Layer-06 seam: keyword-only delivery that never shifts the certified
`signal`/`update` dispatch, the snapshot point (as `execute` is invoked, after the before-hooks),
absence without a provider, and no provider call for a call that never executes."""

from __future__ import annotations

from typing import Any

import pytest

from minion_agent.llm import ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools import ToolExecutionContext
from minion_agent.tools import execute as execute_module
from minion_agent.tools.decisions import Block
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

CONTEXT = ToolExecutionContext(session_id="s", provider="p", model="m", reasoning_level="off")


def _define(execute: Any, *, wants_signal: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="t",
            label="t",
            description="t",
            parameters={"type": "object", "properties": {}},
            execute=execute,
            wants_signal=wants_signal,
            wants_context=True,
        )
    )
    return registry


async def _call(registry: ToolRegistry, provider: Any = lambda: CONTEXT, hook: Any = None) -> Any:
    ctx = Context()
    declare_tools_events(ctx.events)
    if hook is not None:
        ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    return await execute_call(
        ToolCallBlock(id="c", name="t", arguments={}),
        registry=registry,
        ctx=ctx,
        signal=RunAbortController().signal,
        context_provider=provider,
    )


async def test_context_only_keeps_two_positional_slots() -> None:
    seen: dict[str, Any] = {}

    def execute(tool_call_id: str, args: Any, *, context: Any) -> str:
        seen["context"] = context
        return "ok"

    await _call(_define(execute))
    assert seen["context"] is CONTEXT


async def test_update_and_context_keep_update_in_the_third_slot() -> None:
    seen: dict[str, Any] = {}

    def execute(tool_call_id: str, args: Any, update: Any, *, context: Any) -> str:
        seen["update_callable"] = callable(update)
        seen["context"] = context
        return "ok"

    await _call(_define(execute))
    assert seen == {"update_callable": True, "context": CONTEXT}


@pytest.mark.parametrize("with_update", [False, True])
async def test_signal_with_or_without_update_and_context(with_update: bool) -> None:
    seen: dict[str, Any] = {}

    if with_update:

        def execute(tool_call_id: str, args: Any, signal: Any, update: Any, *, context: Any) -> str:
            seen.update(signal=hasattr(signal, "aborted"), update=callable(update), context=context)
            return "ok"

    else:

        def execute(tool_call_id: str, args: Any, signal: Any, *, context: Any) -> str:  # type: ignore[misc]
            seen.update(signal=hasattr(signal, "aborted"), context=context)
            return "ok"

    await _call(_define(execute, wants_signal=True))
    assert seen["signal"] is True and seen["context"] is CONTEXT
    assert seen.get("update", with_update) is with_update


async def test_without_a_provider_the_context_is_absent_and_the_tool_runs() -> None:
    seen: dict[str, Any] = {}

    def execute(tool_call_id: str, args: Any, *, context: Any) -> str:
        seen["context"] = context
        return "ok"

    result = await _call(_define(execute), provider=None)
    assert seen == {"context": None} and not result.is_error


async def test_the_snapshot_is_taken_as_execute_is_invoked_after_the_before_hooks() -> None:
    state = {"model": "before-hook"}
    calls: list[str] = []

    def provider() -> ToolExecutionContext:
        calls.append("provider")
        return ToolExecutionContext(session_id="s", model=state["model"])

    async def hook(call: Any, definition: Any, args: Any, signal: Any, next_: Any) -> Any:
        calls.append("hook")
        state["model"] = "after-hook"
        return await next_()

    seen: dict[str, Any] = {}

    def execute(tool_call_id: str, args: Any, *, context: Any) -> str:
        calls.append("execute")
        seen["model"] = context.model
        return "ok"

    await _call(_define(execute), provider=provider, hook=hook)
    assert calls == ["hook", "provider", "execute"]
    assert seen == {"model": "after-hook"}


async def test_a_blocked_call_never_asks_for_a_context() -> None:
    calls: list[str] = []

    async def block(call: Any, definition: Any, args: Any, signal: Any, next_: Any) -> Any:
        return Block(reason="no")

    def execute(tool_call_id: str, args: Any, *, context: Any) -> str:  # pragma: no cover
        return "ok"

    result = await _call(_define(execute), provider=lambda: calls.append("provider"), hook=block)
    assert result.is_error and calls == []


async def test_a_tool_that_did_not_opt_in_is_called_exactly_as_before() -> None:
    registry = ToolRegistry()
    seen: dict[str, Any] = {}

    def execute(tool_call_id: str, args: Any, update: Any) -> str:
        seen["update_callable"] = callable(update)
        return "ok"

    registry.register(
        ToolDefinition(
            name="t",
            label="t",
            description="t",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )
    calls: list[str] = []
    await _call(registry, provider=lambda: calls.append("provider"))
    assert seen == {"update_callable": True} and calls == []


async def test_control_counting_context_as_positional_breaks_the_context_only_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: arity that counts the `context` keyword invents an `update` slot."""
    import inspect

    monkeypatch.setattr(
        execute_module,
        "_arity",
        lambda execute, **_: len(inspect.signature(execute).parameters),
    )
    seen: dict[str, Any] = {}

    def execute(tool_call_id: str, args: Any, *, context: Any) -> str:
        seen["context"] = context
        return "ok"

    result = await _call(_define(execute))
    assert result.is_error and seen == {}


# --- L0506-D004-I001: a failing provider is an ordinary per-call execution failure -------------


def _boom() -> ToolExecutionContext:
    raise RuntimeError("context boom")


async def test_a_failing_provider_settles_the_call_as_an_error_result() -> None:
    ran: list[str] = []
    ended: list[tuple[str, bool]] = []

    def execute(tool_call_id: str, args: Any, *, context: Any) -> str:  # pragma: no cover
        ran.append("body")
        return "ok"

    async def on_end(call_id: str, name: str, result: Any) -> None:
        ended.append((call_id, result.is_error))

    ctx = Context()
    declare_tools_events(ctx.events)
    result = await execute_call(
        ToolCallBlock(id="c", name="t", arguments={}),
        registry=_define(execute),
        ctx=ctx,
        on_execution_end=on_end,
        context_provider=_boom,
    )
    assert result.is_error and "context boom" in result.content[0].text  # type: ignore[union-attr]
    assert ran == [] and ended == [("c", True)]


async def test_a_failing_provider_fails_only_its_own_call_in_a_batch() -> None:
    from minion_agent.tools.batch import execute_batch
    from minion_agent.tools.definition import ExecutionMode

    seen: list[str] = []
    attempts: list[int] = []

    def provider() -> ToolExecutionContext:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("context boom")
        return CONTEXT

    def execute(tool_call_id: str, args: Any, *, context: Any) -> str:
        seen.append(tool_call_id)
        return "ok"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="t",
            label="t",
            description="t",
            parameters={"type": "object", "properties": {}},
            execute=execute,
            wants_context=True,
            mode=ExecutionMode.SEQUENTIAL,
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)
    outcome = await execute_batch(
        [
            ToolCallBlock(id="a", name="t", arguments={}),
            ToolCallBlock(id="b", name="t", arguments={}),
        ],
        registry=registry,
        ctx=ctx,
        context_provider=provider,
    )
    assert [r.is_error for r in outcome.results] == [True, False] and seen == ["b"]


async def test_control_a_provider_called_outside_the_boundary_escapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: evaluating the provider before (outside) the execute boundary -- the
    rejected 9ced384e placement -- lets its exception escape `execute_call`."""
    real = execute_module._execute_and_finalize

    async def outside(prepared: Any, *, context_provider: Any = None, **kw: Any) -> Any:
        value = (
            context_provider()
            if context_provider is not None and prepared.definition.wants_context
            else None
        )
        return await real(prepared, context_provider=lambda: value, **kw)

    monkeypatch.setattr(execute_module, "_execute_and_finalize", outside)
    with pytest.raises(RuntimeError, match="context boom"):
        await test_a_failing_provider_settles_the_call_as_an_error_result()


# --- L0506-D004-I002: every execute in ONE batch gets its own then-current snapshot -----------


async def _same_batch(provider_wrapper: Any = None) -> tuple[list[str | None], int]:
    from minion_agent.tools.batch import execute_batch
    from minion_agent.tools.definition import ExecutionMode

    state = {"model": "first"}
    calls: list[int] = []

    def provider() -> ToolExecutionContext:
        calls.append(1)
        return ToolExecutionContext(session_id="s", model=state["model"])

    seen: list[str | None] = []

    def execute(tool_call_id: str, args: Any, *, context: Any) -> str:
        seen.append(context.model)
        state["model"] = "second"  # changes the source between the two calls of this batch
        return "ok"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="t",
            label="t",
            description="t",
            parameters={"type": "object", "properties": {}},
            execute=execute,
            wants_context=True,
            mode=ExecutionMode.SEQUENTIAL,
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)
    await execute_batch(
        [
            ToolCallBlock(id="a", name="t", arguments={}),
            ToolCallBlock(id="b", name="t", arguments={}),
        ],
        registry=registry,
        ctx=ctx,
        context_provider=provider if provider_wrapper is None else provider_wrapper(provider),
    )
    return seen, len(calls)


async def test_each_call_in_one_sequential_batch_gets_its_own_snapshot() -> None:
    assert await _same_batch() == (["first", "second"], 2)


async def test_control_a_once_per_batch_capture_fails_the_same_batch_witness() -> None:
    def once_per_batch(provider: Any) -> Any:
        captured: list[ToolExecutionContext] = []

        def lazy() -> ToolExecutionContext:
            if not captured:
                captured.append(provider())
            return captured[0]

        return lazy

    assert await _same_batch(once_per_batch) == (["first", "first"], 1)
