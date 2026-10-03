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
