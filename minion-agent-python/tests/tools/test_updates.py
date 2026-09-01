"""Tools stream partial results. Partial output is live, never logged.

Section 5's rule is "model-visible means logged". A partial result is not
model-visible -- only the finalized result becomes a message -- so logging
every intermediate chunk would inflate the log with content no request can
ever contain, and reconstruction would have to learn to ignore it.
"""

import asyncio
from typing import Any

import pytest

from minion_agent.runtime import Context
from minion_agent.runtime.events import EventBus
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_EXECUTION_END, TOOLS_UPDATE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult, text_result

from .test_execute import _call


def _ctx() -> Context:
    ctx = Context()
    declare_tools_events(ctx.events)
    return ctx


def _streaming(execute: Any) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="d",
            parameters={"type": "object", "properties": {}},
            execute=execute,
            label="Echo",
        )
    )
    return registry


def _partial(text: str) -> ToolResult:
    """A minimal structured partial result -- matching pinned Pi's own `AgentToolResult<T>`
    (`L08-R011`): `tool_call_id`/`tool_name` here are placeholders, since `update()`'s own closure
    normalizes both to the real call's own id/name, the same way it already does for the final
    result (a tool need not stamp its own call's identity onto a partial)."""
    return text_result("ignored", text, "ignored")


async def test_a_tool_may_report_partial_output() -> None:
    ctx = _ctx()
    seen: list[tuple[str, ToolResult]] = []
    ctx.events.on(
        TOOLS_UPDATE,
        lambda call_id, tool_name, arguments, partial: seen.append((call_id, partial)),
    )

    def chatty(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update(_partial("half"))
        update(_partial("most"))
        return "all"

    result = await execute_call(_call(), registry=_streaming(chatty), ctx=ctx)

    assert seen == [
        ("t1", text_result("t1", "half", "echo")),
        ("t1", text_result("t1", "most", "echo")),
    ]
    assert result.content


async def test_a_tool_without_an_update_parameter_is_called_with_two_arguments() -> None:
    """Declaring the callback is opt-in; most tools have nothing to stream.

    `quiet` accepts only `(tool_call_id, args)`. If `_wants_update` mistakenly decided
    to pass the callback anyway, the call would raise a `TypeError` that
    `execute_call` converts into an error result -- `quiet` would never run,
    `seen` would stay empty, and the result would report an error. Both
    assertions below fail in that scenario, so this pins the two-argument
    call shape rather than merely that the tool ran.
    """
    ctx = _ctx()
    seen: list[int] = []

    def quiet(tool_call_id: str, args: dict[str, Any]) -> str:
        seen.append(1)
        return "done"

    result = await execute_call(_call(), registry=_streaming(quiet), ctx=ctx)

    assert seen == [1]
    assert not result.is_error


async def test_updates_carry_the_call_id_so_a_consumer_can_route_them() -> None:
    ctx = _ctx()
    seen: list[str] = []
    ctx.events.on(
        TOOLS_UPDATE, lambda call_id, tool_name, arguments, partial: seen.append(call_id)
    )

    def chatty(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update(_partial("x"))
        return "done"

    await execute_call(_call(), registry=_streaming(chatty), ctx=ctx)

    assert seen == ["t1"]


async def test_updates_carry_the_tool_name_and_original_arguments() -> None:
    """`IR-L06-005`: pinned Pi's `tool_execution_update` event carries `toolCallId`/`toolName`/
    `args`/`partialResult` -- Minion's earlier payload carried only the first and last, exposing
    strictly less than Pi's own live event stream for no stated architectural reason. `args` is
    the ORIGINAL, pre-`prepare_arguments`/validation arguments (pinned Pi's own
    `PreparedToolCall.toolCall.arguments`, not the validated ones `execute()` actually runs
    with)."""
    ctx = _ctx()
    seen: list[tuple[str, dict[str, Any]]] = []
    ctx.events.on(
        TOOLS_UPDATE,
        lambda call_id, tool_name, arguments, partial: seen.append((tool_name, arguments)),
    )

    def chatty(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update(_partial("x"))
        return "done"

    await execute_call(_call(note="original"), registry=_streaming(chatty), ctx=ctx)

    assert seen == [("echo", {"note": "original"})]


async def test_no_listener_makes_updates_harmless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling `update()` with nobody listening must not disturb the tool.

    Asserting only `not result.is_error` would also pass if `update` were a
    silent no-op that never touched the event bus at all -- that reads as
    "harmless" for the wrong reason. To distinguish the two, this spies on
    `EventBus.emit` itself (with zero listeners registered, so the "no
    listener" premise still holds) and requires that the emission actually
    happened, in addition to the tool completing without error.
    """
    ctx = _ctx()
    emitted: list[tuple[str, tuple[Any, ...]]] = []
    original_emit = EventBus.emit

    def spy_emit(self: EventBus, name: str, *args: Any, scope: Any = None) -> None:
        emitted.append((name, args))
        original_emit(self, name, *args, scope=scope)

    monkeypatch.setattr(EventBus, "emit", spy_emit)

    def chatty(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update(_partial("nobody is listening"))
        return "done"

    result = await execute_call(_call(), registry=_streaming(chatty), ctx=ctx)

    update_emissions = [entry for entry in emitted if entry[0] == TOOLS_UPDATE]
    assert update_emissions == [
        (TOOLS_UPDATE, ("t1", "echo", {}, text_result("t1", "nobody is listening", "echo")))
    ]
    assert not result.is_error


async def test_on_execution_update_is_awaited_for_every_call_in_order() -> None:
    """`L08-R002`, PASS 7: `on_execution_update` (the Layer-08-facing live-dispatch hook,
    additive alongside the certified `tools/update` EMIT above) receives every update, in the
    order the tool made them, joined before `execute_call` returns -- pinned Pi's own `await
    Promise.all(updateEvents)` (`agent-loop.ts:670-711`)."""
    ctx = _ctx()
    seen: list[tuple[str, str, dict[str, Any], ToolResult]] = []

    async def on_execution_update(
        call_id: str, tool_name: str, arguments: dict[str, Any], partial: ToolResult
    ) -> None:
        seen.append((call_id, tool_name, arguments, partial))

    def chatty(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update(_partial("half"))
        update(_partial("most"))
        return "all"

    result = await execute_call(
        _call(), registry=_streaming(chatty), ctx=ctx, on_execution_update=on_execution_update
    )

    assert seen == [
        ("t1", "echo", {}, text_result("t1", "half", "echo")),
        ("t1", "echo", {}, text_result("t1", "most", "echo")),
    ]
    assert result.content


async def test_a_failing_on_execution_update_listener_propagates_uncaught() -> None:
    """`L08-R002`, PASS 7: pinned Pi's own `tool_execution_update` dispatch lets a listener's own
    rejection propagate straight out, uncaught -- NOT silently converted into a per-call tool
    error result the way an `execute()` exception is. `tools/execution-end` never fires for this
    call at all: `_finalize`/finalization is only reached once every pending update has been
    joined without error."""
    ctx = _ctx()
    end_emissions: list[Any] = []
    ctx.events.on(TOOLS_EXECUTION_END, lambda *args: end_emissions.append(args))

    async def boom_on_update(
        call_id: str, tool_name: str, arguments: dict[str, Any], partial: ToolResult
    ) -> None:
        raise RuntimeError("update listener exploded")

    def chatty(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update(_partial("x"))
        return "done"

    with pytest.raises(RuntimeError, match="update listener exploded"):
        await execute_call(
            _call(), registry=_streaming(chatty), ctx=ctx, on_execution_update=boom_on_update
        )

    assert end_emissions == []


async def test_on_execution_update_starts_synchronously_but_does_not_block_the_tool() -> None:
    """`L08-R002`, PASS 8: `update(partial)` starts the hook's own coroutine SYNCHRONOUSLY, in the
    same call stack, up to its own first genuine suspension point -- `asyncio.eager_task_factory`,
    not `ensure_future`/`create_task` (tried in PASS 7: a plain `Task` only ever schedules its
    first step through `loop.call_soon`, deferred to the next event-loop iteration, so nothing of
    the hook had run yet by the time `update()` returned -- observably wrong against pinned Pi's
    own `agent-loop.ts:670-711`, which runs a JS `async function` synchronously up to ITS own first
    suspension before returning control to `update()`'s own caller). This is pinned Pi's own exact
    three-step interleaving, reproduced empirically: `listener-entered` (before the hook's own
    suspension), `tool-continued` (the tool's own synchronous work after calling `update()`), then
    `listener-resumed` (once the hook's own suspended `await` settles) -- not
    `tool-continued, listener-entered, listener-resumed`, which is what a merely-scheduled
    (`ensure_future`) hook would have produced instead."""
    ctx = _ctx()
    order: list[str] = []

    async def slow_on_execution_update(
        call_id: str, tool_name: str, arguments: dict[str, Any], partial: ToolResult
    ) -> None:
        order.append("listener-entered")
        await asyncio.sleep(0)
        order.append("listener-resumed")

    def chatty(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update(_partial("x"))
        order.append("tool-continued")
        return "done"

    await execute_call(
        _call(), registry=_streaming(chatty), ctx=ctx, on_execution_update=slow_on_execution_update
    )

    assert order == ["listener-entered", "tool-continued", "listener-resumed"]


async def test_a_late_update_after_the_tool_settles_is_ignored() -> None:
    """Pinned Pi's `AgentToolUpdateCallback`: "Calls made after the tool promise settles are
    ignored." A tool that stashes its own `update` callback and calls it again after `execute()`
    has already returned must not produce a second `tools/update` emission (`TOOL-017`)."""
    ctx = _ctx()
    seen: list[ToolResult] = []
    ctx.events.on(
        TOOLS_UPDATE, lambda call_id, tool_name, arguments, partial: seen.append(partial)
    )
    stashed: list[Any] = []

    def stash_then_settle(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update(_partial("live"))
        stashed.append(update)
        return "done"

    result = await execute_call(_call(), registry=_streaming(stash_then_settle), ctx=ctx)
    stashed[0](_partial("late"))

    assert seen == [text_result("t1", "live", "echo")]
    assert not result.is_error
