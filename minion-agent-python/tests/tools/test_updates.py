"""Tools stream partial results. Partial output is live, never logged.

Section 5's rule is "model-visible means logged". A partial result is not
model-visible -- only the finalized result becomes a message -- so logging
every intermediate chunk would inflate the log with content no request can
ever contain, and reconstruction would have to learn to ignore it.
"""

from typing import Any

import pytest

from minion_agent.runtime import Context
from minion_agent.runtime.events import EventBus
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_UPDATE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

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


async def test_a_tool_may_report_partial_output() -> None:
    ctx = _ctx()
    seen: list[tuple[str, str]] = []
    ctx.events.on(
        TOOLS_UPDATE,
        lambda call_id, tool_name, arguments, partial: seen.append((call_id, partial)),
    )

    def chatty(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update("half")
        update("most")
        return "all"

    result = await execute_call(_call(), registry=_streaming(chatty), ctx=ctx)

    assert seen == [("t1", "half"), ("t1", "most")]
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
        update("x")
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
        update("x")
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
        update("nobody is listening")
        return "done"

    result = await execute_call(_call(), registry=_streaming(chatty), ctx=ctx)

    update_emissions = [entry for entry in emitted if entry[0] == TOOLS_UPDATE]
    assert update_emissions == [(TOOLS_UPDATE, ("t1", "echo", {}, "nobody is listening"))]
    assert not result.is_error


async def test_a_late_update_after_the_tool_settles_is_ignored() -> None:
    """Pinned Pi's `AgentToolUpdateCallback`: "Calls made after the tool promise settles are
    ignored." A tool that stashes its own `update` callback and calls it again after `execute()`
    has already returned must not produce a second `tools/update` emission (`TOOL-017`)."""
    ctx = _ctx()
    seen: list[str] = []
    ctx.events.on(
        TOOLS_UPDATE, lambda call_id, tool_name, arguments, partial: seen.append(partial)
    )
    stashed: list[Any] = []

    def stash_then_settle(tool_call_id: str, args: dict[str, Any], update: Any) -> str:
        update("live")
        stashed.append(update)
        return "done"

    result = await execute_call(_call(), registry=_streaming(stash_then_settle), ctx=ctx)
    stashed[0]("late")

    assert seen == ["live"]
    assert not result.is_error
