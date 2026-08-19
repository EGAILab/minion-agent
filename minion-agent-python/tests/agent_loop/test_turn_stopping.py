"""Stopping decisions, and what cannot be overridden."""

from typing import Any

from minion_agent.agent.decisions import TurnStopping
from minion_agent.agent.events import AGENT_TURN_STOPPING
from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import TextBlock, ToolCallBlock, UserMessage
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind

from .test_single_turn import _loop, _register


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _tool_call() -> ScriptedResponse:
    return ScriptedResponse(
        (ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE
    )


def _steps(loop: AgentLoop) -> int:
    return len([e for e in loop.instance.log.events if e.kind == EventKind.STEP_START])


async def test_a_stop_decision_ends_the_turn_early() -> None:
    loop = _loop(_tool_call(), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda args: "done")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.STOP)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert _steps(loop) == 1


async def test_no_opinion_continues() -> None:
    loop = _loop(_tool_call(), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda args: "done")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.NO_OPINION)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert _steps(loop) == 2


async def test_no_listeners_continues() -> None:
    loop = _loop(_tool_call(), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda args: "done")
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert _steps(loop) == 2


async def test_a_stopped_turn_records_why() -> None:
    loop = _loop(_tool_call(), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda args: "done")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.STOP)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.TURN_END)
    assert end.data["reason"] == "stopped"


async def test_the_event_is_not_dispatched_when_nothing_is_owed() -> None:
    """Hard termination precedes the decision: a turn the loop was already
    going to end never asks."""
    loop = _loop(ScriptedResponse((TextBlock(text="done"),), StopReason.STOP))
    asked: list[str] = []

    def record(*args: Any) -> TurnStopping:
        asked.append("asked")
        return TurnStopping.CONTINUE

    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, record)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert asked == []


async def test_continue_cannot_override_max_steps() -> None:
    """max_steps is a loop invariant; a plugin must not be able to defeat it."""
    loop = _loop(*[_tool_call() for _ in range(10)])
    loop.instance.definition = AgentDefinition(
        name="ada", model=loop.instance.definition.model, system="", max_steps=2
    )
    _register(loop, "echo", lambda args: "again")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.CONTINUE)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert _steps(loop) == 2
