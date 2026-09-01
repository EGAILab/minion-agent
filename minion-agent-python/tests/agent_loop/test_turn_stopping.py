"""Stopping decisions, and what cannot be overridden."""

from typing import Any

from minion_agent.agent.decisions import TurnStopping
from minion_agent.agent.events import AGENT_TURN_STOPPING
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
    return len([e for e in loop.instance.log.events if e.kind == EventKind.TURN_START])


async def test_a_stop_decision_ends_the_turn_early() -> None:
    loop = _loop(_tool_call(), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda tool_call_id, args: "done")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.STOP)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert _steps(loop) == 1


async def test_no_opinion_continues() -> None:
    loop = _loop(_tool_call(), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda tool_call_id, args: "done")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.NO_OPINION)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert _steps(loop) == 2


async def test_no_listeners_continues() -> None:
    loop = _loop(_tool_call(), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda tool_call_id, args: "done")
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert _steps(loop) == 2


async def test_a_stopped_turn_records_why() -> None:
    loop = _loop(_tool_call(), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda tool_call_id, args: "done")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.STOP)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "stopped"


async def test_the_event_is_still_dispatched_when_nothing_is_owed() -> None:
    """Layer 08, PASS 2: pinned pi's own `shouldStopAfterTurn` runs after
    every turn, including a plain text reply with no tool calls at all --
    confirmed directly against `runLoop` (only an error/aborted `stopReason`
    skips it, an early exit this scenario never reaches). A listener
    answering CONTINUE is asked, and still cannot manufacture a second
    request on its own: with nothing else pending, the run still ends after
    one turn. An earlier revision of this test claimed the event was never
    dispatched here at all -- that was never actually true of pinned Pi."""
    loop = _loop(ScriptedResponse((TextBlock(text="done"),), StopReason.STOP))
    asked: list[str] = []

    def record(*args: Any) -> TurnStopping:
        asked.append("asked")
        return TurnStopping.CONTINUE

    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, record)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert asked == ["asked"]
