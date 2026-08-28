"""Inputs carry provenance; runs carry their causal inputs."""

from typing import Any

from minion_agent.agent.envelope import ClaimPolicy
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import TextBlock, UserMessage
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind, EventName

from .test_single_turn import _loop


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _causes(loop: AgentLoop, kind: EventName = EventKind.AGENT_START) -> list[list[dict[str, Any]]]:
    return [event.data["causes"] for event in loop.instance.log.events if event.kind == kind]


async def test_a_run_records_the_origin_of_its_input() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"), origin={"channel": "matrix"})

    await loop.run_until_idle()

    assert _causes(loop)[0][0]["origin"] == {"channel": "matrix"}


async def test_claim_all_gives_one_run_several_causes() -> None:
    """The case a singular run.origin could not express."""
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.next_turn_policy = ClaimPolicy.ALL
    loop.instance.inbox.followup(_say("one"), origin="a")
    loop.instance.inbox.followup(_say("two"), origin="b")
    loop.instance.inbox.followup(_say("three"), origin="c")

    await loop.run_until_idle()

    causes = _causes(loop)
    assert len(causes) == 1
    assert [cause["origin"] for cause in causes[0]] == ["a", "b", "c"]


async def test_one_at_a_time_accumulates_causes_onto_the_same_run() -> None:
    """Layer 08, PASS 2: mid-run follow-up continuation means a second queued
    followup found once the first turn's inner loop exhausts keeps the *same*
    run going (pinned pi's own outer `runLoop` loop) rather than starting a
    fresh `AGENT_START`/`AGENT_END` pair -- one_at_a_time claims one followup
    per continuation, but both still land on one run's own causes."""
    loop = _loop(ScriptedResponse((), StopReason.STOP), ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("one"), origin="a")
    loop.instance.inbox.followup(_say("two"), origin="b")

    await loop.run_until_idle()

    causes = _causes(loop)
    assert [[c["origin"] for c in run] for run in causes] == [["a", "b"]]


async def test_causes_survive_to_agent_end() -> None:
    """A consumer reading only completions can still route the result."""
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"), origin={"room": "!abc"})

    await loop.run_until_idle()

    assert _causes(loop, EventKind.AGENT_END)[0][0]["origin"] == {"room": "!abc"}


async def test_envelope_ids_are_carried_alongside_origins() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    envelope = loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert _causes(loop)[0][0]["id"] == envelope.id


async def test_a_run_with_no_origin_still_records_its_cause() -> None:
    """A proactive run has provenance even when the origin is null."""
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("scheduled work"))

    await loop.run_until_idle()

    assert _causes(loop)[0][0]["origin"] is None
