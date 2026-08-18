"""Properties of the loop over any number of turns."""

from itertools import pairwise

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from minion_agent.agent.identity import AgentStatus
from minion_agent.llm import TextBlock, UserMessage
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind

from .test_single_turn import _loop

turn_counts = st.integers(min_value=1, max_value=6)


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


@given(turn_counts)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
async def test_status_always_alternates(turns: int) -> None:
    """A transition signal must never report the same state twice in a row."""
    loop = _loop(*[ScriptedResponse((), StopReason.STOP) for _ in range(turns)])
    seen: list[AgentStatus] = []
    loop.instance.on_status_change = seen.append

    for index in range(turns):
        loop.instance.inbox.followup(_say(f"turn {index}"))
    await loop.run_until_idle()

    assert all(a is not b for a, b in pairwise(seen))
    assert seen[0] is AgentStatus.RUNNING
    assert seen[-1] is AgentStatus.IDLE


@given(turn_counts)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
async def test_every_turn_is_bracketed(turns: int) -> None:
    loop = _loop(*[ScriptedResponse((), StopReason.STOP) for _ in range(turns)])
    for index in range(turns):
        loop.instance.inbox.followup(_say(f"turn {index}"))

    await loop.run_until_idle()

    kinds = [e.kind for e in loop.instance.log.events]
    assert kinds.count(EventKind.TURN_START) == kinds.count(EventKind.TURN_END) == turns


@given(turn_counts)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
async def test_a_turn_records_exactly_the_causes_it_claimed(turns: int) -> None:
    loop = _loop(*[ScriptedResponse((), StopReason.STOP) for _ in range(turns)])
    sent = [
        loop.instance.inbox.followup(_say(f"turn {index}"), origin=index).id
        for index in range(turns)
    ]

    await loop.run_until_idle()

    recorded = [
        cause["id"]
        for event in loop.instance.log.events
        if event.kind == EventKind.TURN_START
        for cause in event.data["causes"]
    ]
    assert recorded == sent
