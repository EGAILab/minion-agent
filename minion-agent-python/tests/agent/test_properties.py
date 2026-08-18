"""Properties that must hold for any inbox and turn sequence."""

from hypothesis import given
from hypothesis import strategies as st

from minion_agent.agent.envelope import ClaimPolicy, InboxTarget
from minion_agent.agent.inbox import Inbox
from minion_agent.llm import TextBlock, UserMessage

texts = st.lists(st.text(min_size=1, max_size=6), min_size=0, max_size=20)
policies = st.sampled_from([ClaimPolicy.ALL, ClaimPolicy.ONE_AT_A_TIME])


def _message(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _drain(inbox: Inbox, target: InboxTarget, policy: ClaimPolicy) -> list[str]:
    seen: list[str] = []
    while True:
        claimed = inbox.claim(target, policy)
        if not claimed:
            return seen
        seen.extend(envelope.id for envelope in claimed)


@given(texts, policies)
def test_every_sent_message_is_claimed_exactly_once(items: list[str], policy: ClaimPolicy) -> None:
    inbox = Inbox()
    sent = [inbox.followup(_message(text)).id for text in items]

    drained = _drain(inbox, InboxTarget.NEXT_TURN, policy)

    assert drained == sent


@given(texts, texts)
def test_the_two_queues_never_leak_into_each_other(
    turn_items: list[str], step_items: list[str]
) -> None:
    inbox = Inbox()
    turn_ids = {inbox.followup(_message(t)).id for t in turn_items}
    step_ids = {inbox.steer(_message(t)).id for t in step_items}

    drained = set(_drain(inbox, InboxTarget.NEXT_TURN, ClaimPolicy.ALL))

    assert drained == turn_ids
    assert not (drained & step_ids)


@given(texts)
def test_one_at_a_time_preserves_send_order(items: list[str]) -> None:
    inbox = Inbox()
    sent = [inbox.followup(_message(text)).id for text in items]

    drained = _drain(inbox, InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    assert drained == sent


@given(texts)
def test_claiming_never_invents_input(items: list[str]) -> None:
    inbox = Inbox()
    sent = {inbox.followup(_message(text)).id for text in items}

    drained = set(_drain(inbox, InboxTarget.NEXT_TURN, ClaimPolicy.ALL))

    assert drained <= sent


@given(texts)
def test_injection_alone_never_requests_a_wake(items: list[str]) -> None:
    """Silent by construction, however much is injected."""
    inbox = Inbox()
    for text in items:
        inbox.inject(_message(text))

    assert not inbox.wake_requested


@given(st.integers(min_value=1, max_value=8))
def test_a_drained_inbox_stays_drained(rounds: int) -> None:
    inbox = Inbox()
    inbox.followup(_message("only"))
    _drain(inbox, InboxTarget.NEXT_TURN, ClaimPolicy.ALL)

    for _ in range(rounds):
        assert inbox.claim(InboxTarget.NEXT_TURN, ClaimPolicy.ALL) == ()
