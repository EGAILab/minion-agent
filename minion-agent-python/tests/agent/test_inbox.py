"""The inbox carries provenance and claims by policy."""

import pytest

from minion_agent.agent.envelope import ClaimPolicy, InboxTarget
from minion_agent.agent.inbox import Inbox, NotJsonSafeOriginError
from minion_agent.llm import AssistantMessage, TextBlock, ToolResultMessage, UserMessage


def _message(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def test_followup_queues_for_the_next_turn_and_wakes() -> None:
    inbox = Inbox()

    inbox.followup(_message("hello"))

    assert len(inbox.pending(InboxTarget.NEXT_TURN)) == 1
    assert inbox.wake_requested


def test_steer_queues_for_the_next_step_and_wakes() -> None:
    inbox = Inbox()

    inbox.steer(_message("actually, stop"))

    assert len(inbox.pending(InboxTarget.NEXT_STEP)) == 1
    assert inbox.wake_requested


def test_inject_queues_for_the_next_step_without_waking() -> None:
    """Silent context: it rides along with the next thing that does wake."""
    inbox = Inbox()

    inbox.inject(_message("file changed on disk"))

    assert len(inbox.pending(InboxTarget.NEXT_STEP)) == 1
    assert not inbox.wake_requested


def test_taking_the_wake_signal_clears_it() -> None:
    inbox = Inbox()
    inbox.followup(_message("hello"))

    assert inbox.take_wake()
    assert not inbox.wake_requested
    assert not inbox.take_wake()


def test_one_at_a_time_claims_only_the_oldest() -> None:
    inbox = Inbox()
    inbox.followup(_message("first"))
    inbox.followup(_message("second"))

    claimed = inbox.claim(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    assert len(claimed) == 1
    assert len(inbox.pending(InboxTarget.NEXT_TURN)) == 1


def test_all_claims_everything_queued() -> None:
    inbox = Inbox()
    inbox.followup(_message("first"))
    inbox.followup(_message("second"))

    claimed = inbox.claim(InboxTarget.NEXT_TURN, ClaimPolicy.ALL)

    assert len(claimed) == 2
    assert inbox.pending(InboxTarget.NEXT_TURN) == ()


def test_claiming_removes_what_it_claimed() -> None:
    inbox = Inbox()
    inbox.followup(_message("only"))

    inbox.claim(InboxTarget.NEXT_TURN, ClaimPolicy.ALL)

    assert inbox.claim(InboxTarget.NEXT_TURN, ClaimPolicy.ALL) == ()


def test_the_two_queues_are_independent() -> None:
    inbox = Inbox()
    inbox.followup(_message("turn"))
    inbox.steer(_message("step"))

    claimed = inbox.claim(InboxTarget.NEXT_STEP, ClaimPolicy.ALL)

    assert len(claimed) == 1
    assert len(inbox.pending(InboxTarget.NEXT_TURN)) == 1


def test_every_envelope_gets_a_unique_id() -> None:
    inbox = Inbox()

    first = inbox.followup(_message("a"))
    second = inbox.followup(_message("b"))

    assert first.id != second.id


def test_origin_is_carried_verbatim() -> None:
    inbox = Inbox()
    origin = {"channel": "matrix", "room": "!abc:example.org"}

    envelope = inbox.followup(_message("hello"), origin=origin)

    assert envelope.origin == origin


def test_origin_defaults_to_none() -> None:
    assert Inbox().followup(_message("hello")).origin is None


def test_a_non_json_safe_origin_is_rejected_eagerly() -> None:
    """Origin travels in the log and must survive another language."""
    inbox = Inbox()

    with pytest.raises(NotJsonSafeOriginError, match="JSON-safe"):
        inbox.followup(_message("hello"), origin=object())  # type: ignore[arg-type]

    assert inbox.pending(InboxTarget.NEXT_TURN) == ()


def test_nested_json_structures_are_walked() -> None:
    """Validation is structural, not shallow: a bad value hiding inside a
    list or a nested object still fails before anything is stored."""
    inbox = Inbox()

    inbox.followup(_message("ok"), origin={"path": ["a", {"b": [1, 2.5, True, None]}]})

    with pytest.raises(NotJsonSafeOriginError, match=r"origin\.outer\[1\]"):
        inbox.followup(_message("bad"), origin={"outer": ["fine", object()]})

    assert len(inbox.pending(InboxTarget.NEXT_TURN)) == 1


def test_a_non_string_mapping_key_is_rejected() -> None:
    """JSON object keys are strings; an integer key would not survive a
    round trip through the log."""
    inbox = Inbox()

    with pytest.raises(NotJsonSafeOriginError, match="keys must be strings"):
        inbox.followup(_message("bad"), origin={1: "one"})  # type: ignore[dict-item]


# -- AG-011 (L07-R002): the accepted domain is pinned Pi's whole `Message`
# union (`UserMessage | AssistantMessage | ToolResultMessage`), not `UserMessage`
# alone. `CustomAgentMessages` is empty in pinned Pi itself, so `Message` -- the
# already-certified Layer-02 vocabulary -- is the actual, complete domain.


def _assistant_message(text: str) -> AssistantMessage:
    from minion_agent.llm import StopReason, Usage

    return AssistantMessage(
        content=(TextBlock(text=text),),
        stop_reason=StopReason.STOP,
        usage=Usage(),
        model="mock-1",
        provider="mock",
        timestamp=1,
    )


def _tool_result_message(text: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="t1",
        content=(TextBlock(text=text),),
        timestamp=1,
        tool_name="tool",
    )


def test_steer_accepts_an_assistant_message() -> None:
    inbox = Inbox()

    inbox.steer(_assistant_message("assistant steering"))

    assert len(inbox.pending(InboxTarget.NEXT_STEP)) == 1


def test_followup_accepts_an_assistant_message() -> None:
    inbox = Inbox()

    inbox.followup(_assistant_message("assistant follow-up"))

    assert len(inbox.pending(InboxTarget.NEXT_TURN)) == 1


def test_steer_accepts_a_tool_result_message() -> None:
    inbox = Inbox()

    inbox.steer(_tool_result_message("tool output"))

    assert len(inbox.pending(InboxTarget.NEXT_STEP)) == 1


def test_claim_returns_mixed_message_variants_in_fifo_order() -> None:
    inbox = Inbox()
    inbox.followup(_message("user"))
    inbox.followup(_assistant_message("assistant"))
    inbox.followup(_tool_result_message("tool"))

    claimed = inbox.claim(InboxTarget.NEXT_TURN, ClaimPolicy.ALL)

    assert [envelope.message for envelope in claimed] == [
        _message("user"),
        _assistant_message("assistant"),
        _tool_result_message("tool"),
    ]


def test_has_pending_is_false_for_an_empty_inbox() -> None:
    """Pi's `hasQueuedMessages()`: true when EITHER queue has items."""
    assert not Inbox().has_pending()


def test_has_pending_is_true_with_only_a_next_turn_item() -> None:
    inbox = Inbox()
    inbox.followup(_message("hello"))

    assert inbox.has_pending()


def test_has_pending_is_true_with_only_a_next_step_item() -> None:
    inbox = Inbox()
    inbox.steer(_message("actually, stop"))

    assert inbox.has_pending()


def test_clearing_one_target_leaves_the_other_untouched() -> None:
    """Pi's `clearSteeringQueue()`/`clearFollowUpQueue()`: each clears exactly
    its own queue, not the other."""
    inbox = Inbox()
    inbox.followup(_message("turn"))
    inbox.steer(_message("step"))

    inbox.clear(InboxTarget.NEXT_STEP)

    assert inbox.pending(InboxTarget.NEXT_STEP) == ()
    assert len(inbox.pending(InboxTarget.NEXT_TURN)) == 1


def test_clearing_an_empty_target_is_a_harmless_no_op() -> None:
    inbox = Inbox()

    inbox.clear(InboxTarget.NEXT_TURN)

    assert inbox.pending(InboxTarget.NEXT_TURN) == ()


def test_clear_all_empties_both_queues() -> None:
    """Pi's `clearAllQueues()`: both queues, in one call."""
    inbox = Inbox()
    inbox.followup(_message("turn"))
    inbox.steer(_message("step"))

    inbox.clear_all()

    assert not inbox.has_pending()
    assert inbox.pending(InboxTarget.NEXT_TURN) == ()
    assert inbox.pending(InboxTarget.NEXT_STEP) == ()


def test_clearing_does_not_affect_the_wake_signal() -> None:
    """Clearing removes queued content; it is not itself a settle signal --
    only `take_wake()` (driven by the run loop, Layer 08) consumes that."""
    inbox = Inbox()
    inbox.followup(_message("hello"))

    inbox.clear_all()

    assert inbox.wake_requested
