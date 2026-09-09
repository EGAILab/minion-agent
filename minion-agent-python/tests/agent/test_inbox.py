"""The inbox carries provenance and claims by policy."""

import pytest

from minion_agent.agent.envelope import ClaimPolicy, InboxTarget
from minion_agent.agent.inbox import Inbox, NotJsonSafeOriginError
from minion_agent.llm import AssistantMessage, TextBlock, ToolResultMessage, UserMessage, text_of


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


# -- Layer 09, `L09-R012`/`L09-R013`/`L09-R014`: `_reserve()`/`_Reservation` replace peek/commit --


def test_reserve_atomically_claims_the_selected_batch() -> None:
    inbox = Inbox()
    envelope = inbox.followup(_message("only"))

    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    assert reservation.envelopes == (envelope,)
    assert inbox.pending(InboxTarget.NEXT_TURN) == ()  # already removed -- no reentrancy window


def test_reserve_matches_claim_for_all_policy() -> None:
    inbox = Inbox()
    inbox.followup(_message("A"))
    inbox.followup(_message("B"))

    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ALL)

    assert [text_of(e.message) for e in reservation.envelopes] == ["A", "B"]
    assert inbox.pending(InboxTarget.NEXT_TURN) == ()


def test_reserve_on_an_empty_target_returns_an_empty_reservation() -> None:
    inbox = Inbox()

    assert inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME).envelopes == ()
    assert inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ALL).envelopes == ()


def test_commit_leaves_the_reserved_batch_removed() -> None:
    inbox = Inbox()
    inbox.followup(_message("A"))
    inbox.followup(_message("B"))
    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ALL)

    reservation.commit()

    assert inbox.pending(InboxTarget.NEXT_TURN) == ()


def test_rollback_restores_the_exact_reserved_batch() -> None:
    inbox = Inbox()
    envelope = inbox.followup(_message("only"))
    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    reservation.rollback()

    assert inbox.pending(InboxTarget.NEXT_TURN) == (envelope,)


def test_a_reservation_never_settled_leaves_the_batch_removed() -> None:
    """The whole point of a reservation: it is CREATED by an atomic claim(), so a caller that
    obtains one and never calls either `.commit()`/`.rollback()` has already had the effect of a
    plain `claim()` -- there is no "pending, unremoved" state to accidentally leave behind."""
    inbox = Inbox()
    inbox.followup(_message("only"))

    inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)
    # ... caller never settles the reservation ...

    assert inbox.pending(InboxTarget.NEXT_TURN) == ()


def test_the_old_public_restore_method_no_longer_exists() -> None:
    """`L09-R010`: an independent Rust review found the PASS-5 candidate's public `Inbox.restore
    (target, envelopes)` callable by ANY caller with ANY envelope tuple -- including one still
    queued and never claimed, or the same envelope repeatedly -- manufacturing duplicate queue
    entries that shared an id, contradicting this row's own exactly-once invariant
    (`CONTRACT_ASSURANCE_DEFECT`). Remediation removes the method entirely: `Inbox.claim()`
    remains the sole PUBLIC removal operation; `_reserve()`/`_Reservation` (private, `L09-R012`
    convergence) replace it internally."""
    inbox = Inbox()
    assert not hasattr(inbox, "restore")
    assert not hasattr(inbox, "peek")  # PASS-6's own now-superseded method is also gone


def test_the_reviewers_duplicate_id_witness_is_no_longer_expressible() -> None:
    """The exact discriminating witness the independent review executed against the PASS-5
    candidate: `inbox.restore(target, (envelope,))` on an envelope that was never claimed
    produced two entries sharing the same id; calling it again produced three. That attack is no
    longer expressible through any public `Inbox` operation at all."""
    inbox = Inbox()
    envelope = inbox.followup(_message("A"))

    with pytest.raises(AttributeError):
        inbox.restore(InboxTarget.NEXT_TURN, (envelope,))  # type: ignore[attr-defined]

    assert [item.id for item in inbox.pending(InboxTarget.NEXT_TURN)] == [envelope.id]


def test_double_rollback_cannot_duplicate_an_envelope() -> None:
    """`C09-1`'s own required negative witness 1: a second terminal call, whichever method,
    raises rather than mutating the queue again."""
    inbox = Inbox()
    inbox.followup(_message("only"))
    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    reservation.rollback()
    with pytest.raises(RuntimeError):
        reservation.rollback()

    assert len(inbox.pending(InboxTarget.NEXT_TURN)) == 1  # not duplicated


def test_rollback_cannot_accept_a_foreign_envelope() -> None:
    """`C09-1`'s own required negative witness 2: neither terminal method accepts an argument at
    all -- a structural guarantee (Python's own function-signature enforcement), not merely a
    behavioral one."""
    inbox = Inbox()
    envelope = inbox.followup(_message("only"))
    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    with pytest.raises(TypeError):
        reservation.rollback(envelope)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        reservation.commit(envelope)  # type: ignore[call-arg]


def test_commit_then_rollback_cannot_mutate_the_queue_a_second_time() -> None:
    """`C09-1`'s own required negative witness 3."""
    inbox = Inbox()
    inbox.followup(_message("only"))
    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    reservation.commit()
    with pytest.raises(RuntimeError):
        reservation.rollback()

    assert inbox.pending(InboxTarget.NEXT_TURN) == ()  # still removed -- rollback was rejected


def test_rollback_then_commit_cannot_mutate_the_queue_a_second_time() -> None:
    """`C09-1`'s own required negative witness 4 (the reverse ordering)."""
    inbox = Inbox()
    inbox.followup(_message("only"))
    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    reservation.rollback()
    with pytest.raises(RuntimeError):
        reservation.commit()

    assert len(inbox.pending(InboxTarget.NEXT_TURN)) == 1  # still restored -- commit was rejected


def test_rollback_precedes_input_enqueued_after_the_reservation() -> None:
    """The restored batch goes to the FRONT, ahead of anything enqueued in the meantime -- FIFO
    order as if the reservation had never happened. Because the batch is already gone from the
    queue the moment it is reserved, a re-entrant observer's own `steer()`/`followup()` calls
    always land AFTER it in the underlying list; rollback simply re-prepends the original batch."""
    inbox = Inbox()
    inbox.followup(_message("A"))
    inbox.followup(_message("B"))
    reservation = inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ALL)
    inbox.followup(_message("C"))  # a re-entrant observer's own new input

    reservation.rollback()

    pending = inbox.pending(InboxTarget.NEXT_TURN)
    assert [text_of(e.message) for e in pending] == ["A", "B", "C"]


def test_a_reentrant_claim_on_the_same_target_sees_only_unrelated_input() -> None:
    """`L09-R013`'s own root cause, closed by construction: since the reserved batch is already
    gone from the queue before an observer runs, a reentrant `claim()` on the SAME target can only
    ever find genuinely different, unrelated input -- never any part of the reserved batch."""
    inbox = Inbox()
    inbox.followup(_message("A"))
    inbox.followup(_message("B"))
    inbox._reserve(InboxTarget.NEXT_TURN, ClaimPolicy.ALL)  # removes A, B atomically

    reentrant = inbox.claim(InboxTarget.NEXT_TURN, ClaimPolicy.ONE_AT_A_TIME)

    assert reentrant == ()  # nothing left for the observer's own claim to find


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
