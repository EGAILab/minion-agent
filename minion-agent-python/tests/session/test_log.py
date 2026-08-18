"""The log is append-only, sequence-numbered, and JSON-validated at append."""

import pytest

from minion_agent.session.events import EventKind
from minion_agent.session.log import NotJsonSafeError, SessionLog


def test_sequence_numbers_start_at_one_and_increase() -> None:
    log = SessionLog("s1")

    first = log.append(EventKind.USER_MESSAGE, {"text": "a"})
    second = log.append(EventKind.USER_MESSAGE, {"text": "b"})

    assert (first.seq, second.seq) == (1, 2)


def test_events_are_returned_in_append_order() -> None:
    log = SessionLog("s1")
    log.append(EventKind.TURN_START, {"turn": 1})
    log.append(EventKind.USER_MESSAGE, {"text": "a"})

    assert [event.kind for event in log.events] == [
        EventKind.TURN_START,
        EventKind.USER_MESSAGE,
    ]


def test_surface_filters_to_model_visible_events() -> None:
    log = SessionLog("s1")
    log.append(EventKind.TURN_START, {"turn": 1})
    log.append(EventKind.USER_MESSAGE, {"text": "a"})
    log.append(EventKind.ASSISTANT_CHUNK, {"delta": "x"})
    log.append(EventKind.ASSISTANT_MESSAGE, {"text": "b"})

    assert [event.kind for event in log.surface()] == [
        EventKind.USER_MESSAGE,
        EventKind.ASSISTANT_MESSAGE,
    ]


def test_non_json_safe_data_is_rejected_at_append() -> None:
    """Rejecting at the source is what keeps the log replayable and portable."""
    log = SessionLog("s1")

    with pytest.raises(NotJsonSafeError, match="not JSON-safe"):
        log.append(EventKind.USER_MESSAGE, {"blob": object()})

    assert len(log) == 0


def test_nested_non_json_safe_data_is_also_rejected() -> None:
    log = SessionLog("s1")

    with pytest.raises(NotJsonSafeError):
        log.append(EventKind.USER_MESSAGE, {"outer": {"inner": {1, 2}}})


def test_bytes_are_rejected_because_json_has_no_bytes() -> None:
    log = SessionLog("s1")

    with pytest.raises(NotJsonSafeError):
        log.append(EventKind.USER_MESSAGE, {"image": b"\x89PNG"})


def test_non_string_keys_are_rejected() -> None:
    log = SessionLog("s1")

    with pytest.raises(NotJsonSafeError, match="keys must be strings"):
        log.append(EventKind.USER_MESSAGE, {"outer": {1: "a"}})


def test_lists_and_tuples_are_walked() -> None:
    log = SessionLog("s1")

    log.append(EventKind.USER_MESSAGE, {"items": [1, "a", None, [True]]})

    with pytest.raises(NotJsonSafeError, match=r"\[1\]"):
        log.append(EventKind.USER_MESSAGE, {"items": (1, object())})


def test_the_log_reports_its_session_id_and_length() -> None:
    log = SessionLog("s1")
    log.append(EventKind.USER_MESSAGE, {"text": "a"})

    assert log.session_id == "s1"
    assert len(log) == 1


def test_a_root_log_has_no_ancestor() -> None:
    log = SessionLog("s1")

    assert log.ancestor is None
    assert log.boundary == 0
