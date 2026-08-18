"""Reset changes derivation without changing identity or deleting history."""

from minion_agent.llm.content import TextBlock
from minion_agent.llm.messages import UserMessage, text_of
from minion_agent.session.derive import derive_messages, encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.session.operations import reset


def _say(log: SessionLog, text: str) -> None:
    message = UserMessage(content=(TextBlock(text=text),), timestamp=1)
    log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})


def test_reset_excludes_prior_surface_from_derivation() -> None:
    log = SessionLog("s1")
    _say(log, "before")
    reset(log)
    _say(log, "after")

    assert [text_of(m) for m in derive_messages(log)] == ["after"]


def test_reset_preserves_session_identity() -> None:
    """A session id is a durable external handle; clearing history must not
    invalidate the bindings an application holds against it."""
    log = SessionLog("s1")
    _say(log, "before")

    reset(log)

    assert log.session_id == "s1"


def test_reset_does_not_delete_history() -> None:
    log = SessionLog("s1")
    _say(log, "before")
    reset(log)

    assert len(log) == 2
    assert log.events[0].kind is EventKind.USER_MESSAGE


def test_a_second_reset_supersedes_the_first() -> None:
    log = SessionLog("s1")
    _say(log, "one")
    reset(log)
    _say(log, "two")
    reset(log)
    _say(log, "three")

    assert [text_of(m) for m in derive_messages(log)] == ["three"]


def test_reset_on_an_empty_log_is_harmless() -> None:
    log = SessionLog("s1")

    reset(log)
    _say(log, "after")

    assert [text_of(m) for m in derive_messages(log)] == ["after"]


def test_reset_appends_a_reset_event() -> None:
    log = SessionLog("s1")

    event = reset(log)

    assert event.kind is EventKind.SESSION_RESET
