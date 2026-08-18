"""Compaction replaces a surface span in derivation without deleting anything."""

from minion_agent.llm.content import TextBlock
from minion_agent.llm.messages import UserMessage, text_of
from minion_agent.session.derive import derive_messages, encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.session.operations import compact, reset


def _say(log: SessionLog, text: str) -> None:
    message = UserMessage(content=(TextBlock(text=text),), timestamp=1)
    log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})


def _texts(log: SessionLog) -> list[str]:
    return [text_of(message) for message in derive_messages(log)]


def test_compaction_replaces_the_superseded_span_with_a_summary() -> None:
    log = SessionLog("s1")
    for text in ("one", "two", "three", "four"):
        _say(log, text)

    compact(log, summary="talked about numbers", keep=1)

    assert _texts(log) == ["talked about numbers", "four"]


def test_nothing_is_double_projected() -> None:
    """The retained tail is carried forward by reference, never duplicated —
    the failure mode provenance exists to prevent."""
    log = SessionLog("s1")
    for text in ("one", "two", "three"):
        _say(log, text)

    compact(log, summary="summary", keep=2)

    derived = _texts(log)

    assert derived == ["summary", "two", "three"]
    assert derived.count("two") == 1


def test_keeping_nothing_leaves_only_the_summary() -> None:
    log = SessionLog("s1")
    _say(log, "one")
    _say(log, "two")

    compact(log, summary="all of it", keep=0)

    assert _texts(log) == ["all of it"]


def test_messages_after_compaction_append_normally() -> None:
    log = SessionLog("s1")
    _say(log, "one")
    compact(log, summary="summary", keep=0)
    _say(log, "after")

    assert _texts(log) == ["summary", "after"]


def test_repeated_compaction_supersedes_the_earlier_summary() -> None:
    log = SessionLog("s1")
    _say(log, "one")
    _say(log, "two")
    compact(log, summary="first summary", keep=0)
    _say(log, "three")

    compact(log, summary="second summary", keep=1)

    assert _texts(log) == ["second summary", "three"]


def test_compaction_deletes_nothing_from_the_log() -> None:
    log = SessionLog("s1")
    _say(log, "one")
    _say(log, "two")

    compact(log, summary="summary", keep=0)

    assert len(log) == 3
    assert [event.kind for event in log.events][:2] == [
        EventKind.USER_MESSAGE,
        EventKind.USER_MESSAGE,
    ]


def test_reset_after_compaction_clears_the_summary_too() -> None:
    log = SessionLog("s1")
    _say(log, "one")
    compact(log, summary="summary", keep=0)
    reset(log)
    _say(log, "after")

    assert _texts(log) == ["after"]


def test_compacting_an_empty_log_yields_only_the_summary() -> None:
    log = SessionLog("s1")

    compact(log, summary="nothing happened", keep=0)

    assert _texts(log) == ["nothing happened"]
