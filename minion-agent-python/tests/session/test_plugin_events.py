"""The event namespace is open: plugins may declare their own kinds.

§5 states that plugins may declare session events that join the model-visible
surface, so the *name string* is the language-neutral identity. Core constants
are ergonomics, not a closed set.
"""

import pytest

from minion_agent.llm import TextBlock, UserMessage, text_of
from minion_agent.session import (
    CORE_SURFACE_KINDS,
    EventKind,
    SessionLog,
    derive_messages,
    encode_message,
)
from minion_agent.session.events import InvalidEventNameError, validate_event_name


def _user(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def test_a_plugin_may_append_its_own_event_kind() -> None:
    log = SessionLog("s1")

    event = log.append("plugin/notification", {"text": "deploy finished"})

    assert event.kind == "plugin/notification"


def test_a_plugin_event_is_log_only_by_default() -> None:
    """Declaring a name does not silently widen what the model sees."""
    log = SessionLog("s1")
    log.append("plugin/notification", {"text": "ignored"})

    assert log.surface() == ()
    assert derive_messages(log) == ()


def test_a_plugin_event_can_join_the_surface() -> None:
    log = SessionLog("s1", surface_kinds=CORE_SURFACE_KINDS | {"plugin/note"})

    log.append("plugin/note", {"message": encode_message(_user("from a plugin"))})

    assert [text_of(m) for m in derive_messages(log)] == ["from a plugin"]


def test_core_kinds_remain_surface_when_a_plugin_widens_the_set() -> None:
    log = SessionLog("s1", surface_kinds=CORE_SURFACE_KINDS | {"plugin/note"})

    log.append(EventKind.USER_MESSAGE, {"message": encode_message(_user("core"))})
    log.append("plugin/note", {"message": encode_message(_user("plugin"))})

    assert [text_of(m) for m in derive_messages(log)] == ["core", "plugin"]


def test_a_fork_inherits_its_ancestors_surface_set() -> None:
    """Otherwise a plugin event would derive in the parent and vanish in the
    child, which is not a distinction anything asked for."""
    from minion_agent.session.operations import fork

    source = SessionLog("s1", surface_kinds=CORE_SURFACE_KINDS | {"plugin/note"})
    source.append("plugin/note", {"message": encode_message(_user("inherited"))})

    child = fork(source, "s2")
    child.append("plugin/note", {"message": encode_message(_user("own"))})

    assert [text_of(m) for m in derive_messages(child)] == ["inherited", "own"]


def test_a_malformed_event_name_is_rejected() -> None:
    log = SessionLog("s1")

    for bad in ("Plugin/Note", "plugin note", "/leading", "trailing/", ""):
        with pytest.raises(InvalidEventNameError):
            log.append(bad, {})

    assert len(log) == 0


def test_a_non_string_name_is_rejected() -> None:
    with pytest.raises(InvalidEventNameError, match="must be a string"):
        validate_event_name(42)  # type: ignore[arg-type]


def test_validation_accepts_any_well_formed_name() -> None:
    """Validation is about portability, not membership."""
    for good in ("plugin/foo", "a", "plugin/sub-part", "vendor/thing/nested"):
        assert validate_event_name(good) == good


def test_core_names_are_still_valid_identities() -> None:
    for kind in EventKind:
        assert validate_event_name(kind.value) == kind.value


def test_the_core_surface_set_is_exactly_three_kinds() -> None:
    """Normative and language-neutral; a plugin widens its own log, never this."""
    assert {
        EventKind.USER_MESSAGE,
        EventKind.ASSISTANT_MESSAGE,
        EventKind.TOOL_RESULT,
    } == CORE_SURFACE_KINDS


def test_a_raw_string_kind_is_the_same_event_as_its_constant() -> None:
    """The name is the identity. An identity comparison would silently ignore
    the raw-string form, and a second implementation comparing strings would
    disagree with this one."""
    from minion_agent.session.operations import compact

    literal = SessionLog("s1")
    literal.append("user/message", {"message": encode_message(_user("before"))})
    literal.append("session/reset", {})
    literal.append("user/message", {"message": encode_message(_user("after"))})

    constant = SessionLog("s2")
    constant.append(EventKind.USER_MESSAGE, {"message": encode_message(_user("before"))})
    constant.append(EventKind.SESSION_RESET, {})
    constant.append(EventKind.USER_MESSAGE, {"message": encode_message(_user("after"))})

    assert derive_messages(literal) == derive_messages(constant)
    assert [text_of(m) for m in derive_messages(literal)] == ["after"]

    # The same holds for compaction, which also looks up by name.
    raw = SessionLog("s3")
    raw.append("user/message", {"message": encode_message(_user("one"))})
    compact(raw, summary="summarised", keep=0)

    assert [text_of(m) for m in derive_messages(raw)] == ["summarised"]
