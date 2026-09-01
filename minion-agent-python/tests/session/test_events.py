"""Two tiers of event: the surface projects to the model, the rest does not."""

from minion_agent.session.events import SURFACE_KINDS, EventKind, SessionEvent, is_surface


def test_the_surface_is_exactly_three_kinds() -> None:
    """Widening this set widens what the model sees, so it is pinned."""
    assert {
        EventKind.USER_MESSAGE,
        EventKind.ASSISTANT_MESSAGE,
        EventKind.TOOL_RESULT,
    } == SURFACE_KINDS


def test_lifecycle_events_are_not_surface() -> None:
    for kind in (
        EventKind.AGENT_START,
        EventKind.AGENT_END,
        EventKind.TURN_START,
        EventKind.TURN_END,
        EventKind.ASSISTANT_CHUNK,
        EventKind.TOOL_CALL,
        EventKind.REQUEST_HEADER,
    ):
        assert not is_surface(SessionEvent(seq=1, kind=kind, data={}))


def test_a_surface_event_reports_as_surface() -> None:
    event = SessionEvent(seq=1, kind=EventKind.USER_MESSAGE, data={"text": "hi"})

    assert is_surface(event)


def test_operation_events_exist_and_are_not_surface() -> None:
    """Fork, reset, and compaction change derivation without being messages."""
    for kind in (EventKind.SESSION_FORKED, EventKind.SESSION_RESET, EventKind.COMPACTION):
        assert not is_surface(SessionEvent(seq=1, kind=kind, data={}))
