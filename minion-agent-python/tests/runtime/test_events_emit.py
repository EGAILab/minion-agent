"""Event declarations bind a dispatch mode; emit is synchronous and unawaited."""

import pytest

from minion_agent.runtime.errors import EventModeError
from minion_agent.runtime.events import DispatchMode, EventBus


def test_emit_calls_listeners_in_registration_order() -> None:
    bus = EventBus()
    bus.declare("test/emit", DispatchMode.EMIT)
    seen: list[str] = []
    bus.on("test/emit", lambda: seen.append("first"))
    bus.on("test/emit", lambda: seen.append("second"))

    bus.emit("test/emit")

    assert seen == ["first", "second"]


def test_prepend_puts_listener_first() -> None:
    bus = EventBus()
    bus.declare("test/emit", DispatchMode.EMIT)
    seen: list[str] = []
    bus.on("test/emit", lambda: seen.append("ordinary"))
    bus.on("test/emit", lambda: seen.append("prepended"), prepend=True)

    bus.emit("test/emit")

    assert seen == ["prepended", "ordinary"]


def test_disposer_removes_listener() -> None:
    bus = EventBus()
    bus.declare("test/emit", DispatchMode.EMIT)
    seen: list[str] = []
    dispose = bus.on("test/emit", lambda: seen.append("gone"))

    dispose()
    bus.emit("test/emit")

    assert seen == []


def test_emit_passes_arguments() -> None:
    bus = EventBus()
    bus.declare("test/emit", DispatchMode.EMIT)
    seen: list[tuple[int, str]] = []
    bus.on("test/emit", lambda number, label: seen.append((number, label)))

    bus.emit("test/emit", 7, "seven")

    assert seen == [(7, "seven")]


def test_dispatching_in_the_wrong_mode_raises() -> None:
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)

    with pytest.raises(EventModeError, match="declared 'serial'"):
        bus.emit("test/serial")


def test_undeclared_event_raises() -> None:
    bus = EventBus()

    with pytest.raises(EventModeError, match="not declared"):
        bus.emit("test/unknown")


def test_redeclaring_with_a_different_mode_raises() -> None:
    bus = EventBus()
    bus.declare("test/emit", DispatchMode.EMIT)

    with pytest.raises(EventModeError, match="already declared"):
        bus.declare("test/emit", DispatchMode.SERIAL)


def test_redeclaring_with_the_same_mode_is_allowed() -> None:
    bus = EventBus()
    bus.declare("test/emit", DispatchMode.EMIT)
    bus.declare("test/emit", DispatchMode.EMIT)

    assert bus.mode_of("test/emit") is DispatchMode.EMIT


def test_registering_a_listener_for_an_undeclared_event_raises() -> None:
    bus = EventBus()

    with pytest.raises(EventModeError, match="not declared"):
        bus.on("test/unknown", lambda: None)
