"""Dispatch mode is part of each event's public contract."""

import pytest

from minion_agent.runtime import DispatchMode, EventBus, EventModeError
from minion_agent.tools.events import (
    TOOLS_EVENT_MODES,
    TOOLS_POST_EXECUTE,
    TOOLS_PRE_EXECUTE,
    TOOLS_REGISTERED,
    TOOLS_UPDATE,
    declare_tools_events,
)


def test_declaring_registers_every_tools_event() -> None:
    bus = EventBus()

    declare_tools_events(bus)

    for name, mode in TOOLS_EVENT_MODES.items():
        assert bus.mode_of(name) is mode


def test_both_pipeline_events_are_waterfalls() -> None:
    """One is a decision, one is a transformation, but the mechanism is the
    same -- they differ in how listeners use `next`, not in dispatch."""
    assert TOOLS_EVENT_MODES[TOOLS_PRE_EXECUTE] is DispatchMode.WATERFALL
    assert TOOLS_EVENT_MODES[TOOLS_POST_EXECUTE] is DispatchMode.WATERFALL


def test_update_is_emit() -> None:
    """A partial result must not be able to block the tool producing it."""
    assert TOOLS_EVENT_MODES[TOOLS_UPDATE] is DispatchMode.EMIT


def test_registration_announcements_are_emit() -> None:
    """A tool becoming available is an observation, not a negotiation: a
    listener must not be able to block or rewrite it."""
    assert TOOLS_EVENT_MODES[TOOLS_REGISTERED] is DispatchMode.EMIT


def test_declaring_twice_is_harmless() -> None:
    bus = EventBus()

    declare_tools_events(bus)
    declare_tools_events(bus)

    assert bus.mode_of(TOOLS_UPDATE) is DispatchMode.EMIT


def test_dispatching_in_the_wrong_mode_still_raises() -> None:
    bus = EventBus()
    declare_tools_events(bus)

    with pytest.raises(EventModeError):
        bus.emit(TOOLS_PRE_EXECUTE)
