"""Dispatch mode is part of each event's public contract."""

import pytest

from minion_agent.agent.events import (
    AGENT_EVENT_MODES,
    AGENT_PRE_STEP,
    AGENT_STATUS,
    AGENT_TURN_STOPPING,
    declare_agent_events,
)
from minion_agent.runtime import DispatchMode, EventBus, EventModeError


def test_declaring_registers_every_agent_event() -> None:
    bus = EventBus()

    declare_agent_events(bus)

    for name, mode in AGENT_EVENT_MODES.items():
        assert bus.mode_of(name) is mode


def test_pre_step_is_a_waterfall() -> None:
    """It carries a closed decision union and must support short-circuiting."""
    assert AGENT_EVENT_MODES[AGENT_PRE_STEP] is DispatchMode.WATERFALL


def test_turn_stopping_is_serial() -> None:
    assert AGENT_EVENT_MODES[AGENT_TURN_STOPPING] is DispatchMode.SERIAL


def test_status_is_emit() -> None:
    """The settle signal must not be able to block the loop it reports on."""
    assert AGENT_EVENT_MODES[AGENT_STATUS] is DispatchMode.EMIT


def test_declaring_twice_is_harmless() -> None:
    bus = EventBus()

    declare_agent_events(bus)
    declare_agent_events(bus)

    assert bus.mode_of(AGENT_STATUS) is DispatchMode.EMIT


def test_dispatching_in_the_wrong_mode_still_raises() -> None:
    bus = EventBus()
    declare_agent_events(bus)

    with pytest.raises(EventModeError):
        bus.emit(AGENT_PRE_STEP)
