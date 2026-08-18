"""The `agent/*` event vocabulary and its declared dispatch modes."""

from __future__ import annotations

from ..runtime import DispatchMode, EventBus

AGENT_STATUS = "agent/status"
"""Emitted on every idle<->running transition. The settle signal."""

AGENT_PRE_STEP = "agent/pre-step"
"""Waterfall returning `Reject | Enter`, terminal `Enter(claimed messages)`."""

AGENT_TURN_STOPPING = "agent/turn-stopping"
"""Serial, returning `TurnStopping`. Not dispatched when hard termination fires."""

AGENT_INBOX_INSERTED = "agent/inbox/inserted"
AGENT_INBOX_CLAIMED = "agent/inbox/claimed"

AGENT_EVENT_MODES: dict[str, DispatchMode] = {
    AGENT_STATUS: DispatchMode.EMIT,
    AGENT_PRE_STEP: DispatchMode.WATERFALL,
    AGENT_TURN_STOPPING: DispatchMode.SERIAL,
    AGENT_INBOX_INSERTED: DispatchMode.EMIT,
    AGENT_INBOX_CLAIMED: DispatchMode.EMIT,
}


def declare_agent_events(bus: EventBus) -> None:
    """Declare every agent event. Idempotent for matching modes."""
    for name, mode in AGENT_EVENT_MODES.items():
        bus.declare(name, mode)
