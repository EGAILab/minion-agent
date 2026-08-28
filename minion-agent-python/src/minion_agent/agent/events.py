"""The `agent/*` event vocabulary and its declared dispatch modes."""

from __future__ import annotations

from ..runtime import DispatchMode, EventBus

AGENT_STATUS = "agent/status"
"""Emitted on every idle<->running transition. The settle signal."""

AGENT_PRE_STEP = "agent/pre-step"
"""Waterfall returning `Reject | Enter`, terminal `Enter(claimed messages)`."""

AGENT_TURN_STOPPING = "agent/turn-stopping"
"""Serial, returning `TurnStopping`. Listener signature:
`(instance, message, tool_results, context, new_messages) -> TurnStopping`, mirroring pinned Pi's
own `ShouldStopAfterTurnContext` exactly (`L08-R001`, PASS 3 -- an earlier revision gave listeners
no context at all). Dispatched after every turn, including one a tool batch's `terminate` verdict
ended -- pinned Pi's `shouldStopAfterTurn` runs regardless of `hasMoreToolCalls` (`terminate` only
ever affects whether the tool-driven inner loop has more work, a fact this event's own listeners
still get to observe). Not dispatched for a represented `error`/`aborted` assistant message
(`L08-R008`, PASS 3): pinned Pi returns immediately after that turn's own `turn_end`."""

AGENT_PREPARE_NEXT_TURN = "agent/prepare-next-turn"
"""Waterfall returning a `RunConfigUpdate`, terminal `RunConfigUpdate()` (no override). Listener
signature: `(instance, message, tool_results, context, new_messages, next_) -> RunConfigUpdate`,
mirroring pinned Pi's own `PrepareNextTurnContext` exactly (`L08-R001`, PASS 3 -- an earlier
revision passed only the tool-batch outcome, and `RunConfigUpdate` could replace only
`system_prompt`, not pinned Pi's whole `context`). Dispatched after every turn (including one a
tool batch's `terminate` verdict ended), before the stop decision. Any returned
`context`/`model`/`thinking_level` applies to the next provider request only -- never persisted
back to the certified Layer-07 `AgentInstance`. Not dispatched for a represented `error`/`aborted`
assistant message (`L08-R008`, PASS 3): pinned Pi returns immediately after that turn's own
`turn_end`."""

AGENT_INBOX_INSERTED = "agent/inbox/inserted"
AGENT_INBOX_CLAIMED = "agent/inbox/claimed"

AGENT_EVENT_MODES: dict[str, DispatchMode] = {
    AGENT_STATUS: DispatchMode.EMIT,
    AGENT_PRE_STEP: DispatchMode.WATERFALL,
    AGENT_TURN_STOPPING: DispatchMode.SERIAL,
    AGENT_PREPARE_NEXT_TURN: DispatchMode.WATERFALL,
    AGENT_INBOX_INSERTED: DispatchMode.EMIT,
    AGENT_INBOX_CLAIMED: DispatchMode.EMIT,
}


def declare_agent_events(bus: EventBus) -> None:
    """Declare every agent event. Idempotent for matching modes."""
    for name, mode in AGENT_EVENT_MODES.items():
        bus.declare(name, mode)
