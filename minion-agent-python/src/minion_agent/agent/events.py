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

AGENT_LIFECYCLE_EVENT = "agent/lifecycle-event"
"""Serial, no return value. Listener signature: `(instance, event) -> None`, where `event` is one of
`agent.projection`'s own `AgentEvent` union members (`AgentStart`/`TurnStart`/`MessageStart`/
`MessageEnd`/`TurnEnd`/`AgentEnd`) -- pinned Pi's own `AgentEvent`, dispatched LIVE, during the run
(`L08-R002`, PASS 4). Pinned Pi's `Agent.subscribe(listener)` is a public surface with no Minion
equivalent before this pass: `agent/projection.py::project()` only ever reconstructs this vocabulary
*offline*, by walking a completed log, which cannot dispatch or interrupt anything live. This event
is the single seam every lifecycle event -- ordinary turn/run progress AND `handleRunFailure`
recovery alike -- passes through, matching pinned Pi's own `processEvents`: state already durably
recorded (the log append) happens first, then every subscribed listener is awaited in registration
order. A listener that throws aborts the remaining dispatch for that one event and propagates,
exactly like pinned Pi's own bare `for (const listener of this.listeners) await listener(event,
signal)` loop -- `EventBus.serial`'s existing semantics already match this precisely, with no new
dispatch primitive needed.

Deliberately bounded (`L08-R002`, PASS 4): the assistant reply's OWN streamed `message_start`/
`message_update`/`message_end` lifecycle is NOT dispatched through this seam. The already-certified
Layer-02/04 `collect()` accepts only a synchronous `on_chunk` callback (`llm/stream.py`) -- an
intentional, narrow design for that layer, not something this pass reopens -- so a chunk-level
listener cannot itself `await` a dispatch. `streaming_message` fidelity (`L08-R003`, closed) and
this event's own scope are independent concerns: the former is unaffected, and the latter remains
log + offline `project()` only for the assistant's OWN reply, exactly as before this pass. Every
OTHER lifecycle event -- `agent_start`, `turn_start`, every ADMITTED message batch (prompt/steering/
tool-result/follow-up, admitted via `AgentLoop._admit_messages`, all async, none stream-bound),
`turn_end`, and `agent_end` -- runs live through this seam, including the full `handleRunFailure`
recovery sequence."""

AGENT_EVENT_MODES: dict[str, DispatchMode] = {
    AGENT_STATUS: DispatchMode.EMIT,
    AGENT_PRE_STEP: DispatchMode.WATERFALL,
    AGENT_TURN_STOPPING: DispatchMode.SERIAL,
    AGENT_PREPARE_NEXT_TURN: DispatchMode.WATERFALL,
    AGENT_INBOX_INSERTED: DispatchMode.EMIT,
    AGENT_INBOX_CLAIMED: DispatchMode.EMIT,
    AGENT_LIFECYCLE_EVENT: DispatchMode.SERIAL,
}


def declare_agent_events(bus: EventBus) -> None:
    """Declare every agent event. Idempotent for matching modes."""
    for name, mode in AGENT_EVENT_MODES.items():
        bus.declare(name, mode)
