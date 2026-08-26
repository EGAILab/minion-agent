"""One live execution identity: inbox, log, turn state, scope."""

from __future__ import annotations

from collections.abc import Callable

from ..llm import Message, ModelId
from ..runtime import Context, ScopeKey
from ..session import SessionLog, derive_messages
from .envelope import InboxTarget, InputEnvelope, JsonValue
from .events import AGENT_STATUS, declare_agent_events
from .identity import AgentDefinition, AgentInstanceId, AgentStatus, ThinkingLevel
from .inbox import Inbox


class AgentActiveError(RuntimeError):
    """An operation that requires an idle instance was attempted while running."""


def instance_scope_key(definition: AgentDefinition, instance_id: str) -> ScopeKey:
    """The scope an instance registers through.

    A child of the definition's scope, so definition-level registrations are
    visible to every instance while instance-level ones stay private to one
    (design spec section 3).
    """
    return ScopeKey(f"agent-instance:{instance_id}", parent=ScopeKey(definition.scope_name))


class AgentInstance:
    """One conversation with one agent."""

    def __init__(
        self,
        *,
        instance_id: AgentInstanceId,
        definition: AgentDefinition,
        log: SessionLog,
        ctx: Context,
    ) -> None:
        self.id = instance_id
        self.definition = definition
        self.log = log
        self.inbox = Inbox()
        self._status = AgentStatus.IDLE
        self.on_status_change: Callable[[AgentStatus], None] | None = None

        # Mutable per-instance current configuration (`AG-014`, `L07-R001`):
        # pinned Pi's `AgentState.systemPrompt`/`model`/`thinkingLevel` are directly
        # assignable and read for later runs -- distinct from `definition`'s own
        # shared, frozen defaults, which never change once an instance has its own
        # current value. Defaults to the definition's own value, matching pinned
        # Pi's `createMutableAgentState`'s `initialState?.systemPrompt ?? ""` /
        # `initialState?.model ?? DEFAULT_MODEL` pattern.
        self.system_prompt: str = definition.system
        self.model: ModelId = definition.model
        self.thinking_level: ThinkingLevel = ThinkingLevel.OFF

        # Runtime-state vocabulary (`AG-015`): pinned Pi's `streamingMessage`/
        # `pendingToolCalls`/`errorMessage`, initial values only -- Layer 08 owns
        # when and how these change during a run; this layer owns only that they
        # exist and what they start as.
        self.streaming_message: Message | None = None
        self.pending_tool_calls: frozenset[str] = frozenset()
        self.error_message: str | None = None

        declare_agent_events(ctx.events)
        self.scope = ctx.scope(instance_scope_key(definition, instance_id))
        self._ctx = self.scope.ctx

    @property
    def ctx(self) -> Context:
        """This instance's scoped context."""
        return self._ctx

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def messages(self) -> tuple[Message, ...]:
        """The current transcript (`AG-015`): a fresh projection of the
        authoritative `SessionLog` (Layer 03, certified), never a live mutable
        reference -- an intentional divergence from pinned Pi's own
        `state.messages` getter (which returns the live backing array) since
        Session's log is the sole authority for history and must not be
        corruptible by mutating a returned collection in place."""
        return derive_messages(self.log)

    def set_status(self, status: AgentStatus) -> None:
        """Record a status transition and announce it.

        A no-op when the status is unchanged: `agent/status` is a transition
        signal, and emitting it for an assignment would make "settled" mean
        two different things.
        """
        if status is self._status:
            return
        self._status = status
        self._ctx.events.emit(AGENT_STATUS, self, status, scope=self.scope.key)
        if self.on_status_change is not None:
            self.on_status_change(status)

    def reset(self) -> None:
        """Clear runtime state and both queues in place (`AG-016`, `L07-R003`).

        Pinned Pi's `Agent.reset()`, exactly: rejects outright while active
        (`this.activeRun`, matched here by `status is not IDLE`), with the exact
        error text preserved and no partial mutation on rejection. When idle, it
        clears `streamingMessage`/`pendingToolCalls`/`errorMessage` and both
        queues, and retains everything else: object identity, `on_status_change`,
        `definition`, the mutable current `system_prompt`/`model`/`thinking_level`,
        and the tool/scope relationship. This is genuinely in-place -- the same
        `Inbox`/`SessionLog`/scope objects, not a replacement instance -- which is
        the exact property the independent Rust review's rejection turned on.

        One Pi-owned effect is deliberately NOT reproduced: pinned Pi's `reset()`
        also clears `messages` (`this._state.messages = []`). Pi's own transcript
        is a plain in-memory array with no separate persisted log; Minion's
        `SessionLog` (Layer 03, already certified) is append-only by design, with
        no truncate/clear primitive, and adding one -- or teaching `derive_messages`
        to respect a reset boundary -- would be a Layer-03 semantic change this
        narrow Layer-07 remediation has no mandate to make. This is a classified,
        disclosed cross-layer dependency, not a silently dropped requirement: see
        `assurance/layers/07-agent-state-inboxes-python.md`'s PASS 2.5 section.
        """
        if self._status is not AgentStatus.IDLE:
            raise AgentActiveError(
                "Agent is already processing. Wait for completion before resetting."
            )
        self.streaming_message = None
        self.pending_tool_calls = frozenset()
        self.error_message = None
        self.inbox.clear_all()

    def steer(self, message: Message, origin: JsonValue = None) -> InputEnvelope:
        """Pinned Pi's `Agent.steer()` -- the Agent-level public surface
        (`AG-011`), delegating to the authoritative `Inbox`."""
        return self.inbox.steer(message, origin=origin)

    def follow_up(self, message: Message, origin: JsonValue = None) -> InputEnvelope:
        """Pinned Pi's `Agent.followUp()` (`AG-011`)."""
        return self.inbox.followup(message, origin=origin)

    def inject(self, message: Message, origin: JsonValue = None) -> InputEnvelope:
        """Silent context injection -- an intentional Minion extension with no
        Pi equivalent (`AG-011`), exposed at the Agent level alongside `steer`/
        `follow_up` for the same reason those are."""
        return self.inbox.inject(message, origin=origin)

    def has_queued_messages(self) -> bool:
        """Pinned Pi's `Agent.hasQueuedMessages()` -- the Agent-level public
        surface (`AG-011`), delegating to the authoritative `Inbox`."""
        return self.inbox.has_pending()

    def clear_steering_queue(self) -> None:
        """Pinned Pi's `Agent.clearSteeringQueue()` (`AG-013`)."""
        self.inbox.clear(InboxTarget.NEXT_STEP)

    def clear_follow_up_queue(self) -> None:
        """Pinned Pi's `Agent.clearFollowUpQueue()` (`AG-013`)."""
        self.inbox.clear(InboxTarget.NEXT_TURN)

    def clear_all_queues(self) -> None:
        """Pinned Pi's `Agent.clearAllQueues()` (`AG-013`)."""
        self.inbox.clear_all()
