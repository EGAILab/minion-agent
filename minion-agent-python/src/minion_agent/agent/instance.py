"""One live execution identity: inbox, log, turn state, scope."""

from __future__ import annotations

from collections.abc import Callable

from ..llm import Message, ModelId
from ..runtime import Context, ScopeKey
from ..session import SessionLog, derive_messages
from ..session import reset as reset_session_log
from ..tools import ToolDefinition, ToolRegistry
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

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        """The current visible-tool set (`AG-017`): a fresh projection over the
        already-certified Layer-05 `ToolRegistry`, from this instance's own scope --
        never a duplicate store, mirroring `messages`'s read-through pattern above.
        Resolved through `ctx.tools` the same way `AgentLoopFactory.for_instance`
        already does when composing a driver (`agent_loop/__init__.py`); this
        property is not a second resolution mechanism, only a second caller of the
        existing one, exposed at the Agent-instance level Pi's own `state.tools`
        getter parity requires (the independent review's `L07-R005` tools
        sub-finding: the projection existed by construction elsewhere, but nothing
        on `AgentInstance` itself actually answered "what tools does this agent
        see" the way Pi's `Agent.state.tools` does).

        Empty, not an error, when no `tools` service is mounted at all (`L07-R007`,
        third independent Rust review): pinned Pi's own `AgentState.tools` starts as
        an observable empty array regardless of whether an application has wired up
        any tool source yet, and a valid, freshly constructed `AgentInstance` must be
        just as total -- reading a field that simply has nothing in it yet is not the
        same failure as asking for a service that was never provided. Checking
        `ctx.registry.has("tools")` first mirrors `AgentLoopFactory._telemetry()`'s
        own established idiom for exactly this "optional service, empty default when
        absent" shape (`agent_loop/__init__.py`), not a new resolution pattern."""
        if not self._ctx.registry.has("tools"):
            return ()
        registry: ToolRegistry = self._ctx.tools
        return registry.visible_from(self.scope.key)

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
        """Clear runtime state, messages, and both queues in place (`AG-016`, `L07-R003`).

        Pinned Pi's `Agent.reset()`, exactly: rejects outright while active
        (`this.activeRun`, matched here by `status is not IDLE`), with the exact
        error text preserved and no partial mutation on rejection. When idle, it
        clears `streamingMessage`/`pendingToolCalls`/`errorMessage`, `messages`,
        and both queues, and retains everything else: object identity,
        `on_status_change`, `definition`, the mutable current
        `system_prompt`/`model`/`thinking_level`, and the tool/scope relationship.
        This is genuinely in-place -- the same `Inbox`/`SessionLog`/scope objects,
        not a replacement instance -- which is the exact property the independent
        Rust review's rejection turned on.

        `messages` is cleared via the already-certified Layer-03
        `session.reset(log)` (`session/operations.py`), which appends a
        `session/reset` marker event rather than truncating the append-only log:
        `derive_messages` already treats the latest such marker as an exclusive
        floor (`effective_surface`/`_derive` in `session/derive.py`), so every
        event at or before it stops projecting into `AgentInstance.messages`
        immediately, without adding any new primitive to `SessionLog` itself. A
        prior pass concluded no such primitive existed and classified this as an
        unresolvable cross-layer dependency; the second independent Rust review
        correctly rejected that conclusion by pointing at this exact,
        already-certified mechanism, which that prior pass had not found.

        The pending wake signal (`Inbox.wake_requested`), if any, is deliberately
        NOT cleared by reset -- an explicit, normative decision, not an untested
        side effect of delegating to `clear_all()`: wake and queued content are
        orthogonal (`Inbox.clear`'s own docstring), and pinned Pi has no wake
        concept to take a position on at all, so nothing about `reset()`'s Pi
        parity constrains this either way. Preserving it is intentional because a
        wake that arrived before a caller reset an idle instance still describes
        a real, unconsumed signal (`test_reset_does_not_clear_a_pending_wake_signal`).
        """
        if self._status is not AgentStatus.IDLE:
            raise AgentActiveError(
                "Agent is already processing. Wait for completion before resetting."
            )
        self.streaming_message = None
        self.pending_tool_calls = frozenset()
        self.error_message = None
        self.inbox.clear_all()
        reset_session_log(self.log)

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
