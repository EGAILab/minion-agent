"""One live execution identity: inbox, log, turn state, scope."""

from __future__ import annotations

from collections.abc import Callable

from ..runtime import Context, ScopeKey
from ..session import SessionLog
from .events import AGENT_STATUS, declare_agent_events
from .identity import AgentDefinition, AgentInstanceId, AgentStatus
from .inbox import Inbox


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
