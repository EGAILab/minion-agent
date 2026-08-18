"""`ctx.agents`: creating instances and owning their teardown."""

from __future__ import annotations

from ..runtime import Context
from ..session import SessionService
from .identity import AgentDefinition, AgentInstanceId
from .instance import AgentInstance


class DuplicateInstanceError(ValueError):
    """An instance id was reused while the first one is still live."""


class AgentHandle:
    """The teardown capability for exactly one instance.

    Bound to the instance it was issued for, so a stale handle cannot remove a
    later instance that happens to reuse the id.
    """

    __slots__ = ("_disposed", "_registry", "instance")

    def __init__(self, registry: AgentRegistry, instance: AgentInstance) -> None:
        self._registry = registry
        self.instance = instance
        self._disposed = False

    async def dispose(self) -> None:
        """Remove the instance and unwind its scope. Idempotent."""
        if self._disposed:
            return
        self._disposed = True
        await self._registry.detach(self.instance)


class AgentRegistry:
    """Owns every live agent instance."""

    __service_name__ = "agents"

    def __init__(self, ctx: Context, sessions: SessionService) -> None:
        self._ctx = ctx
        self._sessions = sessions
        self._instances: dict[AgentInstanceId, AgentInstance] = {}

    def create(self, instance_id: AgentInstanceId, definition: AgentDefinition) -> AgentHandle:
        """Create a live instance of `definition` under `instance_id`."""
        if instance_id in self._instances:
            raise DuplicateInstanceError(f"instance {instance_id!r} is already live")

        instance = AgentInstance(
            instance_id=instance_id,
            definition=definition,
            log=self._sessions.create(instance_id),
            ctx=self._ctx,
        )
        self._instances[instance_id] = instance
        return AgentHandle(self, instance)

    def get(self, instance_id: AgentInstanceId) -> AgentInstance | None:
        return self._instances.get(instance_id)

    def instances(self) -> tuple[AgentInstance, ...]:
        return tuple(self._instances.values())

    async def detach(self, instance: AgentInstance) -> None:
        """Remove `instance` if it is still the live one for its id.

        Called by the handle that owns it; the id-identity check is what stops
        a stale handle from removing a later instance that reused the id.
        """
        if self._instances.get(instance.id) is instance:
            del self._instances[instance.id]
        await instance.scope.dispose()
