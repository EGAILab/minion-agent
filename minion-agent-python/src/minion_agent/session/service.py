"""The `ctx.sessions` seam."""

from __future__ import annotations

from ..runtime import Context, plugin
from .artifacts import ArtifactStore
from .events import CORE_SURFACE_KINDS, EventName
from .log import SessionLog
from .operations import fork


class SessionService:
    """Owns every live session log and the shared artifact store."""

    __service_name__ = "sessions"

    def __init__(self) -> None:
        self.artifacts = ArtifactStore()
        """Shared across sessions: content addressing only pays off if a
        stable block is stored once for the deployment, not once per session."""
        self._logs: dict[str, SessionLog] = {}

    def create(
        self,
        session_id: str,
        surface_kinds: frozenset[EventName] = CORE_SURFACE_KINDS,
    ) -> SessionLog:
        """Create and register a new session log.

        `surface_kinds` widens what projects into model history, for a
        deployment whose plugins declare surface events (§5).
        """
        log = SessionLog(session_id, surface_kinds=surface_kinds)
        self._logs[session_id] = log
        return log

    def get(self, session_id: str) -> SessionLog | None:
        """The log for `session_id`, or None."""
        return self._logs.get(session_id)

    def fork(self, source_id: str, session_id: str, at: int | None = None) -> SessionLog:
        """Fork `source_id` into a new registered session."""
        source = self._logs[source_id]
        child = fork(source, session_id, at=at)
        self._logs[session_id] = child
        return child


@plugin(name="sessions", provides="sessions")
async def session_plugin(ctx: Context, config: None) -> None:
    ctx.provide("sessions", SessionService())
