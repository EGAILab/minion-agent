"""The agent loop plugin.

The driver is package-internal: only this module's factory constructs one, so
nothing above can reach past the `agent` package's interface into loop
internals.
"""

from __future__ import annotations

from ..agent.instance import AgentInstance
from ..runtime import Context, plugin
from ..telemetry import TelemetryService
from .driver import AgentLoop


class AgentLoopFactory:
    """Builds a driver for an instance, wired to the mounted services."""

    __service_name__ = "agent_loop"

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    def _telemetry(self) -> TelemetryService | None:
        """Telemetry if a sink stack is mounted, otherwise None.

        Resolved by asking the registry rather than injected: injecting it
        would make an observational projection a *precondition* for running,
        the opposite of what section 7 says it is. `getattr` with a default
        would not do -- an absent service raises ServiceNotFoundError, which
        is not an AttributeError.
        """
        if not self._ctx.registry.has("telemetry"):
            return None
        telemetry: TelemetryService = self._ctx.telemetry
        return telemetry

    def for_instance(self, instance: AgentInstance) -> AgentLoop:
        """A driver for `instance`, sharing this context's services."""
        return AgentLoop(
            instance=instance,
            llm=self._ctx.llm,
            tools=self._ctx.tools,
            artifacts=self._ctx.sessions.artifacts,
            telemetry=self._telemetry(),
        )


@plugin(
    name="agent-loop",
    inject=["agents", "llm", "tools", "sessions"],
    provides="agent_loop",
)
async def agent_loop_plugin(ctx: Context, config: None) -> None:
    """Provide the loop factory once every service it drives exists.

    Telemetry is deliberately *not* injected: it is observational, so the loop
    must run without it rather than wait for it.
    """
    ctx.provide("agent_loop", AgentLoopFactory(ctx))


__all__ = ["AgentLoopFactory", "agent_loop_plugin"]
