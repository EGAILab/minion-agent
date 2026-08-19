"""Mounting the agents registry."""

from __future__ import annotations

from ..runtime import Context, plugin
from .registry import AgentRegistry


@plugin(name="agents", inject=["sessions"], provides="agents")
async def agents_plugin(ctx: Context, config: None) -> None:
    """Provide the agents registry.

    Injects `sessions` because every instance owns a session log, and the
    registry mints one per instance.
    """
    ctx.provide("agents", AgentRegistry(ctx=ctx, sessions=ctx.sessions))
