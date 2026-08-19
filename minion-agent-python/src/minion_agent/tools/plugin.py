"""Mounting the tool registry."""

from __future__ import annotations

from ..runtime import Context, plugin
from .events import declare_tools_events
from .registry import ToolRegistry


@plugin(name="tools", provides="tools")
async def tools_plugin(ctx: Context, config: None) -> None:
    """Provide the tool registry and declare the pipeline events.

    Declaration happens here rather than at first dispatch because a listener
    cannot register for an undeclared event: a plugin that mounts alongside
    this one must be able to subscribe immediately.
    """
    declare_tools_events(ctx.events)
    ctx.provide("tools", ToolRegistry())
