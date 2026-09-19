"""Mounting the execution capability seams on the runtime (`L12-PY-R003`).

Three separate `@plugin`-decorated functions, one per service, matching the established
one-plugin-one-service convention every other certified layer in this codebase already uses
(`llm/plugin.py`, `session/service.py`, `telemetry/plugin.py`, `tools/plugin.py`). Each
constructs its own `Local*` provider independently -- they still end up sharing ONE
execution-world identity by construction, since `ExecutionWorldIdentity.local()` returns the
SAME fixed value regardless of which provider calls it (no shared state needs to be threaded
between the three plugin functions for this to hold).
"""

from __future__ import annotations

from ..runtime import Context, plugin
from .filesystem import LocalFileSystem
from .shell import LocalShell
from .subprocess import LocalSubprocess


@plugin(name="execution-fs", provides="fs")
async def fs_plugin(ctx: Context, config: None) -> None:
    ctx.provide("fs", LocalFileSystem())


@plugin(name="execution-shell", provides="shell")
async def shell_plugin(ctx: Context, config: None) -> None:
    ctx.provide("shell", LocalShell())


@plugin(name="execution-subprocess", provides="subprocess")
async def subprocess_plugin(ctx: Context, config: None) -> None:
    ctx.provide("subprocess", LocalSubprocess())
