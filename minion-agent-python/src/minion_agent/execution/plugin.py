"""Mounting the execution capability seams on the runtime (`L12-PY-R003`).

Three separate `@plugin`-decorated functions, one per service, matching the established
one-plugin-one-service convention every other certified layer in this codebase already uses
(`llm/plugin.py`, `session/service.py`, `telemetry/plugin.py`, `tools/plugin.py`). Each
constructs its own `Local*` provider independently -- they still end up sharing ONE
execution-world identity by construction, since `ExecutionWorldIdentity.local()` returns the
SAME fixed value regardless of which provider calls it (no shared state needs to be threaded
between the three plugin functions for this to hold).

`shell_plugin` genuinely DEPENDS on `subprocess` (refined at `L12-PY-R003`, second review): an
earlier revision declared this dependency only via `inject=["subprocess"]` -- which, matching
this codebase's own established convention (`registry.py`'s `is_visible` check), gates this
fiber's own service VISIBILITY on `subprocess` already being mounted, but does NOT itself pass
the resolved service into the plugin body. The body still constructed its own PRIVATE
`LocalShell()` (whose constructor defaults to its own private `LocalSubprocess()`), silently
bypassing the Runtime-mounted `subprocess` capability entirely -- defeating the whole point of
`ctx.shell` depending on an independently-swappable `ctx.subprocess`. Corrected: `shell_plugin`
resolves `ctx.subprocess` (the SAME dynamic `Context.__getattr__` resolution every other
plugin's own body already uses, e.g. `llm/plugin.py`'s `ctx.llm.register(...)`) and passes it
explicitly as `LocalShell`'s own `subprocess_seam`. Per the SAME established convention
(`llm-mock` depends on `llm` being mounted first, not the runtime auto-resolving order), the
CALLER mounting these three plugins must mount `subprocess_plugin` before `shell_plugin`.
"""

from __future__ import annotations

from ..runtime import Context, plugin
from .filesystem import LocalFileSystem
from .shell import LocalShell
from .subprocess import LocalSubprocess, Subprocess


@plugin(name="execution-fs", provides="fs")
async def fs_plugin(ctx: Context, config: None) -> None:
    ctx.provide("fs", LocalFileSystem())


@plugin(name="execution-shell", provides="shell", inject=["subprocess"])
async def shell_plugin(ctx: Context, config: None) -> None:
    """Requires `subprocess_plugin` to be mounted FIRST (per `inject=["subprocess"]`'s own
    activation-gating convention) -- resolves the Runtime-mounted `ctx.subprocess` and injects
    it into `LocalShell`, rather than letting `LocalShell` construct its own private, unmounted
    `LocalSubprocess` (`L12-PY-R003`)."""
    subprocess_seam: Subprocess = ctx.subprocess
    ctx.provide("shell", LocalShell(subprocess_seam=subprocess_seam))


@plugin(name="execution-subprocess", provides="subprocess")
async def subprocess_plugin(ctx: Context, config: None) -> None:
    ctx.provide("subprocess", LocalSubprocess())
