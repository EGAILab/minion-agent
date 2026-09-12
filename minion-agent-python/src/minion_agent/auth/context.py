"""The default `AuthContext` reference implementation (`PROV-006`; Pi
`defaultProviderAuthContext`, `auth/context.ts:23-45`).

Ported without Pi's own browser/Node conditional -- Minion has no browser target, so this always
reads `os.environ` and checks the real filesystem. Deliberately does NOT read any provider- or
CLI-specific credential file; that remains a provider's own login-flow concern, never baked into
this generic context (design instruction: auth must not collapse into "read one well-known file").
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


class DefaultAuthContext:
    """Reads environment variables from `os.environ`; checks file existence with `~` expansion.
    An empty or whitespace-only environment value is treated as absent, matching Pi exactly."""

    async def env(self, name: str) -> str | None:
        value = os.environ.get(name)
        return value if value and value.strip() else None

    async def file_exists(self, path: str) -> bool:
        """`L11-R013` (remediated): matches Pi's own `fileExists` exactly, not merely its general
        shape. Two corrections from a prior revision:

        1. ANY-leading-`~` expansion is literal string concatenation (Pi: `resolved.startsWith("~")
           ? homedir() + resolved.slice(1) : resolved`) -- the home directory replaces ONLY the
           leading `~` character, with everything after it appended UNCHANGED, no path-join
           insertion and no other-user (`~username`) lookup semantics. This is deliberately NOT
           `Path(path).expanduser()`, whose own platform-specific conventions (a POSIX `~username`
           lookup, or Windows' own differing interpretation of a bare `~` prefix followed by
           non-separator characters) do not match Pi's naive concatenation and can diverge
           observably for an input like `~suffix` (no separator after `~`).
        2. The WHOLE operation -- module/path resolution and the filesystem access itself -- is
           one failure boundary that returns `False` on any error (Pi's own `try { ... } catch {
           return false; }`), not only "the target does not exist." A permission error or other
           filesystem failure must report `False`, the same as a genuinely missing path -- never
           propagate the underlying exception.
        """
        resolved = path
        if resolved.startswith("~"):
            resolved = os.path.expanduser("~") + resolved[1:]
        try:
            return await asyncio.to_thread(Path(resolved).exists)
        except OSError:
            return False
