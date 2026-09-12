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
        return await asyncio.to_thread(Path(path).expanduser().exists)
