"""Content-addressed storage for large, mostly-stable request components.

A resident agent's system prompt is large, mostly stable, and partly dynamic.
Snapshotting it every step does not scale, and whole-header change detection
does not help — in practice something changes nearly every step, so a
530-token change would force a 19,000-token snapshot.

Storing components by content hash means a stable block is stored once for the
life of a session, however often its neighbours change.

Artifacts holding model-visible content are never deleted, inheriting the
discipline that governs the log itself. There is deliberately no removal API:
no artifact may be reclaimed while any request header references it, and the
log never stops referencing one.
"""

from __future__ import annotations

import hashlib

_PREFIX = "sha256:"


class MissingArtifactError(KeyError):
    """A reference named content this store does not hold."""


class ArtifactStore:
    """Maps content hashes to content."""

    def __init__(self) -> None:
        self._content: dict[str, bytes] = {}

    def __len__(self) -> int:
        return len(self._content)

    @staticmethod
    def _as_bytes(content: str | bytes) -> bytes:
        return content.encode("utf-8") if isinstance(content, str) else content

    def put(self, content: str | bytes) -> str:
        """Store `content` and return its reference. Idempotent."""
        raw = self._as_bytes(content)
        ref = _PREFIX + hashlib.sha256(raw).hexdigest()
        self._content.setdefault(ref, raw)
        return ref

    def get(self, ref: str) -> bytes:
        """Return the content `ref` names."""
        try:
            return self._content[ref]
        except KeyError:
            raise MissingArtifactError(f"no artifact for {ref}") from None

    def has(self, ref: str) -> bool:
        """Whether this store holds `ref`."""
        return ref in self._content
