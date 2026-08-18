"""Scope-aware storage for services that hold registrations.

The runtime decides eligibility; the owning service decides composition.
This helper answers only "which entries are visible from here", nearest scope
first, and deliberately takes no position on same-name collisions: a keyed
registry shadows by taking the first match, an additive one keeps every entry.
"""

from __future__ import annotations

from collections.abc import Callable

from .scope import ScopeKey


class ScopedRegistry[V]:
    """Entries filed by scope, queried by visibility."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        # Insertion-ordered per scope; None keys the untagged (global) bucket.
        self._entries: dict[ScopeKey | None, list[tuple[str, V] | None]] = {}

    def __len__(self) -> int:
        return sum(1 for bucket in self._entries.values() for entry in bucket if entry is not None)

    def add(self, key: ScopeKey | None, name: str, value: V) -> Callable[[], None]:
        """File `value` under `key`; returns a handle that withdraws it."""
        bucket = self._entries.setdefault(key, [])
        index = len(bucket)
        bucket.append((name, value))

        def remove() -> None:
            bucket[index] = None

        return remove

    def visible_from(self, key: ScopeKey | None) -> list[tuple[str, V]]:
        """Entries visible from `key`: its own, then ancestors', then untagged.

        Nearest-first ordering is what lets a keyed registry shadow by taking
        the first match for a name.
        """
        chain: tuple[ScopeKey | None, ...] = key.chain() if key is not None else ()
        out: list[tuple[str, V]] = []
        for scope in (*chain, None):
            for entry in self._entries.get(scope, ()):
                if entry is not None:
                    out.append(entry)
        return out
