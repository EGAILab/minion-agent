"""Scoped registration: tagged contexts that own what is registered through them.

The governing contract, from the design spec section 3: the registration
context determines both visibility and ownership. A registration can therefore
never be visible in one scope but disposed with another.

Nesting depth is the application's choice. The runtime guarantees arbitrary
nesting and key-agnostic tags; it defines no hierarchy of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .disposable import DisposableList

if TYPE_CHECKING:
    from .context import Context


@dataclass(frozen=True, slots=True)
class ScopeKey:
    """An opaque scope identity with an optional parent.

    Parents are fixed at construction, so a cycle is unrepresentable and no
    rebinding protocol is needed.
    """

    name: str
    parent: ScopeKey | None = None

    def chain(self) -> tuple[ScopeKey, ...]:
        """This key then its ancestors, nearest first."""
        out: list[ScopeKey] = []
        node: ScopeKey | None = self
        while node is not None:
            out.append(node)
            node = node.parent
        return tuple(out)


class Scope:
    """A tagged context plus the disposables registered through it."""

    __slots__ = ("_disposables", "_disposed", "ctx", "key")

    def __init__(self, key: ScopeKey, ctx: Context, disposables: DisposableList) -> None:
        self.key = key
        self.ctx = ctx
        self._disposables = disposables
        self._disposed = False

    @property
    def disposed(self) -> bool:
        """Whether this scope has been disposed."""
        return self._disposed

    async def dispose(self) -> None:
        """Unwind this scope's registrations in reverse. Idempotent."""
        if self._disposed:
            return
        self._disposed = True
        await self._disposables.dispose_all()


def scope_of(ctx: Context) -> ScopeKey | None:
    """The nearest scope key a context carries, or None if context-global."""
    return getattr(ctx, "_scope_key", None)
