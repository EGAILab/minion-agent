"""Scoped registration: tagged contexts that own what is registered through them.

The governing contract, from the design spec section 3: the registration
context determines both visibility and ownership. A registration can therefore
never be visible in one scope but disposed with another.

Nesting depth is the application's choice. The runtime guarantees arbitrary
nesting and key-agnostic tags; it defines no hierarchy of its own.
"""

from __future__ import annotations

from collections.abc import Callable
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


class ScopeTree:
    """Tracks every scope minted under one `Context` tree, by key.

    This is what lets `Scope.dispose()` settle its own live descendants
    before itself (RT-012) as part of its normal public behavior, rather
    than requiring a caller to walk the scope graph and dispose each scope
    individually. One tree is shared by a `Context` and every context
    `extend()`d from it, the same way `_registry`/`_events`/`_plugins` are
    shared — so any scope minted anywhere in the tree is visible to any
    other scope's disposal in the same tree.
    """

    __slots__ = ("_scopes",)

    def __init__(self) -> None:
        self._scopes: dict[ScopeKey, Scope] = {}

    def register(self, scope: Scope) -> None:
        """Track `scope` for descendant lookups. Re-minting under an
        already-used key (e.g. after that key's prior scope disposed)
        replaces the tracked entry, so a disposed scope is never mistaken
        for a live descendant."""
        self._scopes[scope.key] = scope

    def children_of(self, key: ScopeKey) -> list[Scope]:
        """Live scopes directly parented by `key`, creation order."""
        return [
            scope
            for scope in self._scopes.values()
            if scope.key.parent == key and not scope.disposed
        ]


class Scope:
    """A tagged context plus the disposables registered through it."""

    __slots__ = ("_disposables", "_disposed", "_tree", "ctx", "key", "on_disposed")

    def __init__(
        self,
        key: ScopeKey,
        ctx: Context,
        disposables: DisposableList,
        tree: ScopeTree,
    ) -> None:
        self.key = key
        self.ctx = ctx
        self._disposables = disposables
        self._disposed = False
        self._tree = tree
        self.on_disposed: Callable[[Scope], None] | None = None
        """Fired once, after this scope actually disposes -- whether disposed
        directly or as a still-live descendant swept in by an ancestor's
        disposal. Lets an embedder (or a test harness) observe disposal the
        same way `Fiber.on_state_change` lets one observe lifecycle
        transitions, without needing to walk the scope graph itself."""

    @property
    def disposed(self) -> bool:
        """Whether this scope has been disposed."""
        return self._disposed

    async def dispose(self) -> None:
        """Settle this scope's still-live descendants, deepest first, then
        unwind this scope's own registrations in reverse. Idempotent.

        Descendant ownership follows scope nesting, not the caller: disposing
        an ancestor here disposes its live descendants through this same
        method (design spec section 3, RT-012) -- a caller never has to
        compute or walk the scope tree to get correct cascading disposal.
        """
        if self._disposed:
            return
        self._disposed = True
        for child in reversed(self._tree.children_of(self.key)):
            await child.dispose()
        await self._disposables.dispose_all()
        if self.on_disposed is not None:
            self.on_disposed(self)


def scope_of(ctx: Context) -> ScopeKey | None:
    """The nearest scope key a context carries, or None if context-global."""
    return getattr(ctx, "_scope_key", None)
