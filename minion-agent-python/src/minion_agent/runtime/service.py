"""The service registry: exclusive registration, active-gated visibility.

Resolution rules are normative (design spec, section 3):

* Identity is the service name. There is exactly one slot per name.
* Registration is exclusive; a second provider raises rather than winning.
* There is no fallback stack. Revoking frees the name; it does not reveal
  an earlier provider, because none was retained.
* Visibility is narrower than registration: a service resolves only while its
  owning fiber is ACTIVE and its optional `check` predicate holds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import ServiceConflictError
from .fiber import FiberState


@dataclass(slots=True)
class Impl:
    """One service registration."""

    name: str
    value: Any
    owner: Any
    check: Callable[[], bool] | None = None

    def is_visible(self) -> bool:
        """Whether this registration currently resolves."""
        if getattr(self.owner, "state", None) is not FiberState.ACTIVE:
            return False
        return self.check is None or bool(self.check())


class ServiceRegistry:
    """Maps service names to their single registration."""

    __slots__ = ("_impls",)

    def __init__(self) -> None:
        self._impls: dict[str, Impl] = {}

    def provide(
        self,
        name: str,
        value: Any,
        owner: Any,
        check: Callable[[], bool] | None = None,
    ) -> Callable[[], None]:
        """Register `name`; returns a disposer that revokes it.

        Raises `ServiceConflictError` when the name is already held, naming the
        fiber that holds it.
        """
        existing = self._impls.get(name)
        if existing is not None:
            holder = getattr(existing.owner, "name", "<unknown>")
            raise ServiceConflictError(f"service {name!r} has been registered at <{holder}>")

        impl = Impl(name=name, value=value, owner=owner, check=check)
        self._impls[name] = impl

        def revoke() -> None:
            if self._impls.get(name) is impl:
                del self._impls[name]

        return revoke

    def impl_of(self, name: str) -> Impl | None:
        """Return the registration for `name` regardless of visibility."""
        return self._impls.get(name)

    def resolve(self, name: str, *, strict: bool = True) -> Any | None:
        """Return the service value, or None when it does not currently resolve.

        `strict=False` bypasses the ACTIVE-state gate but still honors `check`.
        """
        impl = self._impls.get(name)
        if impl is None:
            return None
        if strict and not impl.is_visible():
            return None
        if not strict and impl.check is not None and not impl.check():
            return None
        return impl.value

    def has(self, name: str) -> bool:
        """Whether `name` currently resolves."""
        return self.resolve(name) is not None

    def names(self) -> frozenset[str]:
        """Every registered service name, visible or not."""
        return frozenset(self._impls)
