"""The context: a repository of services and the surface plugins are handed.

Attribute access (`ctx.tools`) is the ergonomic door; `ctx.require(ToolService)`
is the statically-checked one. Both resolve through one mechanism keyed by
service name — the protocol is a typed view, never a second key space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from .errors import ServiceNotFoundError
from .events import EventBus
from .service import ServiceRegistry

if TYPE_CHECKING:
    from .fiber import Fiber

T = TypeVar("T")

_RESERVED = frozenset({"registry", "events", "root", "fiber"})


class Context:
    """A view onto the shared service registry and event bus."""

    def __init__(self) -> None:
        self._registry = ServiceRegistry()
        self._events = EventBus()
        self._root: Context = self
        self._fiber: Fiber | None = None
        self._meta: dict[str, Any] = {}

    @property
    def registry(self) -> ServiceRegistry:
        return self._registry

    @property
    def events(self) -> EventBus:
        return self._events

    @property
    def root(self) -> Context:
        return self._root

    @property
    def fiber(self) -> Fiber | None:
        return self._fiber

    def extend(self, **meta: Any) -> Context:
        """Create a child context sharing this one's registry and bus."""
        child = object.__new__(Context)
        child._registry = self._registry
        child._events = self._events
        child._root = self._root
        child._fiber = meta.pop("fiber", self._fiber)
        child._meta = {**self._meta, **meta}
        return child

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in _RESERVED:
            raise AttributeError(name)

        meta = self.__dict__.get("_meta", {})
        if name in meta:
            return meta[name]

        registry: ServiceRegistry | None = self.__dict__.get("_registry")
        if registry is not None:
            value = registry.resolve(name)
            if value is not None:
                return value
            raise ServiceNotFoundError(f"no active provider for service {name!r}")
        raise AttributeError(name)

    def require(self, protocol: type[T]) -> T:
        """Resolve the service `protocol` declares, by name."""
        name = getattr(protocol, "__service_name__", None)
        if not isinstance(name, str):
            raise TypeError(
                f"{protocol!r} does not declare __service_name__; "
                "a service protocol must name the service it describes"
            )
        value = self._registry.resolve(name)
        if value is None:
            raise ServiceNotFoundError(f"no active provider for service {name!r}")
        return value  # type: ignore[no-any-return]
