"""The context: a repository of services and the surface plugins are handed.

Attribute access (`ctx.tools`) is the ergonomic door; `ctx.require(ToolService)`
is the statically-checked one. Both resolve through one mechanism keyed by
service name — the protocol is a typed view, never a second key space.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, TypeVar

from .disposable import Disposer
from .errors import ServiceNotFoundError
from .events import EventBus
from .plugin import spec_of
from .registry import PluginRegistry
from .service import ServiceRegistry

if TYPE_CHECKING:
    from .fiber import Fiber

T = TypeVar("T")

_RESERVED = frozenset({"registry", "events", "root", "fiber", "plugins"})


class Context:
    """A view onto the shared service registry and event bus."""

    def __init__(self) -> None:
        self._registry = ServiceRegistry()
        self._events = EventBus()
        self._root: Context = self
        self._fiber: Fiber | None = None
        self._meta: dict[str, Any] = {}
        self._plugins = PluginRegistry(self)

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
        child._plugins = self._plugins
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

    @property
    def plugins(self) -> PluginRegistry:
        """The plugin registry shared by this context tree."""
        return self._plugins

    async def plugin(self, candidate: Any, config: Any = None) -> Fiber:
        """Mount `candidate` as a plugin and reconcile dependencies.

        Config is validated against the plugin's declared model before the
        fiber is created, so an invalid config never runs a plugin body.
        """
        spec = spec_of(candidate)
        resolved = config
        if spec.config_model is not None:
            resolved = spec.config_model.model_validate(config or {})
        fiber = self._plugins.mount(spec, resolved, self)
        await self._plugins.reconcile()
        return fiber

    def effect(
        self,
        execute: Callable[[], Disposer | AbstractContextManager[Any] | None],
        label: str | None = None,
    ) -> Callable[[], Any]:
        """Register a reversible effect on the owning fiber."""
        if self._fiber is None:
            raise RuntimeError("ctx.effect() requires a fiber; call it inside a plugin")
        return self._fiber.effect(execute, label)

    def on(
        self,
        name: str,
        listener: Callable[..., Any],
        *,
        prepend: bool = False,
    ) -> Callable[[], Any]:
        """Register an event listener, auto-disposed with the owning fiber."""
        if self._fiber is None:
            return self._events.on(name, listener, prepend=prepend)
        return self.effect(
            lambda: self._events.on(name, listener, prepend=prepend),
            f"on({name})",
        )

    def provide(
        self,
        name: str,
        value: Any,
        check: Callable[[], bool] | None = None,
    ) -> Callable[[], Any]:
        """Provide a service, withdrawn when the owning fiber unloads.

        Revocation reconciles, so a dependent unloads as soon as the service
        it needs disappears.
        """
        if self._fiber is None:
            raise RuntimeError("ctx.provide() requires a fiber; call it inside a plugin")
        fiber = self._fiber
        plugins = self._plugins
        registry = self._registry

        def register() -> Disposer:
            revoke = registry.provide(name, value, fiber, check)

            async def undo() -> None:
                revoke()
                await plugins.reconcile()

            return undo

        return self.effect(register, f"provide({name})")

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
