"""The plugin runtime: contexts, fibers, services, events, and reversible effects.

Cordis-semantic, not a Cordis port: faithful to Cordis's semantics, not its
mechanics. See the design spec, section 3.
"""

from .context import Context
from .disposable import DisposableList, Disposer
from .errors import (
    EventModeError,
    InactiveFiberError,
    RuntimeError_,
    ServiceConflictError,
    ServiceNotFoundError,
    WaterfallError,
)
from .events import DispatchMode, EventBus
from .fiber import Fiber, FiberState
from .plugin import PluginSpec, plugin, spec_of
from .registry import PluginRegistry
from .scope import Scope, ScopeKey, ScopeTree, scope_of
from .scoped_registry import ScopedRegistry
from .service import Impl, ServiceRegistry

__all__ = [
    "Context",
    "DispatchMode",
    "DisposableList",
    "Disposer",
    "EventBus",
    "EventModeError",
    "Fiber",
    "FiberState",
    "Impl",
    "InactiveFiberError",
    "PluginRegistry",
    "PluginSpec",
    "RuntimeError_",
    "Scope",
    "ScopeKey",
    "ScopeTree",
    "ScopedRegistry",
    "ServiceConflictError",
    "ServiceNotFoundError",
    "ServiceRegistry",
    "WaterfallError",
    "plugin",
    "scope_of",
    "spec_of",
]
