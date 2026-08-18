"""The plugin registry: mounting, unmounting, and dependency reconciliation.

Reconciliation is the reactive mechanism. After any change to the set of
visible services, every mounted fiber is re-evaluated: satisfied fibers load,
unsatisfied ones unload. Repeating until the world stops changing lets one
service's arrival cascade through a chain of dependents.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from .fiber import Fiber, FiberState

if TYPE_CHECKING:
    from .context import Context
    from .plugin import PluginSpec

_MAX_PASSES = 100


class PluginRegistry:
    """Owns every mounted fiber and keeps their load state consistent."""

    __slots__ = ("_fibers", "_root")

    def __init__(self, root: Context) -> None:
        self._root = root
        self._fibers: list[Fiber] = []

    @property
    def fibers(self) -> tuple[Fiber, ...]:
        return tuple(self._fibers)

    def mount(self, spec: PluginSpec, config: Any, parent: Context) -> Fiber:
        """Create a fiber for `spec`. It loads during the next reconcile."""
        fiber = Fiber(name=spec.name, parent=parent, plugin=spec, config=config)
        self._fibers.append(fiber)
        return fiber

    async def unmount(self, fiber: Fiber) -> None:
        """Dispose `fiber` and reconcile the fibers that depended on it."""
        if fiber in self._fibers:
            self._fibers.remove(fiber)
        await fiber.dispose()
        await self.reconcile()

    def _is_satisfied(self, fiber: Fiber) -> bool:
        registry = self._root.registry
        return all(registry.has(name) for name in fiber.inject)

    async def reconcile(self) -> None:
        """Load satisfied fibers and unload unsatisfied ones until stable."""
        for _ in range(_MAX_PASSES):
            changed = False

            for fiber in list(self._fibers):
                if fiber.state is FiberState.ACTIVE and not self._is_satisfied(fiber):
                    await fiber.unload()
                    changed = True

            for fiber in list(self._fibers):
                if fiber.state is FiberState.PENDING and self._is_satisfied(fiber):
                    # Re-checked at commit: the body may itself withdraw a
                    # service this fiber depends on.
                    await fiber.load(validate=partial(self._is_satisfied, fiber))
                    changed = True

            if not changed:
                return

        raise RuntimeError(  # pragma: no cover - a cycle in plugin dependencies
            "plugin reconciliation did not stabilize; check for a dependency cycle"
        )
