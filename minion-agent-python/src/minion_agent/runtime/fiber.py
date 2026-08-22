"""Fibers: one loaded plugin instance, its lifecycle, config, and effects."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .disposable import DisposableList, Disposer
from .errors import InactiveFiberError

if TYPE_CHECKING:
    from .context import Context
    from .plugin import PluginSpec


class FiberState(StrEnum):
    """Lifecycle state of one plugin instance."""

    PENDING = "pending"
    """Mounted but not satisfied: at least one injected service is missing."""

    LOADING = "loading"
    """Dependencies satisfied; the plugin body is running."""

    ACTIVE = "active"
    """Loaded. Its services are visible and its effects are live."""

    FAILED = "failed"
    """The plugin body raised. Effects created before the failure are unwound."""

    UNLOADING = "unloading"
    """Effects are being disposed."""

    DISPOSED = "disposed"
    """Terminal. The fiber cannot be reused."""


_LIVE_STATES = frozenset({FiberState.LOADING, FiberState.ACTIVE})


class Fiber:
    """One loaded plugin instance: its state, config, and reversible effects."""

    def __init__(
        self,
        *,
        name: str,
        parent: Context,
        plugin: PluginSpec,
        config: Any,
    ) -> None:
        self.name = name
        self.plugin = plugin
        self.config = config
        self.inject: tuple[str, ...] = plugin.inject
        self._state = FiberState.PENDING
        self._disposables = DisposableList()
        self.on_state_change: Callable[[Fiber, FiberState], None] | None = None
        self.on_effect: Callable[[Fiber, str, str], None] | None = None
        self.ctx = parent.extend(fiber=self)

    @property
    def state(self) -> FiberState:
        return self._state

    def _transition(self, state: FiberState) -> None:
        self._state = state
        if self.on_state_change is not None:
            self.on_state_change(self, state)

    def effect(
        self,
        execute: Callable[[], Disposer | AbstractContextManager[Any] | None],
        label: str | None = None,
    ) -> Callable[[], Awaitable[None]]:
        """Run `execute` now and collect its disposer.

        `execute` may return a disposer, a context manager, or None. The
        returned disposer tears this effect down; calling it twice is a no-op,
        and the fiber's own unload runs it if it has not already.
        """
        if self._state not in _LIVE_STATES:
            raise InactiveFiberError(
                f"cannot create effect on fiber <{self.name}> in state {self._state.value!r}"
            )

        outcome = execute()
        disposer: Disposer | None
        if outcome is None:
            disposer = None
        elif isinstance(outcome, AbstractContextManager):
            outcome.__enter__()

            def disposer() -> None:
                outcome.__exit__(None, None, None)
        else:
            disposer = outcome

        effect_label = label or "<unlabelled>"
        if self.on_effect is not None:
            self.on_effect(self, "created", effect_label)

        settled = False

        async def run_disposer() -> None:
            nonlocal settled
            if settled:
                return
            settled = True
            if disposer is not None:
                result = disposer()
                if inspect.isawaitable(result):
                    await result
            if self.on_effect is not None:
                self.on_effect(self, "disposed", effect_label)

        remove = self._disposables.push(run_disposer)

        async def dispose() -> None:
            remove()
            await run_disposer()

        return dispose

    async def load(self, validate: Callable[[], bool] | None = None) -> None:
        """Run the plugin body, transitioning PENDING -> LOADING -> ACTIVE.

        Loading is a transaction over owned effects. `validate` is re-checked
        after the body returns: a dependency can disappear while the body runs,
        and an attempt invalidated that way must not commit. The effects it
        created unwind in reverse and the fiber settles back at PENDING —
        without passing through ACTIVE or UNLOADING, because a stale attempt
        was never active and has nothing to unload.
        """
        if self._state is not FiberState.PENDING:
            return
        self._transition(FiberState.LOADING)
        try:
            result = self.plugin.apply(self.ctx, self.config)
            if inspect.isawaitable(result):
                await result
        except Exception:
            # The state settles even if unwind itself raises (a disposer
            # failing on top of the body's own failure) -- a fiber stuck at
            # LOADING forever would be worse than a disposal failure that
            # still propagates to the caller after the state is final.
            try:
                await self._unwind()
            finally:
                self._transition(FiberState.FAILED)
            return

        if validate is not None and not validate():
            try:
                await self._unwind()
            finally:
                self._transition(FiberState.PENDING)
            return

        self._transition(FiberState.ACTIVE)

    async def _unwind(self) -> None:
        """Dispose this fiber's effects and start a fresh disposable list."""
        try:
            await self._disposables.dispose_all()
        finally:
            self._disposables = DisposableList()

    async def unload(self) -> None:
        """Unwind effects and return to PENDING, ready to load again.

        A no-op on FAILED. That phase is stable: it has no restart operation,
        so recovery is disposal followed by a fresh mount. Returning a failed
        fiber to PENDING would let the next reconcile silently re-run an
        initialization that already reported failure.
        """
        if self._state not in _LIVE_STATES:
            return
        self._transition(FiberState.UNLOADING)
        try:
            await self._unwind()
        finally:
            # Reached even if a disposer raised: UNLOADING must not be a
            # trap a failing disposer can strand the fiber in forever. The
            # aggregated ExceptionGroup from _unwind() still propagates to
            # the caller after this settles -- the failure is surfaced, not
            # swallowed, the same principle DisposableList itself applies to
            # individual disposers.
            self._transition(FiberState.PENDING)

    async def dispose(self) -> None:
        """Unwind effects and enter the terminal DISPOSED state.

        Only an active fiber announces UNLOADING. The lifecycle names that
        phase for teardown of a fiber that was serving, and the normative
        transitions read `Failed -> dispose -> Disposed` and
        `Loading -> dispose -> unwind -> Disposed` (design spec section 3): a
        failed fiber's effects were already unwound when it failed, and an
        interrupted loading attempt was never active. Announcing a phase with
        nothing to do would make a second implementation reproduce a
        transition the specification does not describe.
        """
        if self._state is FiberState.DISPOSED:
            return
        unwind = self._state in _LIVE_STATES
        if self._state is FiberState.ACTIVE:
            self._transition(FiberState.UNLOADING)
        try:
            if unwind:
                await self._unwind()
        finally:
            # DISPOSED is terminal and must always be reached, even if
            # unwind fails -- the same reasoning as unload() above.
            self._transition(FiberState.DISPOSED)
