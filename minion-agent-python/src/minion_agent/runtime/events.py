"""Typed events with four declared dispatch modes.

Dispatch mode is part of an event's public contract: it is declared where the
event is declared, and a mismatch between declaration and dispatch site is an
error rather than a silent behavior change.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from .errors import EventModeError, WaterfallError


class DispatchMode(StrEnum):
    """How an event's listeners are invoked and combined."""

    EMIT = "emit"
    """Fire and forget. Synchronous, registration order, no return value."""

    PARALLEL = "parallel"
    """Awaited, concurrent, no return value. Listener errors are aggregated."""

    SERIAL = "serial"
    """Awaited, registration order, returns a value."""

    WATERFALL = "waterfall"
    """Awaited around-middleware. A listener delegates via `next` or short-circuits."""


class _Listener:
    __slots__ = ("callback",)

    def __init__(self, callback: Callable[..., Any]) -> None:
        self.callback = callback


class EventBus:
    """Holds event declarations and their listeners."""

    __slots__ = ("_declarations", "_listeners")

    def __init__(self) -> None:
        self._declarations: dict[str, DispatchMode] = {}
        self._listeners: dict[str, list[_Listener]] = {}

    def declare(self, name: str, mode: DispatchMode) -> None:
        """Bind `name` to `mode`. Re-declaring the same mode is a no-op."""
        existing = self._declarations.get(name)
        if existing is None:
            self._declarations[name] = mode
            return
        if existing is not mode:
            raise EventModeError(
                f"event {name!r} is already declared {existing.value!r}; "
                f"cannot redeclare as {mode.value!r}"
            )

    def mode_of(self, name: str) -> DispatchMode:
        """Return the declared mode for `name`, raising when it is undeclared."""
        mode = self._declarations.get(name)
        if mode is None:
            raise EventModeError(f"event {name!r} is not declared")
        return mode

    def _require_mode(self, name: str, expected: DispatchMode) -> None:
        actual = self.mode_of(name)
        if actual is not expected:
            raise EventModeError(
                f"event {name!r} is declared {actual.value!r}; "
                f"it cannot be dispatched with {expected.value!r}"
            )

    def on(
        self,
        name: str,
        listener: Callable[..., Any],
        *,
        prepend: bool = False,
    ) -> Callable[[], None]:
        """Register `listener` for `name`; returns a disposer that removes it."""
        self.mode_of(name)
        entry = _Listener(listener)
        listeners = self._listeners.setdefault(name, [])
        if prepend:
            listeners.insert(0, entry)
        else:
            listeners.append(entry)

        def dispose() -> None:
            current = self._listeners.get(name)
            if current is not None and entry in current:
                current.remove(entry)

        return dispose

    def _chain(self, name: str) -> list[Callable[..., Any]]:
        return [entry.callback for entry in self._listeners.get(name, ())]

    def emit(self, name: str, *args: Any) -> None:
        """Invoke every listener synchronously in registration order."""
        self._require_mode(name, DispatchMode.EMIT)
        for callback in self._chain(name):
            callback(*args)

    @staticmethod
    async def _call(callback: Callable[..., Any], *args: Any) -> Any:
        result = callback(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def parallel(self, name: str, *args: Any) -> None:
        """Invoke every listener concurrently, aggregating any failures."""
        self._require_mode(name, DispatchMode.PARALLEL)
        callbacks = self._chain(name)
        if not callbacks:
            return
        outcomes = await asyncio.gather(
            *(self._call(callback, *args) for callback in callbacks),
            return_exceptions=True,
        )
        failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
        if failures:
            raise ExceptionGroup(f"errors in {name!r} listeners", failures)

    async def serial(self, name: str, *args: Any) -> Any:
        """Invoke listeners in registration order; the last value wins."""
        self._require_mode(name, DispatchMode.SERIAL)
        result: Any = None
        for callback in self._chain(name):
            result = await self._call(callback, *args)
        return result

    async def waterfall(self, name: str, *args: Any, terminal: Any = None) -> Any:
        """Invoke listeners as around-middleware.

        Each listener receives `next` as its final positional argument.
        `next()` delegates with the current arguments; `next(*replacement)`
        delegates with replacements. Returning without calling `next`
        short-circuits, and that return value becomes the chain result.

        Delegating past the last listener yields `terminal`, as does an empty
        chain. Events declare their own terminal so that a chain whose
        listeners all cooperatively delegate returns the transformed payload
        rather than None — the transformation pattern depends on it.

        `next` may be called at most once; a second call raises. Memoizing it
        instead would be incoherent, since a second call may carry different
        replacement arguments.
        """
        self._require_mode(name, DispatchMode.WATERFALL)
        callbacks = self._chain(name)

        async def step(index: int, current: tuple[Any, ...]) -> Any:
            if index >= len(callbacks):
                return terminal

            used = False

            async def next_(*replacement: Any) -> Any:
                nonlocal used
                if used:
                    raise WaterfallError(
                        f"`next` may be called at most once per listener "
                        f"(event {name!r}, listener index {index})"
                    )
                used = True
                return await step(index + 1, replacement or current)

            return await self._call(callbacks[index], *current, next_)

        return await step(0, args)
