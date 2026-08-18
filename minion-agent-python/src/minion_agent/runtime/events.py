"""Typed events with four declared dispatch modes.

Dispatch mode is part of an event's public contract: it is declared where the
event is declared, and a mismatch between declaration and dispatch site is an
error rather than a silent behavior change.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from .errors import EventModeError


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
