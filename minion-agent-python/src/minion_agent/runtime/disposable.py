"""Reverse-ordered disposal, the primitive behind reversible effects."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

type Disposer = Callable[[], Awaitable[None] | None]


class DisposableList:
    """An ordered collection of disposers, unwound last-in-first-out.

    Reverse order is normative: a later effect may depend on an earlier one,
    so tearing down in creation order could observe a half-disposed world.
    """

    __slots__ = ("_disposed", "_disposers")

    def __init__(self) -> None:
        self._disposers: list[Disposer | None] = []
        self._disposed = False

    def __len__(self) -> int:
        return sum(1 for disposer in self._disposers if disposer is not None)

    @property
    def disposed(self) -> bool:
        """Whether `dispose_all` has run."""
        return self._disposed

    def push(self, disposer: Disposer) -> Callable[[], None]:
        """Register `disposer`; returns a handle that removes it without running it."""
        index = len(self._disposers)
        self._disposers.append(disposer)

        def remove() -> None:
            self._disposers[index] = None

        return remove

    async def dispose_all(self) -> None:
        """Run every disposer in reverse order, exactly once.

        Every disposer runs even if one raises; failures are collected and
        re-raised together, because a disposer that fails must not strand the
        ones queued behind it.
        """
        if self._disposed:
            return
        self._disposed = True

        failures: list[Exception] = []
        for disposer in reversed(self._disposers):
            if disposer is None:
                continue
            try:
                result: Any = disposer()
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                failures.append(error)

        self._disposers.clear()
        if failures:
            raise ExceptionGroup("errors while disposing", failures)
