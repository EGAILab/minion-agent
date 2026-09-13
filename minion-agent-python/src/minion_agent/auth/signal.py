"""Additive, auth-owned signal composition (`PROV-008`; Pi `auth/resolve.ts:149-153`).

Pi composes the OAuth refresh call's own cancellation with `AbortSignal.any([signal,
AbortSignal.timeout(DEFAULT_OAUTH_REFRESH_TIMEOUT_MS)])` -- an OR of the caller's own signal and a
fixed wall-clock budget. Certified Layer 09's own `RunSignal` is deliberately poll-only and has no
composition primitive of its own (`runtime/signal.py`), and this module does not add one there --
Layer 09 is not reopened by this pass. Instead, this is the smallest additive seam scoped entirely
to `auth`'s own refresh-authority module: a structural `Abortable` protocol every poll-based signal
already satisfies (`RunSignal` included, duck-typed, no inheritance required), and `CombinedSignal`,
a second, auth-owned implementation of that same protocol.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol


class Abortable(Protocol):
    """The one property every abort-aware auth helper actually polls. `RunSignal` (Layer 09)
    already satisfies this structurally; `CombinedSignal` below is a second, independent
    implementation -- neither needs to inherit from the other."""

    @property
    def aborted(self) -> bool: ...


class CombinedSignal:
    """Aborted once EITHER an optional caller signal aborts OR a fixed budget elapses (Pi
    `AbortSignal.any([signal, AbortSignal.timeout(ms)])`). `now` is injectable so tests can prove
    the budget deterministically, with no real waiting."""

    __slots__ = ("_deadline", "_now", "_signal")

    def __init__(
        self,
        signal: Abortable | None,
        timeout_seconds: float,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._signal = signal
        self._now = now
        self._deadline = now() + timeout_seconds

    @property
    def aborted(self) -> bool:
        if self._signal is not None and self._signal.aborted:
            return True
        return self._now() >= self._deadline
