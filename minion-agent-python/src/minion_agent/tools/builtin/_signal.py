"""Pi's built-in tools reject with `"Operation aborted"` the moment their `AbortSignal` fires, while
their own async work carries on untouched (`read.ts:230-249`, `ls.ts:111-178`): the rejection does
not cancel an in-flight filesystem call, and that work later stops at the tool's own checkpoints
(`read`) or runs to completion (`ls`) with its result discarded. `RunSignal` is poll-based
(`runtime/signal.py`), so the work is raced against it here -- and, when the abort wins, the work is
left running rather than cancelled (`L13-WP131-I001`; the Layer 09 signal contract never forcibly
interrupts tool work)."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from ...runtime.signal import RunSignal
from .paths import aborted

_POLL_INTERVAL_S = 0.01

_ABANDONED: set[asyncio.Future[Any]] = set()
"""Work whose caller was already answered `"Operation aborted"`. Held here so the event loop keeps
running it to its own end (a task nothing references may be garbage-collected mid-flight)."""


def _discard(task: asyncio.Future[Any]) -> None:
    _ABANDONED.discard(task)
    if not task.cancelled():
        task.exception()  # retrieved: a late failure of discarded work is not an error


async def _settle[T](work: Coroutine[Any, Any, T], signal: RunSignal) -> T:
    """The work's settle point (`L13-WP131-I001`, R-I1): the outcome is decided by ORDER. Pi's
    abort listener rejects the tool's promise synchronously, so an abort that happened before the
    work settles wins -- including over a failure the work throws after the abort (read:
    `if (!aborted) reject(error)`; ls: the promise is already rejected). The check below is the
    last synchronous step before the result or failure leaves the work, so the poll interval can
    delay the answer but never change it."""
    try:
        result = await work
    except BaseException:
        if signal.aborted:
            raise aborted() from None
        raise
    if signal.aborted:
        raise aborted()
    return result


async def race_abort[T](work: Coroutine[Any, Any, T], signal: RunSignal | None) -> T:
    """Run `work`; if `signal` aborts first, raise `"Operation aborted"` immediately and leave
    `work` running. The race only decides how SOON the abort is answered; WHICH outcome stands is
    decided at the work's settle point (`_settle`). A result that settled before the abort is
    returned even if the abort lands before this coroutine resumes -- Pi's promise is already
    resolved by then."""
    if signal is None:
        return await work
    if signal.aborted:
        work.close()
        raise aborted()
    task = asyncio.ensure_future(_settle(work, signal))

    async def watch() -> None:
        while not signal.aborted:
            await asyncio.sleep(_POLL_INTERVAL_S)

    watcher = asyncio.ensure_future(watch())
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watcher.cancel()
    if task.done():
        return task.result()
    _ABANDONED.add(task)
    task.add_done_callback(_discard)
    raise aborted()
