"""Pi's built-in tools reject with `"Operation aborted"` the moment their `AbortSignal` fires, even
while their own work is still in flight (`read.ts:230-241`, `ls.ts:111-125`). `RunSignal` is
poll-based (`runtime/signal.py`), so the tool's work is raced against it here, the same way Layer
12's `_race_signal` settles a blocked read promptly."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from ...runtime.signal import RunSignal
from .paths import aborted

_POLL_INTERVAL_S = 0.01


async def race_abort[T](work: Coroutine[Any, Any, T], signal: RunSignal | None) -> T:
    """Run `work`; if `signal` aborts first, cancel it and raise `"Operation aborted"`. An abort
    after `work` has settled changes nothing -- Pi's result is already resolved by then."""
    if signal is None:
        return await work
    if signal.aborted:
        work.close()
        raise aborted()
    task = asyncio.ensure_future(work)

    async def watch() -> None:
        while not signal.aborted:
            await asyncio.sleep(_POLL_INTERVAL_S)

    watcher = asyncio.ensure_future(watch())
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watcher.cancel()
        settled = task.done()
        if not settled:
            task.cancel()
    if not settled:
        raise aborted()
    return task.result()
