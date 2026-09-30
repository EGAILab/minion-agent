"""The shared `write`/`edit` mutation queue (`TOOL-032`; pinned Pi
`core/tools/file-mutation-queue.ts`).

spec/tools.md WP-13.2 "Mutation queue":
- REGISTRATION is one critical section, globally ordered across every pending `write`/`edit` call
  of the process, in call order. A later registration (including its `canonical_path`) does not
  begin until every earlier one has SETTLED, whether it succeeded or failed (Pi's
  `registrationQueue` chain; `L13-WP132-R003`).
- The key is `ctx.fs.canonical_path(p)`, falling back to `absolute_path(p)` on `not_found`,
  `not_directory` (Pi's `ENOENT`/`ENOTDIR`) and `not_supported` (a provider that cannot
  canonicalize). Any other failure fails registration and leaves no entry. Layer 12's `resolve()`
  is deliberately NOT used: its fallback set omits `not_directory` (`minion-agent#78`).
- Queues are scoped per `ctx.fs` provider instance (`MINION_ARCHITECTURAL_MAPPING`); within one
  provider, equal keys share one FIFO; different keys run concurrently.
- The entry is released when `fn` returns or raises, and the key's queue is dropped once empty.
No `ctx.fs` call here receives a signal, and nothing here listens for aborts (`TOOL-033`).
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ...execution import Err, FileSystem, FsErrorCode
from .paths import BuiltinToolError, cause

_FALLBACK_CODES = (FsErrorCode.NOT_FOUND, FsErrorCode.NOT_DIRECTORY, FsErrorCode.NOT_SUPPORTED)


@dataclass
class _State:
    """One event loop's queues. `registration` is the tail of the global registration chain;
    `tails[(provider, key)]` is the release future of the last entry queued for that key."""

    registration: asyncio.Future[None] | None = None
    tails: dict[tuple[int, str], asyncio.Future[None]] = field(default_factory=dict)


_STATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _State] = weakref.WeakKeyDictionary()


def _state() -> _State:
    loop = asyncio.get_running_loop()
    state = _STATES.get(loop)
    if state is None:
        state = _STATES[loop] = _State()
    return state


async def mutation_queue_key(fs: FileSystem, p: str, path: str) -> str:
    """`getMutationQueueKey`. `p` is the preprocessed path, `path` the argument as given (for the
    `"Cannot resolve <path>: <cause>"` text, `TOOL-039` / `L13-WP132-O1`)."""
    canonical = await fs.canonical_path(p)
    if not isinstance(canonical, Err):
        return canonical.value
    if canonical.error.code not in _FALLBACK_CODES:
        raise BuiltinToolError(f"Cannot resolve {path}: {cause(canonical.error.code)}")
    absolute = await fs.absolute_path(p)
    if isinstance(absolute, Err):
        raise BuiltinToolError(f"Cannot resolve {path}: {cause(absolute.error.code)}")
    return absolute.value


async def with_mutation_queue[T](
    fs: FileSystem, p: str, path: str, fn: Callable[[], Awaitable[T]]
) -> T:
    """`withFileMutationQueue(p, fn)` for provider `fs`."""
    loop = asyncio.get_running_loop()
    state = _state()
    previous_registration = state.registration
    registered = loop.create_future()
    state.registration = registered
    try:
        if previous_registration is not None:
            await asyncio.shield(previous_registration)
        scope = (id(fs), await mutation_queue_key(fs, p, path))
        current = state.tails.get(scope)
        released = loop.create_future()
        state.tails[scope] = released
    finally:
        registered.set_result(None)  # settled either way: the next registration may begin

    def release(_: object = None) -> None:
        released.set_result(None)
        if state.tails.get(scope) is released:
            del state.tails[scope]

    try:
        if current is not None:
            await asyncio.shield(current)
        return await fn()
    finally:
        if current is None or current.done():
            release()
        else:  # our own wait was interrupted: keep FIFO -- release only after the earlier entry
            current.add_done_callback(release)
