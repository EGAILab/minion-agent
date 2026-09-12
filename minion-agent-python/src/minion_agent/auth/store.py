"""The credential-store seam: one serialized read-modify-write path per provider id (`PROV-007`;
Pi `CredentialStore`, `auth/types.ts:65-94`; reference implementation `auth/credential-store.ts`).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Protocol

from .credential import AuthOperationOptions, Credential, CredentialInfo

_ABORT_POLL_INTERVAL_SECONDS = 0.05
"""Matches `device_code.py::abortable_sleep`'s own poll interval: `AuthOperationOptions.signal`
is Layer 09's certified, poll-only `RunSignal` (no push/event mechanism), so "prompt enough"
cancellation means polling at short, fixed steps -- not a redesign of Layer 09."""


class CredentialStoreOperationCancelled(Exception):
    """Raised when `options.signal` is observed aborted at one of the checkpoints Pi's own
    reference implementation checks: immediately for `read`/`list`; for `modify`/`delete`, once
    immediately if already aborted, again right after a queued call's own turn begins (i.e. once
    any earlier call for the SAME provider id has itself finished or failed), again for `modify`
    right after its own `fn` resolves but before its result would commit, and — distinctly, on top
    of all three -- at any point while the caller is still waiting, racing the in-flight operation
    itself (Pi `raceWithAbortSignal`).

    This last race is what makes cancellation OBSERVABLE PROMPTLY even while a `modify`/`delete`
    call's own callback is still running: the caller stops waiting and sees this exception before
    the underlying operation has necessarily finished. The operation itself is NOT stopped by this
    race alone -- it keeps running in the background -- but for `modify`, the SEPARATE post-`fn`
    checkpoint above still fires once that background `fn` finishes, so an `fn` that only finishes
    AFTER the signal has aborted has its own result DISCARDED, never committed, even though it ran
    to completion. Prompt rejection and eventual discard are two independent checkpoints that
    happen to compose this way -- neither one exists to substitute for the other, and Pi's own
    `credential-store.ts` performs both exactly the same way (`enqueue`/`raceWithAbortSignal`
    never cancels the queued task itself; `modify`'s own `options?.signal?.throwIfAborted()` after
    `await fn(current)` is what actually discards a too-late result)."""


class CredentialStore(Protocol):
    """App-owned credential storage, keyed by provider id, one credential per provider (Pi
    `CredentialStore`). `modify` is the ONLY write path: every mutation is a serialized
    read-modify-write, so a caller that needs to refresh a rotating token (`refresh_if_expiring`)
    can run its own check-and-refresh sequence entirely inside one `modify()` callback and be sure
    no second concurrent caller for the SAME provider id can read the same stale value and
    independently commit its own refresh too.

    Error semantics: `read` resolves `None` for a missing entry -- never raises for "not found."
    Methods raise only on genuine storage failure (or `CredentialStoreOperationCancelled` if an
    `AuthOperationOptions.signal` was aborted -- see that exception's own docstring for the exact
    checkpoints and the important caveat that a raced-away `modify`/`delete` call is NOT itself
    stopped, only the caller's own wait for it).

    `read(provider_id)` returns the stored credential, possibly expired -- display/status use.
    Resolved REQUEST auth (with refresh-if-needed applied) comes from `refresh_if_expiring`, not
    from `read()` directly. `list()` returns stored credential metadata for every provider id, in
    INSERTION order of each provider's own CURRENT entry (Pi `Map`/Python `dict` iteration order:
    a provider deleted and later given a fresh credential re-appears at the END of that order, not
    its original position) -- never resolving or exposing secrets.

    `modify(provider_id, fn)` is the only write path. `fn` sees the CURRENT credential (or `None`)
    and returns the new credential to store, or `None` to leave the entry UNCHANGED (not the same
    as deletion -- `delete()` is the separate, dedicated removal primitive). Serialized per
    `provider_id`: two concurrent `modify()` calls for the SAME id never run their own `fn`
    concurrently -- the second call's own `fn` sees whatever the first call's own `fn` committed
    (or, if the first call's own `fn` raised, the value from before that failed attempt; a raising
    `fn` leaves the stored credential untouched and its own exception propagates to the caller).
    Returns the post-write credential (the new one if `fn` returned one, otherwise whatever was
    already stored).

    `delete(provider_id)` removes a credential (logout), serialized against `modify()` for the
    SAME provider id through the same per-id ordering `modify()` itself uses.
    """

    async def read(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> Credential | None: ...

    async def list(
        self, options: AuthOperationOptions | None = None
    ) -> tuple[CredentialInfo, ...]: ...

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None: ...

    async def delete(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> None: ...


class InMemoryCredentialStore:
    """Default in-memory credential store (Pi `InMemoryCredentialStore`,
    `auth/credential-store.ts`).

    Proves IN-PROCESS serialization only. It does NOT claim cross-process locking, filesystem
    locking, or distributed locking -- the generic `CredentialStore` protocol permits a concrete
    backing store to provide those stronger guarantees, but does not require them, and this
    reference implementation makes no promise beyond one Python process's own event loop.

    `read`/`list` are NOT serialized against `modify`/`delete` -- matching Pi's own reference
    implementation exactly, they observe whatever is currently stored at the moment they run, so a
    `read()` racing a `modify()` may return the value from either side of the mutation.

    `modify`/`delete` for the SAME provider id share one FIFO chain (Pi's own per-provider promise
    chain, ported here as a chain of `asyncio.Task`s): the second queued call's own callback never
    runs until the first call's own mutation has fully committed, or failed without mutating
    anything. The chain entry for a provider id is pruned once its own tail finishes IF no later
    call has since replaced it (matching Pi's own `credential-store.ts` exactly -- an earlier
    revision of this module's own docstring incorrectly claimed the chain is "never pruned"; it
    genuinely is, both here and in Pi).

    Cancellation (`AuthOperationOptions.signal`) races the CALLER'S OWN WAIT against the queued
    operation, matching Pi's `raceWithAbortSignal` exactly: if the signal aborts while a
    `modify`/`delete` call is still queued behind an earlier one, OR while its own callback is
    still running, the caller sees `CredentialStoreOperationCancelled` promptly, without waiting
    for that operation to finish. The operation itself is NOT stopped by this race alone -- it
    keeps running in the background -- but `modify`'s own separate post-`fn` abort checkpoint
    still applies once that background `fn` eventually resolves, so a too-late `fn` result is
    DISCARDED, never committed (see `CredentialStoreOperationCancelled`'s own docstring for the
    full checkpoint list). A signal already aborted before a queued call's own turn begins prevents
    its `fn`/removal from ever running at all -- only an abort that arrives strictly AFTER that
    checkpoint can race an already-started callback this way.
    """

    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._chains: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def _check_not_aborted(options: AuthOperationOptions | None) -> None:
        if options is not None and options.signal is not None and options.signal.aborted:
            raise CredentialStoreOperationCancelled

    async def read(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> Credential | None:
        self._check_not_aborted(options)
        return self._credentials.get(provider_id)

    async def list(self, options: AuthOperationOptions | None = None) -> tuple[CredentialInfo, ...]:
        self._check_not_aborted(options)
        return tuple(
            CredentialInfo(provider_id=provider_id, type=credential.type)
            for provider_id, credential in self._credentials.items()
        )

    async def _enqueue[T](
        self,
        provider_id: str,
        task: Callable[[], Awaitable[T]],
        options: AuthOperationOptions | None,
    ) -> T:
        """Pi `enqueue` (`credential-store.ts:13-27`): wait for whatever chain entry currently
        occupies `provider_id` (swallowing its own failure -- one caller's error must never sour
        the chain for the next), THEN check `options.signal` -- this is the "queued pre-task"
        checkpoint, distinct from the immediate pre-call check every OTHER method here performs --
        and only then run `task`. The caller's own wait for the result is raced against the
        signal separately (`_race_with_abort`), so a cancellation observed WHILE `task` is already
        running does not stop `task` itself."""
        previous = self._chains.get(provider_id)

        async def queued() -> T:
            # `previous`, when present, is always an earlier call's own `tail` (below) -- which
            # already swallows its own exception before this chain entry is ever published to
            # `self._chains` -- so `previous` itself never raises here; no try/except needed.
            if previous is not None:
                await previous
            self._check_not_aborted(options)
            return await task()

        queued_task: asyncio.Task[T] = asyncio.ensure_future(queued())

        async def tail() -> None:
            with contextlib.suppress(Exception):
                await queued_task

        tail_task = asyncio.ensure_future(tail())
        self._chains[provider_id] = tail_task

        def _prune(_finished: asyncio.Task[None]) -> None:
            if self._chains.get(provider_id) is tail_task:
                del self._chains[provider_id]

        tail_task.add_done_callback(_prune)

        return await self._race_with_abort(queued_task, options)

    @staticmethod
    async def _race_with_abort[T](task: asyncio.Task[T], options: AuthOperationOptions | None) -> T:
        """Pi `raceWithAbortSignal`: stop WAITING as soon as the signal aborts, without cancelling
        `task` itself -- `task` keeps running (its own eventual exception, if any, is consumed by
        `_enqueue`'s own `tail`, so it never becomes an unretrieved-exception warning). `RunSignal`
        is poll-only (Layer 09, no push/event mechanism), so this polls at a short, fixed interval
        rather than redesigning Layer 09 with one."""
        signal = options.signal if options is not None else None
        if signal is None:
            return await task
        if signal.aborted:
            raise CredentialStoreOperationCancelled
        while True:
            done, _pending = await asyncio.wait({task}, timeout=_ABORT_POLL_INTERVAL_SECONDS)
            if task in done:
                return task.result()
            if signal.aborted:
                raise CredentialStoreOperationCancelled

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        async def task() -> Credential | None:
            current = self._credentials.get(provider_id)
            next_credential = await fn(current)
            self._check_not_aborted(options)
            if next_credential is not None:
                self._credentials[provider_id] = next_credential
            return next_credential if next_credential is not None else current

        return await self._enqueue(provider_id, task, options)

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        async def task() -> None:
            self._credentials.pop(provider_id, None)

        await self._enqueue(provider_id, task, options)
