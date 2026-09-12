"""The credential-store seam: one serialized read-modify-write path per provider id (`PROV-007`;
Pi `CredentialStore`, `auth/types.ts:65-94`; reference implementation `auth/credential-store.ts`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from .credential import AuthOperationOptions, Credential, CredentialInfo


class CredentialStoreOperationCancelled(Exception):
    """Raised when `options.signal` was aborted -- checked at the same two points Pi's own
    reference implementation checks `signal.throwIfAborted()`: before a queued mutation begins,
    and after its own callback resolves but before its result commits."""


class CredentialStore(Protocol):
    """App-owned credential storage, keyed by provider id, one credential per provider (Pi
    `CredentialStore`). `modify` is the ONLY write path: every mutation is a serialized
    read-modify-write, so a caller that needs to refresh a rotating token (`refresh_if_expiring`)
    can run its own check-and-refresh sequence entirely inside one `modify()` callback and be sure
    no second concurrent caller for the SAME provider id can read the same stale value and
    independently commit its own refresh too.

    Error semantics: `read` resolves `None` for a missing entry -- never raises for "not found."
    Methods raise only on genuine storage failure (or `CredentialStoreOperationCancelled` if an
    `AuthOperationOptions.signal` was aborted).

    `read(provider_id)` returns the stored credential, possibly expired -- display/status use.
    Resolved REQUEST auth (with refresh-if-needed applied) comes from `refresh_if_expiring`, not
    from `read()` directly. `list()` returns stored credential metadata for every provider id,
    without resolving or exposing secrets.

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
    `read()` racing a `modify()` may return the value from either side of the mutation. `modify`/
    `delete` for the SAME provider id share one lock (created lazily, one per id, never removed --
    a disclosed, immaterial difference from Pi's own self-pruning promise-chain map, since an idle
    `asyncio.Lock` costs nothing observable): the second queued call's own callback never runs
    until the first call's own mutation has fully committed, or failed without mutating anything.
    """

    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, provider_id: str) -> asyncio.Lock:
        lock = self._locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[provider_id] = lock
        return lock

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

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        self._check_not_aborted(options)
        async with self._lock_for(provider_id):
            current = self._credentials.get(provider_id)
            next_credential = await fn(current)
            self._check_not_aborted(options)
            if next_credential is not None:
                self._credentials[provider_id] = next_credential
            return next_credential if next_credential is not None else current

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        self._check_not_aborted(options)
        async with self._lock_for(provider_id):
            self._credentials.pop(provider_id, None)
