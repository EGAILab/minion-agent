"""Credential ownership / refresh mutation authority (`PROV-008`; Pi `resolveStoredOAuth`,
`auth/resolve.ts:122-179`).

This is the mechanism that makes `CredentialStore.modify()`'s own serialization guarantee
OBSERVABLE for the one operation that actually needs it: refreshing a rotating OAuth token
without ever letting two concurrent callers both read the same stale token and both commit their
own rotation. A bare `CredentialStore` proves nothing about double-refresh prevention by itself;
this double-checked-locking helper is what exercises it.

Ownership rule (design instruction, Pi `resolveProviderAuth`'s own comment,
`auth/resolve.ts:44-49`): a STORED credential owns the provider. Minion may refresh/mutate it
because the auth contract (this module) grants that authority explicitly, through the store's own
serialized `modify()` -- never by independently reading a token from some other source (a
filesystem, a CLI's own credential file) and rewriting it outside this seam. A credential source
this module does not own may be READ (via `CredentialStore.read`) but this module never assumes
mutation authority over it beyond what `modify()` itself grants for the store `refresh_if_expiring`
was given.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from .credential import AuthOperationOptions, Credential, OAuthCredential
from .signal import Abortable, CombinedSignal
from .store import CredentialStore

DEFAULT_MINIMUM_VALIDITY_MS = 5 * 60 * 1000.0
"""Pi's own default trigger window (`auth/resolve.ts::DEFAULT_OAUTH_MINIMUM_VALIDITY_MS`): a
credential within five minutes of expiring triggers a refresh. This is a TRIGGER threshold, not an
enforced contract -- see `minimum_validity_ms`'s own doc below for the difference."""

DEFAULT_REFRESH_TIMEOUT_SECONDS = 15.0
"""Pi's own fixed refresh budget (`auth/resolve.ts::DEFAULT_OAUTH_REFRESH_TIMEOUT_MS`, 15_000 ms):
the signal passed to `refresh` aborts on its own after this many seconds even if the caller never
cancels anything, so a hung refresh call cannot block a credential's own store lock forever."""


class AuthorityError(Exception):
    """Base for `refresh_if_expiring`'s own failures."""


class OAuthRefreshError(AuthorityError):
    """The injected `refresh` callable itself failed (Pi `ModelsError("oauth", ...)`) -- the
    provider rejected the refresh (e.g. `invalid_grant`), not a local storage problem."""


class CredentialStoreError(AuthorityError):
    """The credential store's own read/modify mechanism failed (Pi `ModelsError("auth", ...)`) --
    a local storage problem, not a rejection from the provider."""


def _expires_soon(credential: OAuthCredential, minimum_validity_ms: float, now_ms: float) -> bool:
    return now_ms + minimum_validity_ms >= credential.expires


async def refresh_if_expiring(
    store: CredentialStore,
    provider_id: str,
    refresh: Callable[[OAuthCredential, Abortable], Awaitable[OAuthCredential]],
    *,
    minimum_validity_ms: float | None = None,
    refresh_timeout_seconds: float = DEFAULT_REFRESH_TIMEOUT_SECONDS,
    now_ms: Callable[[], float] = lambda: time.time() * 1000,
    timeout_now: Callable[[], float] = time.monotonic,
    options: AuthOperationOptions | None = None,
) -> OAuthCredential | None:
    """Return the current OAuth credential for `provider_id`, refreshing it FIRST if it is within
    the trigger window of expiring -- double-checked under the store's own per-provider lock, so a
    second concurrent caller that also observed "expiring soon" cannot independently refresh again
    once the first caller's own refresh has already committed (Pi `resolveStoredOAuth`).

    `minimum_validity_ms` serves two roles Pi keeps distinct: it is ALWAYS combined with
    `DEFAULT_MINIMUM_VALIDITY_MS` (via `max`) to decide whether a refresh triggers at all, and that
    SAME effective (`max`'d) threshold is what the post-refresh check enforces too -- rejecting a
    refreshed credential that still does not meet it -- ONLY when the caller passed a value (`None`
    means "use the default trigger, do not enforce anything stronger afterward"). `L11-R011`
    (remediated): a prior revision post-validated against the RAW caller-supplied value instead of
    the effective one, so an explicit minimum SMALLER than the five-minute default (e.g. one
    minute) would incorrectly accept a refreshed credential still expiring within the default
    five-minute window -- Pi's own `resolveStoredOAuth` reuses the SAME `expiresSoon` closure
    (built from the effective threshold) for the initial trigger, the under-lock recheck, AND the
    post-refresh validation; there is only ever one threshold in Pi, not two.

    `refresh` receives the expiring credential AND a second, `Abortable` argument (`L11-R002`; Pi
    `OAuthAuth.refresh(credential, signal)`, `auth/types.ts:222`) -- a `CombinedSignal` (`signal.
    py`) that aborts when EITHER `options.signal` aborts OR `refresh_timeout_seconds` elapses
    (Pi's own `AbortSignal.any([signal, AbortSignal.timeout(15_000)])`, `resolve.ts:149-153`), so a
    hung refresh call cannot block this credential's own store lock forever even when the caller
    never cancels anything. This module does not itself abort `refresh`'s own execution -- exactly
    like Pi, it hands the callable a live signal to poll/honor cooperatively, the same contract
    every other `Abortable`-consuming seam in this package already follows. `timeout_now` is the
    injectable monotonic clock `CombinedSignal` measures its own budget against, defaulting to
    `time.monotonic` -- separate from `now_ms`, which is the wall-clock-epoch-milliseconds clock
    the EXPIRY trigger check above uses; the two are independent, unrelated clocks.

    Returns `None` when no OAuth credential is currently stored for `provider_id` (never stored,
    logged out, or replaced by a non-OAuth credential -- checked both before AND after refreshing,
    since either can become true concurrently while `refresh` itself is in flight). Raises
    `OAuthRefreshError` if `refresh` itself fails, or `CredentialStoreError` if the store's own
    read/modify mechanism fails -- the SAME error-code split Pi's own `ModelsError` makes, so a
    caller can distinguish "the provider rejected the refresh" from "the local credential store is
    broken."
    """
    try:
        stored = await store.read(provider_id, options)
    except Exception as error:
        raise CredentialStoreError(f"credential store read failed for {provider_id!r}") from error

    if not isinstance(stored, OAuthCredential):
        return None

    trigger_validity_ms = max(DEFAULT_MINIMUM_VALIDITY_MS, minimum_validity_ms or 0.0)
    if not _expires_soon(stored, trigger_validity_ms, now_ms()):
        return stored

    async def attempt_refresh(current: Credential | None) -> Credential | None:
        if not isinstance(current, OAuthCredential):
            return None  # Logged out (or replaced by a non-OAuth credential) meanwhile.
        if not _expires_soon(current, trigger_validity_ms, now_ms()):
            return None  # Another concurrent caller already refreshed it under this same lock.
        caller_signal = options.signal if options is not None else None
        combined_signal = CombinedSignal(caller_signal, refresh_timeout_seconds, now=timeout_now)
        try:
            return await refresh(current, combined_signal)
        except Exception as error:
            raise OAuthRefreshError(f"OAuth refresh failed for {provider_id!r}") from error

    try:
        result = await store.modify(provider_id, attempt_refresh, options)
    except OAuthRefreshError:
        raise
    except Exception as error:
        raise CredentialStoreError(f"credential store modify failed for {provider_id!r}") from error

    if not isinstance(result, OAuthCredential):
        return None  # Logged out meanwhile.

    if minimum_validity_ms is not None and _expires_soon(result, trigger_validity_ms, now_ms()):
        raise OAuthRefreshError(
            f"OAuth refresh returned a token that expires too soon for {provider_id!r}"
        )
    return result
