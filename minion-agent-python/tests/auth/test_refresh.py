"""`refresh_if_expiring`'s own double-checked-locking ownership/authority behavior (`PROV-008`)."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from minion_agent.auth.credential import (
    ApiKeyCredential,
    AuthOperationOptions,
    Credential,
    OAuthCredential,
)
from minion_agent.auth.refresh import (
    CredentialStoreError,
    OAuthRefreshError,
    refresh_if_expiring,
)
from minion_agent.auth.signal import Abortable
from minion_agent.auth.store import InMemoryCredentialStore
from minion_agent.runtime.signal import RunAbortController

FIVE_MINUTES_MS = 5 * 60 * 1000.0


async def _set(credential: Credential | None) -> Credential | None:
    return credential


async def test_no_stored_credential_returns_none_without_calling_refresh() -> None:
    store = InMemoryCredentialStore()
    calls = 0

    async def refresh(credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        nonlocal calls
        calls += 1
        return credential

    result = await refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0)

    assert result is None
    assert calls == 0


async def test_a_non_oauth_stored_credential_returns_none_without_calling_refresh() -> None:
    store = InMemoryCredentialStore()
    await store.modify("p", lambda _c: _set(ApiKeyCredential(key="k")))
    calls = 0

    async def refresh(credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        nonlocal calls
        calls += 1
        return credential

    result = await refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0)

    assert result is None
    assert calls == 0


async def test_a_credential_with_ample_validity_is_returned_unrefreshed() -> None:
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=1_000_000.0)
    await store.modify("p", lambda _c: _set(stored))
    calls = 0

    async def refresh(credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        nonlocal calls
        calls += 1
        return credential

    result = await refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0)

    assert result == stored
    assert calls == 0


async def test_a_credential_expiring_soon_is_refreshed_and_persisted() -> None:
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)
    await store.modify("p", lambda _c: _set(stored))
    refreshed = OAuthCredential(access="a2", refresh="r2", expires=1_000_000.0)

    async def refresh(credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        assert credential == stored
        return refreshed

    result = await refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0)

    assert result == refreshed
    assert await store.read("p") == refreshed


async def test_refresh_receives_a_combined_signal_reflecting_caller_abort() -> None:
    """`L11-R002`: Pi's own `OAuthAuth.refresh(credential, signal)` hands the refresh call a live
    signal composing the caller's own cancellation with a fixed timeout budget
    (`AbortSignal.any([signal, AbortSignal.timeout(15_000)])`) -- `refresh` here must receive a
    second, `Abortable` argument whose `.aborted` reflects the CALLER's own signal, not a bare
    credential-only call. The SAME signal also guards the outer `store.modify()` call (matching
    Pi's own `resolveStoredOAuth`, which passes the identical `signal` both ways), so aborting it
    mid-refresh correctly discards the eventual result too (`CredentialStoreError`) -- this test's
    own concern is only that `refresh` was handed a signal that actually reflects the abort."""
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)
    await store.modify("p", lambda _c: _set(stored))
    controller = RunAbortController()
    observed: list[bool] = []

    async def refresh(credential: OAuthCredential, signal: Abortable) -> OAuthCredential:
        observed.append(signal.aborted)
        controller.abort()
        observed.append(signal.aborted)
        return OAuthCredential(access="a2", refresh="r2", expires=1_000_000.0)

    with pytest.raises(CredentialStoreError):
        await refresh_if_expiring(
            store,
            "p",
            refresh,
            now_ms=lambda: 0.0,
            options=AuthOperationOptions(signal=controller.signal),
        )

    assert observed == [False, True]  # unaborted before, aborted once the caller signal fires


async def test_refresh_signal_aborts_on_its_own_after_the_timeout_budget_elapses() -> None:
    """`L11-R002`: even with no caller signal at all, `refresh`'s own combined signal aborts once
    `refresh_timeout_seconds` elapses (Pi's own fixed 15-second refresh budget), proven here with
    an injected, deterministic `timeout_now` clock rather than real waiting."""
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)
    await store.modify("p", lambda _c: _set(stored))
    times = iter([0.0, 20.0])  # deadline set at t=0 (budget 15s); refresh's own check at t=20

    async def refresh(credential: OAuthCredential, signal: Abortable) -> OAuthCredential:
        assert signal.aborted is True
        return OAuthCredential(access="a2", refresh="r2", expires=1_000_000.0)

    result = await refresh_if_expiring(
        store,
        "p",
        refresh,
        refresh_timeout_seconds=15.0,
        now_ms=lambda: 0.0,
        timeout_now=lambda: next(times),
    )

    assert result == OAuthCredential(access="a2", refresh="r2", expires=1_000_000.0)


async def test_refresh_failure_raises_oauth_refresh_error_and_leaves_credential_unchanged() -> None:
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)
    await store.modify("p", lambda _c: _set(stored))

    async def refresh(_credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        raise RuntimeError("invalid_grant")

    with pytest.raises(OAuthRefreshError, match="p"):
        await refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0)

    assert await store.read("p") == stored


async def test_a_store_modify_failure_unrelated_to_refresh_raises_credential_store_error() -> None:
    """`store.modify()` itself can fail for reasons that have nothing to do with `attempt_refresh`
    (e.g. a broken lock/storage layer) -- distinct from `OAuthRefreshError`, which is raised only
    when the INJECTED `refresh` callable itself fails."""
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)

    class BrokenModifyStore(InMemoryCredentialStore):
        async def modify(
            self,
            provider_id: str,
            fn: Callable[[Credential | None], Awaitable[Credential | None]],
            options: AuthOperationOptions | None = None,
        ) -> Credential | None:
            raise RuntimeError("lock layer exploded")

    broken = BrokenModifyStore()
    broken._credentials["p"] = stored  # seed state directly; modify() itself always raises

    async def refresh(credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        return credential

    with pytest.raises(CredentialStoreError, match="p"):
        await refresh_if_expiring(broken, "p", refresh, now_ms=lambda: 0.0)


async def test_a_store_read_failure_raises_credential_store_error() -> None:
    class BrokenStore(InMemoryCredentialStore):
        async def read(
            self, provider_id: str, options: AuthOperationOptions | None = None
        ) -> Credential | None:
            raise RuntimeError("disk on fire")

    async def refresh(credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        return credential

    with pytest.raises(CredentialStoreError, match="p"):
        await refresh_if_expiring(BrokenStore(), "p", refresh, now_ms=lambda: 0.0)


async def test_two_concurrent_expiring_callers_refresh_exactly_once() -> None:
    """The double-checked-locking guarantee this whole module exists for: two callers that BOTH
    observe "expiring soon" via the optimistic (unlocked) read must not both independently commit
    a refresh -- only the first to acquire the store's own per-provider lock actually calls
    `refresh`; the second sees the already-refreshed, no-longer-expiring credential under the same
    lock and returns it without calling `refresh` again."""
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)
    await store.modify("p", lambda _c: _set(stored))

    refresh_calls = 0
    first_refresh_entered = asyncio.Event()
    release_first_refresh = asyncio.Event()

    async def refresh(credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        nonlocal refresh_calls
        refresh_calls += 1
        first_refresh_entered.set()
        await release_first_refresh.wait()
        return OAuthCredential(access="a2", refresh="r2", expires=1_000_000.0)

    first_task = asyncio.create_task(refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0))
    await first_refresh_entered.wait()

    second_task = asyncio.create_task(refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0))
    await asyncio.sleep(0)  # let the second call queue behind the first's own in-flight modify()
    release_first_refresh.set()

    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert refresh_calls == 1
    assert first_result == OAuthCredential(access="a2", refresh="r2", expires=1_000_000.0)
    assert second_result == first_result


async def test_logged_out_meanwhile_returns_none() -> None:
    """The optimistic read sees an expiring OAuth credential, but by the time the lock is
    acquired the credential has been deleted -- the double-checked re-read inside `modify()` must
    observe that and return `None`, not attempt to refresh a credential that no longer exists."""
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)
    await store.modify("p", lambda _c: _set(stored))

    entered_modify = asyncio.Event()
    release_modify = asyncio.Event()
    real_modify = store.modify

    async def delayed_modify(
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        entered_modify.set()
        await release_modify.wait()
        return await real_modify(provider_id, fn, options)

    store.modify = delayed_modify  # type: ignore[method-assign]

    async def refresh(credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        raise AssertionError("refresh must not be called once logged out meanwhile")

    task = asyncio.create_task(refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0))
    await entered_modify.wait()
    await store.delete("p")  # the dedicated removal primitive -- logs the provider out
    release_modify.set()

    assert await task is None


async def test_explicit_minimum_validity_override_rejects_a_too_short_refresh() -> None:
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)
    await store.modify("p", lambda _c: _set(stored))

    async def refresh(_credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        return OAuthCredential(access="a2", refresh="r2", expires=200.0)

    with pytest.raises(OAuthRefreshError, match="expires too soon"):
        await refresh_if_expiring(
            store, "p", refresh, minimum_validity_ms=FIVE_MINUTES_MS, now_ms=lambda: 0.0
        )


async def test_explicit_minimum_smaller_than_default_still_uses_the_effective_threshold() -> None:
    """`L11-R011`: an explicit minimum SMALLER than the five-minute default must still be
    post-validated against the EFFECTIVE (`max`'d) threshold, not the raw caller value -- Pi's own
    `resolveStoredOAuth` reuses the SAME `expiresSoon` closure (built from the effective threshold)
    for the initial trigger, the under-lock recheck, AND the post-refresh validation; there is only
    ever one threshold in Pi, not two. A refreshed credential expiring at 120_000ms, with now=0 and
    an explicit minimum of 60_000ms, is still inside the EFFECTIVE 300_000ms window and must be
    rejected -- the pre-existing `..._rejects_a_too_short_refresh` test above uses an explicit
    minimum EQUAL to the default and cannot discriminate raw-vs-effective at all."""
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=0.0)
    await store.modify("p", lambda _c: _set(stored))

    async def refresh(_credential: OAuthCredential, _signal: Abortable) -> OAuthCredential:
        return OAuthCredential(access="a2", refresh="r2", expires=120_000.0)

    with pytest.raises(OAuthRefreshError, match="expires too soon"):
        await refresh_if_expiring(
            store, "p", refresh, minimum_validity_ms=60_000.0, now_ms=lambda: 0.0
        )
