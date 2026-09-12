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
from minion_agent.auth.store import InMemoryCredentialStore

FIVE_MINUTES_MS = 5 * 60 * 1000.0


async def _set(credential: Credential | None) -> Credential | None:
    return credential


async def test_no_stored_credential_returns_none_without_calling_refresh() -> None:
    store = InMemoryCredentialStore()
    calls = 0

    async def refresh(credential: OAuthCredential) -> OAuthCredential:
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

    async def refresh(credential: OAuthCredential) -> OAuthCredential:
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

    async def refresh(credential: OAuthCredential) -> OAuthCredential:
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

    async def refresh(credential: OAuthCredential) -> OAuthCredential:
        assert credential == stored
        return refreshed

    result = await refresh_if_expiring(store, "p", refresh, now_ms=lambda: 0.0)

    assert result == refreshed
    assert await store.read("p") == refreshed


async def test_refresh_failure_raises_oauth_refresh_error_and_leaves_credential_unchanged() -> None:
    store = InMemoryCredentialStore()
    stored = OAuthCredential(access="a1", refresh="r1", expires=100.0)
    await store.modify("p", lambda _c: _set(stored))

    async def refresh(_credential: OAuthCredential) -> OAuthCredential:
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

    async def refresh(credential: OAuthCredential) -> OAuthCredential:
        return credential

    with pytest.raises(CredentialStoreError, match="p"):
        await refresh_if_expiring(broken, "p", refresh, now_ms=lambda: 0.0)


async def test_a_store_read_failure_raises_credential_store_error() -> None:
    class BrokenStore(InMemoryCredentialStore):
        async def read(
            self, provider_id: str, options: AuthOperationOptions | None = None
        ) -> Credential | None:
            raise RuntimeError("disk on fire")

    async def refresh(credential: OAuthCredential) -> OAuthCredential:
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

    async def refresh(credential: OAuthCredential) -> OAuthCredential:
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

    async def refresh(credential: OAuthCredential) -> OAuthCredential:
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

    async def refresh(_credential: OAuthCredential) -> OAuthCredential:
        return OAuthCredential(access="a2", refresh="r2", expires=200.0)

    with pytest.raises(OAuthRefreshError, match="expires too soon"):
        await refresh_if_expiring(
            store, "p", refresh, minimum_validity_ms=FIVE_MINUTES_MS, now_ms=lambda: 0.0
        )
