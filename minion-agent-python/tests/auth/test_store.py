"""`CredentialStore`/`InMemoryCredentialStore` concurrency behavior matrix (`PROV-007`).

Six discriminating cases, matching Pi's own reference implementation exactly
(`auth/credential-store.ts`): each case name below is the exact case this test proves.
"""

import asyncio

import pytest

from minion_agent.auth.credential import ApiKeyCredential, AuthOperationOptions, Credential
from minion_agent.auth.store import CredentialStoreOperationCancelled, InMemoryCredentialStore
from minion_agent.runtime.signal import RunAbortController


async def _to(credential: Credential) -> Credential:
    return credential


async def test_case1_single_modify_transitions_from_initial_to_new_state() -> None:
    store = InMemoryCredentialStore()

    async def to_a(_current: Credential | None) -> Credential | None:
        return ApiKeyCredential(key="A")

    result = await store.modify("p", to_a)

    assert result == ApiKeyCredential(key="A")
    assert await store.read("p") == ApiKeyCredential(key="A")


async def test_case2_modify_callback_failure_leaves_stored_credential_unchanged() -> None:
    store = InMemoryCredentialStore()

    async def to_a(_current: Credential | None) -> Credential | None:
        return ApiKeyCredential(key="A")

    await store.modify("p", to_a)

    async def failing(_current: Credential | None) -> Credential | None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await store.modify("p", failing)

    assert await store.read("p") == ApiKeyCredential(key="A")


async def test_case3_two_concurrent_modify_calls_serialize_and_see_committed_state() -> None:
    store = InMemoryCredentialStore()
    order: list[str] = []
    first_started = asyncio.Event()

    async def first_fn(_current: Credential | None) -> Credential | None:
        order.append("first-start")
        first_started.set()
        await asyncio.sleep(0)  # yield so the second call can be queued behind this one
        order.append("first-end")
        return ApiKeyCredential(key="B")

    async def second_fn(current: Credential | None) -> Credential | None:
        key = current.key if isinstance(current, ApiKeyCredential) else None
        order.append(f"second-sees-{key}")
        return ApiKeyCredential(key="C")

    first_task = asyncio.create_task(store.modify("p", first_fn))
    await first_started.wait()
    second_task = asyncio.create_task(store.modify("p", second_fn))
    await asyncio.gather(first_task, second_task)

    assert order == ["first-start", "first-end", "second-sees-B"]
    assert await store.read("p") == ApiKeyCredential(key="C")


async def test_case4_read_is_not_serialized_against_an_in_flight_modify() -> None:
    store = InMemoryCredentialStore()
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def slow_fn(_current: Credential | None) -> Credential | None:
        entered.set()
        await proceed.wait()
        return ApiKeyCredential(key="B")

    modify_task = asyncio.create_task(store.modify("p", slow_fn))
    await entered.wait()

    observed = await asyncio.wait_for(store.read("p"), timeout=1.0)
    assert observed is None  # the in-flight modify has not committed yet, and read() is not
    # queued behind it -- it observes whatever is currently stored, matching Pi exactly.

    proceed.set()
    await modify_task
    assert await store.read("p") == ApiKeyCredential(key="B")


async def test_case5_modify_on_an_absent_credential_sees_none() -> None:
    store = InMemoryCredentialStore()
    seen: list[Credential | None] = []

    async def observe(current: Credential | None) -> Credential | None:
        seen.append(current)
        return None

    result = await store.modify("p", observe)

    assert seen == [None]
    assert result is None
    assert await store.read("p") is None


async def test_case6_modify_returning_none_leaves_the_entry_unchanged_not_removed() -> None:
    store = InMemoryCredentialStore()

    async def to_a(_current: Credential | None) -> Credential | None:
        return ApiKeyCredential(key="A")

    await store.modify("p", to_a)

    async def no_change(_current: Credential | None) -> Credential | None:
        return None

    result = await store.modify("p", no_change)

    assert result == ApiKeyCredential(key="A")  # the UNCHANGED current, not None
    assert await store.read("p") == ApiKeyCredential(key="A")


async def test_delete_is_the_dedicated_removal_primitive_not_modify() -> None:
    store = InMemoryCredentialStore()

    async def to_a(_current: Credential | None) -> Credential | None:
        return ApiKeyCredential(key="A")

    await store.modify("p", to_a)
    await store.delete("p")

    assert await store.read("p") is None


async def test_delete_serializes_against_modify_for_the_same_provider_id() -> None:
    store = InMemoryCredentialStore()
    order: list[str] = []
    modify_started = asyncio.Event()
    release_modify = asyncio.Event()

    async def slow_fn(_current: Credential | None) -> Credential | None:
        order.append("modify-start")
        modify_started.set()
        await release_modify.wait()
        order.append("modify-end")
        return ApiKeyCredential(key="A")

    modify_task = asyncio.create_task(store.modify("p", slow_fn))
    await modify_started.wait()
    delete_task = asyncio.create_task(store.delete("p"))
    await asyncio.sleep(0)  # let delete_task queue behind the in-flight modify
    release_modify.set()
    await asyncio.gather(modify_task, delete_task)

    order.append("delete-ran")
    assert order == ["modify-start", "modify-end", "delete-ran"]
    assert await store.read("p") is None  # delete ran after modify committed A, then removed it


async def test_different_provider_ids_do_not_serialize_against_each_other() -> None:
    store = InMemoryCredentialStore()
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def slow_fn(_current: Credential | None) -> Credential | None:
        entered.set()
        await proceed.wait()
        return ApiKeyCredential(key="slow")

    slow_task = asyncio.create_task(store.modify("provider-a", slow_fn))
    await entered.wait()

    async def fast_fn(_current: Credential | None) -> Credential | None:
        return ApiKeyCredential(key="fast")

    fast_result = await asyncio.wait_for(store.modify("provider-b", fast_fn), timeout=1.0)
    assert fast_result == ApiKeyCredential(key="fast")

    proceed.set()
    await slow_task


async def test_list_reports_every_provider_without_secrets() -> None:
    store = InMemoryCredentialStore()
    await store.modify("a", lambda _c: _to(ApiKeyCredential(key="x")))
    await store.modify("b", lambda _c: _to(ApiKeyCredential(key="y")))

    infos = await store.list()

    assert {info.provider_id for info in infos} == {"a", "b"}
    assert all(info.type == "api_key" for info in infos)


async def test_modify_raises_immediately_if_the_signal_is_already_aborted() -> None:
    controller = RunAbortController()
    controller.abort()
    store = InMemoryCredentialStore()

    options = AuthOperationOptions(signal=controller.signal)
    with pytest.raises(CredentialStoreOperationCancelled):
        await store.modify("p", lambda _c: _to(ApiKeyCredential(key="A")), options)

    assert await store.read("p") is None


async def test_modify_discards_its_result_if_the_signal_aborts_during_fn() -> None:
    controller = RunAbortController()
    store = InMemoryCredentialStore()

    async def fn(_current: Credential | None) -> Credential | None:
        controller.abort()
        return ApiKeyCredential(key="X")

    with pytest.raises(CredentialStoreOperationCancelled):
        await store.modify("p", fn, AuthOperationOptions(signal=controller.signal))

    assert await store.read("p") is None  # discarded, never committed


async def test_read_and_list_raise_immediately_if_the_signal_is_already_aborted() -> None:
    controller = RunAbortController()
    controller.abort()
    store = InMemoryCredentialStore()

    with pytest.raises(CredentialStoreOperationCancelled):
        await store.read("p", AuthOperationOptions(signal=controller.signal))
    with pytest.raises(CredentialStoreOperationCancelled):
        await store.list(AuthOperationOptions(signal=controller.signal))
