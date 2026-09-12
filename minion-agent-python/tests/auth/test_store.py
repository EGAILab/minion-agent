"""`CredentialStore`/`InMemoryCredentialStore` concurrency behavior matrix (`PROV-007`).

Six discriminating cases, matching Pi's own reference implementation exactly
(`auth/credential-store.ts`): each case name below is the exact case this test proves.
"""

import asyncio

import pytest

from minion_agent.auth.credential import (
    ApiKeyCredential,
    AuthOperationOptions,
    Credential,
    OAuthCredential,
)
from minion_agent.auth.store import CredentialStoreOperationCancelled, InMemoryCredentialStore
from minion_agent.runtime.signal import RunAbortController


async def _to(credential: Credential) -> Credential:
    return credential


async def test_w_r006_env_new_key_survives_a_store_round_trip() -> None:
    """`L11-R006` (owner-decided Pi-parity): the store itself performs no defensive copy either --
    assigning a brand-new top-level key on `env` through a credential returned by `modify()` is
    observed by a later `read()` for the same provider id, exactly matching Pi's own
    `InMemoryCredentialStore`, which holds direct references into its own backing `Map`. `env`'s
    own domain stays flat (`L11-R010`); the nested-value equivalent lives on `extra` below."""
    store = InMemoryCredentialStore()
    stored = ApiKeyCredential(key="sk-test", env={})

    committed = await store.modify("p", lambda _c: _to(stored))
    assert committed is not None and committed.env is not None
    committed.env["NEW"] = "v"

    reread = await store.read("p")
    assert reread is not None and reread.env is not None
    assert reread.env == {"NEW": "v"}


async def test_w_r006_extra_nested_mutation_survives_a_store_round_trip() -> None:
    """`L11-R006`: same round-trip guarantee as above, exercised on `OAuthCredential.extra`'s own
    recursive JSON domain -- a NESTED value mutation persists through the store."""
    store = InMemoryCredentialStore()
    extra = {"nested": {"value": "A"}}
    stored = OAuthCredential(access="a", refresh="r", expires=1234.0, extra=extra)

    committed = await store.modify("p", lambda _c: _to(stored))
    assert isinstance(committed, OAuthCredential)
    committed.extra["nested"]["value"] = "B"  # type: ignore[index]

    reread = await store.read("p")
    assert isinstance(reread, OAuthCredential)
    assert reread.extra["nested"] == {"value": "B"}


async def test_w_r009_scalar_field_mutation_survives_a_store_round_trip() -> None:
    """`L11-R009`: Pi's own returned live credential permits mutating a scalar field directly, and
    a later `read()` observes it -- the store performs no defensive copy of the credential object
    itself, so this is a genuine end-to-end proof, not merely a bare-dataclass-level one."""
    store = InMemoryCredentialStore()
    stored = ApiKeyCredential(key="A")

    committed = await store.modify("p", lambda _c: _to(stored))
    assert isinstance(committed, ApiKeyCredential)
    committed.key = "B"

    reread = await store.read("p")
    assert isinstance(reread, ApiKeyCredential)
    assert reread.key == "B"


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


async def test_a_queued_modify_still_runs_after_an_earlier_queued_modify_raised() -> None:
    """One caller's own `fn` failing must never sour the chain for the NEXT queued caller (Pi
    `credential-store.ts`'s own `await previous.catch(() => {})` intent) -- a queued second
    `modify` call must still run once the first, failing one settles."""
    store = InMemoryCredentialStore()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def failing_fn(_current: Credential | None) -> Credential | None:
        entered.set()
        await release.wait()
        raise RuntimeError("boom")

    first_task = asyncio.create_task(store.modify("p", failing_fn))
    await entered.wait()

    async def second_fn(_current: Credential | None) -> Credential | None:
        return ApiKeyCredential(key="B")

    second_task = asyncio.create_task(store.modify("p", second_fn))
    await asyncio.sleep(0)  # let second_task queue behind the still-in-flight first_task
    release.set()

    with pytest.raises(RuntimeError, match="boom"):
        await first_task

    assert await second_task == ApiKeyCredential(key="B")
    assert await store.read("p") == ApiKeyCredential(key="B")


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


async def test_list_reports_every_provider_without_secrets_in_insertion_order() -> None:
    """`L11-R007`: Pi's own reference store iterates its insertion-ordered `Map`
    (`[...this.credentials]`); this pins the exact SAME observable order, not merely the set of
    provider ids -- a prior revision's own test weakened this to a set comparison, which would not
    have caught an order regression at all."""
    store = InMemoryCredentialStore()
    await store.modify("b", lambda _c: _to(ApiKeyCredential(key="y")))
    await store.modify("a", lambda _c: _to(ApiKeyCredential(key="x")))

    infos = await store.list()

    assert [info.provider_id for info in infos] == ["b", "a"]
    assert all(info.type == "api_key" for info in infos)


async def test_list_order_follows_the_current_entry_not_the_original_insertion() -> None:
    """A provider deleted and later given a fresh credential re-appears at the END of `list()`'s
    own order, not its original position -- matching Pi's own `Map`/Python's own `dict` semantics
    for a delete-then-set-again sequence."""
    store = InMemoryCredentialStore()
    await store.modify("a", lambda _c: _to(ApiKeyCredential(key="1")))
    await store.modify("b", lambda _c: _to(ApiKeyCredential(key="2")))
    await store.delete("a")
    await store.modify("a", lambda _c: _to(ApiKeyCredential(key="3")))

    infos = await store.list()

    assert [info.provider_id for info in infos] == ["b", "a"]


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


async def test_modify_queued_behind_another_never_runs_its_fn_once_aborted_while_still_queued() -> (
    None
):
    """`L11-R001`, witness 1 (Pi `enqueue`'s own queued pre-task checkpoint,
    `credential-store.ts:13-27`): a queued `modify` call's own `fn` must never run at all if the
    signal aborts while it is still waiting for an EARLIER `modify` call (for the same provider
    id) to finish -- not merely reported as cancelled after already having been invoked."""
    store = InMemoryCredentialStore()
    controller = RunAbortController()
    b_called = False
    a_entered = asyncio.Event()
    hold_a = asyncio.Event()

    async def fn_a(_current: Credential | None) -> Credential | None:
        a_entered.set()
        await hold_a.wait()
        return ApiKeyCredential(key="A")

    async def fn_b(_current: Credential | None) -> Credential | None:
        nonlocal b_called
        b_called = True
        return ApiKeyCredential(key="B")

    a_task = asyncio.create_task(store.modify("p", fn_a))
    await a_entered.wait()

    options = AuthOperationOptions(signal=controller.signal)
    b_task = asyncio.create_task(store.modify("p", fn_b, options))
    await asyncio.sleep(0)  # let b_task's own _enqueue queue it behind a_task before aborting
    controller.abort()

    with pytest.raises(CredentialStoreOperationCancelled):
        await asyncio.wait_for(b_task, timeout=1.0)

    hold_a.set()
    await a_task

    assert b_called is False
    assert await store.read("p") == ApiKeyCredential(key="A")


async def test_delete_queued_behind_a_modify_never_runs_once_aborted_while_still_queued() -> None:
    """`L11-R001`, witness 2: the same queued pre-task checkpoint, exercised via `delete()`."""
    store = InMemoryCredentialStore()
    controller = RunAbortController()
    modify_entered = asyncio.Event()
    hold_modify = asyncio.Event()

    async def slow_fn(_current: Credential | None) -> Credential | None:
        modify_entered.set()
        await hold_modify.wait()
        return ApiKeyCredential(key="B")

    modify_task = asyncio.create_task(store.modify("p", slow_fn))
    await modify_entered.wait()

    options = AuthOperationOptions(signal=controller.signal)
    delete_task = asyncio.create_task(store.delete("p", options))
    await asyncio.sleep(0)  # let delete_task queue behind modify_task before aborting
    controller.abort()

    with pytest.raises(CredentialStoreOperationCancelled):
        await asyncio.wait_for(delete_task, timeout=1.0)

    hold_modify.set()
    await modify_task

    assert await store.read("p") == ApiKeyCredential(key="B")  # delete never ran


async def test_modify_wait_rejects_promptly_when_aborted_mid_callback_and_result_is_discarded() -> (
    None
):
    """`L11-R001`, witness 3 (Pi `raceWithAbortSignal` racing the caller's own wait, COMBINED with
    the existing post-`fn` `throwIfAborted()` checkpoint): the CALLER's own wait stops as soon as
    the signal aborts, even while `fn` is already running -- `fn` itself is NOT cancelled and runs
    to completion in the background, but once it finishes, the SAME post-`fn` abort check every
    `modify()` call performs (`test_modify_discards_its_result_if_the_signal_aborts_during_fn`,
    above) fires again and discards its result -- it is never committed. Prompt rejection and
    eventual discard are two DIFFERENT checkpoints that happen to compose correctly here, not one
    mechanism standing in for the other."""
    store = InMemoryCredentialStore()
    controller = RunAbortController()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fn(_current: Credential | None) -> Credential | None:
        entered.set()
        await release.wait()
        return ApiKeyCredential(key="A")

    options = AuthOperationOptions(signal=controller.signal)
    modify_task = asyncio.create_task(store.modify("p", fn, options))
    await entered.wait()
    controller.abort()

    with pytest.raises(CredentialStoreOperationCancelled):
        await asyncio.wait_for(modify_task, timeout=1.0)

    assert await store.read("p") is None  # rejected promptly, well before fn has even returned

    release.set()
    for _ in range(200):
        await asyncio.sleep(0)  # let the background fn/task run to completion

    assert await store.read("p") is None  # fn's own result was discarded, never committed
