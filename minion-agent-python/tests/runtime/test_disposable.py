"""DisposableList unwinds in reverse and is idempotent."""

import pytest

from minion_agent.runtime.disposable import DisposableList


async def test_disposes_in_reverse_order() -> None:
    order: list[str] = []
    disposables = DisposableList()
    for label in ("first", "second", "third"):
        disposables.push(lambda label=label: order.append(label))

    await disposables.dispose_all()

    assert order == ["third", "second", "first"]


async def test_dispose_all_is_idempotent() -> None:
    order: list[str] = []
    disposables = DisposableList()
    disposables.push(lambda: order.append("once"))

    await disposables.dispose_all()
    await disposables.dispose_all()

    assert order == ["once"]


async def test_awaits_async_disposers() -> None:
    order: list[str] = []
    disposables = DisposableList()

    async def async_disposer() -> None:
        order.append("async")

    disposables.push(async_disposer)
    disposables.push(lambda: order.append("sync"))

    await disposables.dispose_all()

    assert order == ["sync", "async"]


async def test_remove_handle_prevents_later_disposal() -> None:
    order: list[str] = []
    disposables = DisposableList()
    disposables.push(lambda: order.append("kept"))
    remove = disposables.push(lambda: order.append("removed"))

    remove()
    await disposables.dispose_all()

    assert order == ["kept"]


async def test_all_disposers_run_even_when_one_raises() -> None:
    order: list[str] = []
    disposables = DisposableList()
    disposables.push(lambda: order.append("first"))
    disposables.push(lambda: (_ for _ in ()).throw(ValueError("boom")))
    disposables.push(lambda: order.append("third"))

    with pytest.raises(ExceptionGroup) as excinfo:
        await disposables.dispose_all()

    assert order == ["third", "first"]
    assert len(excinfo.value.exceptions) == 1


async def test_len_counts_live_disposers() -> None:
    disposables = DisposableList()
    disposables.push(lambda: None)
    remove = disposables.push(lambda: None)

    assert len(disposables) == 2

    remove()

    assert len(disposables) == 1


async def test_disposed_reports_state() -> None:
    disposables = DisposableList()

    assert not disposables.disposed

    await disposables.dispose_all()

    assert disposables.disposed
