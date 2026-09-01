"""Parallel fans out and aggregates errors; serial runs in order and returns."""

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from minion_agent.runtime.errors import EventModeError
from minion_agent.runtime.events import DispatchMode, EventBus


async def test_parallel_runs_listeners_concurrently() -> None:
    bus = EventBus()
    bus.declare("test/parallel", DispatchMode.PARALLEL)
    started = asyncio.Event()
    finished: list[str] = []

    async def slow() -> None:
        await started.wait()
        finished.append("slow")

    async def fast() -> None:
        started.set()
        finished.append("fast")

    bus.on("test/parallel", slow)
    bus.on("test/parallel", fast)

    await bus.parallel("test/parallel")

    assert finished == ["fast", "slow"]


async def test_parallel_aggregates_listener_errors() -> None:
    bus = EventBus()
    bus.declare("test/parallel", DispatchMode.PARALLEL)

    async def boom_one() -> None:
        raise ValueError("one")

    async def boom_two() -> None:
        raise ValueError("two")

    bus.on("test/parallel", boom_one)
    bus.on("test/parallel", boom_two)

    with pytest.raises(ExceptionGroup) as excinfo:
        await bus.parallel("test/parallel")

    assert len(excinfo.value.exceptions) == 2


async def test_parallel_without_listeners_is_a_noop() -> None:
    bus = EventBus()
    bus.declare("test/parallel", DispatchMode.PARALLEL)

    await bus.parallel("test/parallel")


async def test_serial_runs_in_registration_order_and_returns_last_value() -> None:
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)
    seen: list[str] = []

    async def first() -> str:
        seen.append("first")
        return "first-result"

    async def second() -> str:
        seen.append("second")
        return "second-result"

    bus.on("test/serial", first)
    bus.on("test/serial", second)

    result = await bus.serial("test/serial")

    assert seen == ["first", "second"]
    assert result == "second-result"


async def test_serial_returns_none_without_listeners() -> None:
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)

    assert await bus.serial("test/serial") is None


async def test_serial_accepts_sync_listeners() -> None:
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)
    bus.on("test/serial", lambda: "sync-result")

    assert await bus.serial("test/serial") == "sync-result"


async def test_parallel_rejects_wrong_mode() -> None:
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)

    with pytest.raises(EventModeError):
        await bus.parallel("test/serial")


async def test_serial_rejects_wrong_mode() -> None:
    bus = EventBus()
    bus.declare("test/parallel", DispatchMode.PARALLEL)

    with pytest.raises(EventModeError):
        await bus.serial("test/parallel")


def _eager_probe(coro: Coroutine[Any, Any, object]) -> asyncio.Task[object]:
    """Start `coro` synchronously, matching pinned Pi's own JS `await` semantics -- calling an
    `async function` runs its body immediately up to its own first genuine suspension, before
    returning control to the caller. `asyncio.eager_task_factory` is Python's own structural
    analogue (Layer 08, PASS 9, `L08-R002`)."""
    return asyncio.eager_task_factory(asyncio.get_running_loop(), coro)


async def test_serial_by_default_does_not_yield_between_synchronous_listeners() -> None:
    """`L08-R002`, PASS 9: `yield_after_each` defaults to `False` -- every existing `serial()`
    caller keeps this module's own certified behavior exactly. Two synchronous listeners run
    back-to-back, with no scheduler turn in between, so a caller driven eagerly enough to observe
    state right after dispatch starts sees BOTH listeners' own effects already applied -- the
    narrow lower-layer delta audit's own acceptance evidence that the new parameter changes
    nothing for a caller that omits it."""
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)
    order: list[str] = []
    bus.on("test/serial", lambda: order.append("listener-1"))
    bus.on("test/serial", lambda: order.append("listener-2"))

    async def dispatch() -> None:
        await bus.serial("test/serial")

    task = _eager_probe(dispatch())
    order.append("caller-continued")
    await task

    assert order == ["listener-1", "listener-2", "caller-continued"]


async def test_serial_yield_after_each_suspends_between_every_listener() -> None:
    """`L08-R002`, PASS 9: `yield_after_each=True` reproduces pinned Pi's own `for (const listener
    of listeners) { await listener(event, signal); }` (`agent.ts:544-591`) exactly -- a JS `await`
    always defers its continuation by at least one microtask turn, even for a fully synchronous
    listener, so pinned Pi's own dispatch suspends between EVERY listener, not only when one
    genuinely performs async work. An eagerly-driven caller therefore observes only listener 1's
    own effect before its own next statement runs, exactly matching the independent Rust
    re-review's own focused two-listener probe (`listener-1, tool-continued, listener-2`)."""
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)
    order: list[str] = []
    bus.on("test/serial", lambda: order.append("listener-1"))
    bus.on("test/serial", lambda: order.append("listener-2"))

    async def dispatch() -> None:
        await bus.serial("test/serial", yield_after_each=True)

    task = _eager_probe(dispatch())
    order.append("caller-continued")
    await task

    assert order == ["listener-1", "caller-continued", "listener-2"]


async def test_serial_yield_after_each_suspends_after_every_listener_not_only_the_first() -> None:
    """`L08-R002`, PASS 9: the yield is unconditional, per listener -- a THIRD listener still gets
    its own suspension boundary before it runs, not merely a single deferral after listener 1."""
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)
    order: list[str] = []
    bus.on("test/serial", lambda: order.append("listener-1"))
    bus.on("test/serial", lambda: order.append("listener-2"))
    bus.on("test/serial", lambda: order.append("listener-3"))

    async def dispatch() -> None:
        await bus.serial("test/serial", yield_after_each=True)

    task = _eager_probe(dispatch())
    order.append("after-eager-start")
    await asyncio.sleep(0)
    order.append("after-one-tick")
    await task

    assert order == [
        "listener-1",
        "after-eager-start",
        "listener-2",
        "after-one-tick",
        "listener-3",
    ]
