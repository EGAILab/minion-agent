"""Parallel fans out and aggregates errors; serial runs in order and returns."""

import asyncio

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
