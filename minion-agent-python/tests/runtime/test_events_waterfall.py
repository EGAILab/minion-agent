"""Waterfall: one contract serving both decision and transformation patterns."""

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from minion_agent.runtime.errors import EventModeError, WaterfallError
from minion_agent.runtime.events import DispatchMode, EventBus


def _bus() -> EventBus:
    bus = EventBus()
    bus.declare("test/waterfall", DispatchMode.WATERFALL)
    return bus


async def test_listeners_delegate_through_next() -> None:
    bus = _bus()
    seen: list[str] = []

    async def outer(next_: Callable[..., Awaitable[Any]]) -> Any:
        seen.append("outer-before")
        result = await next_()
        seen.append("outer-after")
        return result

    async def inner(next_: Callable[..., Awaitable[Any]]) -> Any:
        seen.append("inner")
        return "inner-result"

    bus.on("test/waterfall", outer)
    bus.on("test/waterfall", inner)

    assert await bus.waterfall("test/waterfall") == "inner-result"
    assert seen == ["outer-before", "inner", "outer-after"]


async def test_short_circuit_skips_downstream_listeners() -> None:
    bus = _bus()
    seen: list[str] = []

    async def decider(next_: Callable[..., Awaitable[Any]]) -> Any:
        seen.append("decider")
        return "decided"

    async def never(next_: Callable[..., Awaitable[Any]]) -> Any:
        seen.append("never")
        return "never-result"

    bus.on("test/waterfall", decider)
    bus.on("test/waterfall", never)

    assert await bus.waterfall("test/waterfall") == "decided"
    assert seen == ["decider"]


async def test_upstream_may_replace_the_downstream_result() -> None:
    bus = _bus()

    async def replacer(next_: Callable[..., Awaitable[Any]]) -> Any:
        await next_()
        return "replaced"

    async def original(next_: Callable[..., Awaitable[Any]]) -> Any:
        return "original"

    bus.on("test/waterfall", replacer)
    bus.on("test/waterfall", original)

    assert await bus.waterfall("test/waterfall") == "replaced"


async def test_terminal_is_returned_when_every_listener_delegates() -> None:
    """The transformation pattern depends on this: a fully cooperative chain
    must yield the transformed payload, never None."""
    bus = _bus()
    seen: list[int] = []

    async def first(value: int, next_: Callable[..., Awaitable[Any]]) -> Any:
        seen.append(value)
        return await next_(value + 1)

    async def second(value: int, next_: Callable[..., Awaitable[Any]]) -> Any:
        seen.append(value)
        return await next_(value + 1)

    bus.on("test/waterfall", first)
    bus.on("test/waterfall", second)

    result = await bus.waterfall("test/waterfall", 1, terminal="unset")

    assert seen == [1, 2]
    assert result == "unset"


async def test_replacement_arguments_reach_downstream_listeners() -> None:
    bus = _bus()
    seen: list[str] = []

    async def rewriter(text: str, next_: Callable[..., Awaitable[Any]]) -> Any:
        seen.append(text)
        return await next_("rewritten")

    async def last(text: str, next_: Callable[..., Awaitable[Any]]) -> Any:
        seen.append(text)
        return text

    bus.on("test/waterfall", rewriter)
    bus.on("test/waterfall", last)

    assert await bus.waterfall("test/waterfall", "original") == "rewritten"
    assert seen == ["original", "rewritten"]


async def test_empty_chain_returns_the_terminal() -> None:
    bus = _bus()

    assert await bus.waterfall("test/waterfall", terminal="fallback") == "fallback"


async def test_terminal_defaults_to_none() -> None:
    bus = _bus()

    assert await bus.waterfall("test/waterfall") is None


async def test_calling_next_twice_raises() -> None:
    """Memoizing next is incoherent once it accepts replacement arguments:
    a second call carrying different arguments has no defensible answer."""
    bus = _bus()

    async def greedy(next_: Callable[..., Awaitable[Any]]) -> Any:
        await next_()
        await next_()
        return "unreachable"

    async def last(next_: Callable[..., Awaitable[Any]]) -> Any:
        return "value"

    bus.on("test/waterfall", greedy)
    bus.on("test/waterfall", last)

    with pytest.raises(WaterfallError, match="at most once"):
        await bus.waterfall("test/waterfall")


async def test_sync_listeners_are_supported() -> None:
    bus = _bus()
    bus.on("test/waterfall", lambda next_: "sync-result")

    assert await bus.waterfall("test/waterfall") == "sync-result"


async def test_waterfall_rejects_wrong_mode() -> None:
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)

    with pytest.raises(EventModeError):
        await bus.waterfall("test/serial")
