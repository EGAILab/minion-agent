"""The terminal continuation may be computed from the current arguments."""

from typing import Any

from minion_agent.runtime import DispatchMode, EventBus


def _bus() -> EventBus:
    bus = EventBus()
    bus.declare("test/chain", DispatchMode.WATERFALL)
    return bus


async def test_a_constant_terminal_still_works() -> None:
    """The existing contract, unchanged."""
    bus = _bus()

    assert await bus.waterfall("test/chain", "original", terminal="fixed") == "fixed"


async def test_an_empty_chain_yields_the_computed_terminal() -> None:
    bus = _bus()

    result = await bus.waterfall("test/chain", "original", terminal=lambda value: f"{value}!")

    assert result == "original!"


async def test_a_lone_transforming_listener_is_not_discarded() -> None:
    """The failure the terminal exists to prevent. With a constant terminal
    this returns the dispatcher's value and the transformation is lost."""
    bus = _bus()

    async def transform(value: str, next_: Any) -> Any:
        return await next_(value.upper())

    bus.on("test/chain", transform)

    result = await bus.waterfall("test/chain", "original", terminal=lambda value: value)

    assert result == "ORIGINAL"


async def test_registration_order_equals_application_order() -> None:
    bus = _bus()

    async def first(value: str, next_: Any) -> Any:
        return await next_(f"{value}-first")

    async def second(value: str, next_: Any) -> Any:
        return await next_(f"{value}-second")

    bus.on("test/chain", first)
    bus.on("test/chain", second)

    result = await bus.waterfall("test/chain", "x", terminal=lambda value: value)

    assert result == "x-first-second"


async def test_a_short_circuit_still_wins_over_the_terminal() -> None:
    bus = _bus()

    async def decide(value: str, next_: Any) -> str:
        return "owned"

    bus.on("test/chain", decide)

    assert await bus.waterfall("test/chain", "x", terminal=lambda v: v) == "owned"
