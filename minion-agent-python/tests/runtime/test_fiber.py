"""Fibers own lifecycle state and reverse-unwound effects."""

from contextlib import contextmanager

import pytest

from minion_agent.runtime.context import Context
from minion_agent.runtime.errors import InactiveFiberError
from minion_agent.runtime.fiber import Fiber, FiberState
from minion_agent.runtime.plugin import PluginSpec


def _spec(body=None, *, inject=(), name="test-plugin"):
    async def noop(ctx, config):
        return None

    return PluginSpec(
        name=name,
        apply=body or noop,
        inject=tuple(inject),
        config_model=None,
        provides=None,
    )


async def test_effects_unwind_in_reverse_on_unload() -> None:
    order: list[str] = []

    async def body(ctx, config):
        ctx.effect(lambda: lambda: order.append("first"), "first")
        ctx.effect(lambda: lambda: order.append("second"), "second")

    fiber = Fiber(name="subject", parent=Context(), plugin=_spec(body), config=None)
    await fiber.load()
    assert fiber.state is FiberState.ACTIVE

    await fiber.unload()

    assert order == ["second", "first"]
    assert fiber.state is FiberState.PENDING


async def test_effect_disposer_can_be_called_early_and_is_idempotent() -> None:
    order: list[str] = []
    captured: dict[str, object] = {}

    async def body(ctx, config):
        captured["dispose"] = ctx.effect(lambda: lambda: order.append("once"), "once")

    fiber = Fiber(name="subject", parent=Context(), plugin=_spec(body), config=None)
    await fiber.load()

    dispose = captured["dispose"]
    await dispose()  # type: ignore[operator]
    await dispose()  # type: ignore[operator]
    await fiber.unload()

    assert order == ["once"]


async def test_effect_on_a_disposed_fiber_raises() -> None:
    fiber = Fiber(name="subject", parent=Context(), plugin=_spec(), config=None)
    await fiber.load()
    await fiber.dispose()

    with pytest.raises(InactiveFiberError):
        fiber.effect(lambda: None, "too-late")


async def test_a_failing_body_marks_the_fiber_failed_and_unwinds() -> None:
    order: list[str] = []

    async def body(ctx, config):
        ctx.effect(lambda: lambda: order.append("created-before-failure"), "early")
        raise ValueError("boom")

    fiber = Fiber(name="subject", parent=Context(), plugin=_spec(body), config=None)

    await fiber.load()

    assert fiber.state is FiberState.FAILED
    assert order == ["created-before-failure"]


async def test_state_changes_are_reported_in_order() -> None:
    fiber = Fiber(name="subject", parent=Context(), plugin=_spec(), config=None)
    seen: list[FiberState] = []
    fiber.on_state_change = lambda _fiber, state: seen.append(state)

    await fiber.load()
    await fiber.dispose()

    assert seen == [
        FiberState.LOADING,
        FiberState.ACTIVE,
        FiberState.UNLOADING,
        FiberState.DISPOSED,
    ]


async def test_generator_effects_are_supported() -> None:
    order: list[str] = []

    @contextmanager
    def managed():
        order.append("enter")
        yield
        order.append("exit")

    async def body(ctx, config):
        ctx.effect(managed, "managed")

    fiber = Fiber(name="subject", parent=Context(), plugin=_spec(body), config=None)
    await fiber.load()
    await fiber.unload()

    assert order == ["enter", "exit"]


async def test_effects_reporting_hook_sees_creation_and_disposal() -> None:
    seen: list[tuple[str, str]] = []

    async def body(ctx, config):
        ctx.effect(lambda: None, "labelled")

    fiber = Fiber(name="subject", parent=Context(), plugin=_spec(body), config=None)
    fiber.on_effect = lambda _f, phase, label: seen.append((phase, label))

    await fiber.load()
    await fiber.unload()

    assert seen == [("created", "labelled"), ("disposed", "labelled")]


async def test_dispose_is_idempotent_and_load_after_dispose_is_a_noop() -> None:
    fiber = Fiber(name="subject", parent=Context(), plugin=_spec(), config=None)
    await fiber.load()
    await fiber.dispose()
    await fiber.dispose()

    await fiber.load()

    assert fiber.state is FiberState.DISPOSED
