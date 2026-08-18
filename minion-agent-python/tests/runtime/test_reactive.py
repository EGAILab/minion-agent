"""Fibers load when their injected services appear and unload when they vanish."""

import pytest
from pydantic import BaseModel, ValidationError

from minion_agent.runtime.context import Context
from minion_agent.runtime.events import DispatchMode
from minion_agent.runtime.fiber import FiberState
from minion_agent.runtime.plugin import plugin


@plugin(name="provider", provides="tools")
async def provider(ctx, config):
    ctx.provide("tools", object())


async def test_dependent_stays_pending_until_its_service_appears() -> None:
    loaded: list[str] = []

    @plugin(name="consumer", inject=["tools"])
    async def consumer(ctx, config):
        loaded.append("consumer")

    root = Context()
    consumer_fiber = await root.plugin(consumer)

    assert consumer_fiber.state is FiberState.PENDING
    assert loaded == []

    await root.plugin(provider)

    assert consumer_fiber.state is FiberState.ACTIVE
    assert loaded == ["consumer"]


async def test_dependent_unloads_when_its_service_disappears() -> None:
    disposed: list[str] = []

    @plugin(name="consumer", inject=["tools"])
    async def consumer(ctx, config):
        ctx.effect(lambda: lambda: disposed.append("consumer-effect"), "consumer-effect")

    root = Context()
    provider_fiber = await root.plugin(provider)
    consumer_fiber = await root.plugin(consumer)
    assert consumer_fiber.state is FiberState.ACTIVE

    await root.plugins.unmount(provider_fiber)

    assert consumer_fiber.state is FiberState.PENDING
    assert disposed == ["consumer-effect"]


async def test_dependent_reloads_when_the_service_returns() -> None:
    loads: list[int] = []

    @plugin(name="consumer", inject=["tools"])
    async def consumer(ctx, config):
        loads.append(len(loads) + 1)

    root = Context()
    first_provider = await root.plugin(provider)
    await root.plugin(consumer)
    await root.plugins.unmount(first_provider)

    await root.plugin(provider)

    assert loads == [1, 2]


async def test_listeners_registered_via_ctx_on_are_auto_disposed() -> None:
    seen: list[str] = []

    @plugin(name="listener-plugin")
    async def listener_plugin(ctx, config):
        ctx.on("test/emit", lambda: seen.append("heard"))

    root = Context()
    root.events.declare("test/emit", DispatchMode.EMIT)
    fiber = await root.plugin(listener_plugin)

    root.events.emit("test/emit")
    assert seen == ["heard"]

    await root.plugins.unmount(fiber)
    root.events.emit("test/emit")

    assert seen == ["heard"]


async def test_config_is_validated_against_the_declared_model() -> None:
    class Config(BaseModel):
        timeout_ms: int = 120_000

    captured: dict[str, object] = {}

    @plugin(name="configured", config=Config)
    async def configured(ctx, config):
        captured["config"] = config

    root = Context()
    await root.plugin(configured, {"timeout_ms": 500})

    assert isinstance(captured["config"], Config)
    assert captured["config"].timeout_ms == 500  # type: ignore[union-attr]


async def test_invalid_config_raises_before_the_body_runs() -> None:
    class Config(BaseModel):
        timeout_ms: int

    ran: list[str] = []

    @plugin(name="configured", config=Config)
    async def configured(ctx, config):
        ran.append("body")

    root = Context()

    with pytest.raises(ValidationError):
        await root.plugin(configured, {"timeout_ms": "not-an-int"})

    assert ran == []


async def test_a_chain_of_dependents_activates_in_one_reconcile() -> None:
    """Reconciliation repeats until stable, so one service's arrival cascades."""
    order: list[str] = []

    @plugin(name="middle", inject=["tools"], provides="middleware")
    async def middle(ctx, config):
        order.append("middle")
        ctx.provide("middleware", object())

    @plugin(name="leaf", inject=["middleware"])
    async def leaf(ctx, config):
        order.append("leaf")

    root = Context()
    await root.plugin(leaf)
    await root.plugin(middle)
    assert order == []

    await root.plugin(provider)

    assert order == ["middle", "leaf"]
