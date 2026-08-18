"""Edge paths that the behavioural suites do not reach.

These are ordinary reachable branches — misuse guards, optional shapes, and
accessors — covered here rather than exempted, so the coverage gate keeps
meaning what it says.
"""

from contextlib import contextmanager

import pytest

from minion_agent.runtime import Context, DispatchMode, FiberState, ScopeKey
from minion_agent.runtime.fiber import Fiber
from minion_agent.runtime.plugin import PluginSpec


def _spec(name: str = "edge-plugin") -> PluginSpec:
    async def noop(ctx, config):
        return None

    return PluginSpec(name=name, apply=noop, inject=(), config_model=None, provides=None)


def test_attribute_access_on_an_uninitialised_context_raises_attribute_error() -> None:
    """__getattr__ must not claim a service exists when no registry does."""
    bare = object.__new__(Context)

    with pytest.raises(AttributeError):
        _ = bare.tools


def test_effect_outside_a_plugin_raises() -> None:
    with pytest.raises(RuntimeError, match="requires a fiber"):
        Context().effect(lambda: None, "no-fiber")


def test_provide_outside_a_plugin_raises() -> None:
    with pytest.raises(RuntimeError, match="requires a fiber"):
        Context().provide("tools", object())


def test_on_outside_a_plugin_registers_directly_on_the_bus() -> None:
    ctx = Context()
    ctx.events.declare("test/emit", DispatchMode.EMIT)
    seen: list[str] = []

    dispose = ctx.on("test/emit", lambda: seen.append("heard"))
    ctx.events.emit("test/emit")
    dispose()
    ctx.events.emit("test/emit")

    assert seen == ["heard"]


async def test_scoped_effect_accepts_an_execute_returning_none() -> None:
    scope = Context().scope(ScopeKey("s"))

    scope.ctx.effect(lambda: None, "nothing-to-undo")

    await scope.dispose()


async def test_scoped_effect_accepts_a_context_manager() -> None:
    order: list[str] = []

    @contextmanager
    def managed():
        order.append("enter")
        yield
        order.append("exit")

    scope = Context().scope(ScopeKey("s"))
    scope.ctx.effect(managed, "managed")

    await scope.dispose()

    assert order == ["enter", "exit"]


async def test_scoped_effect_awaits_an_async_disposer() -> None:
    order: list[str] = []
    scope = Context().scope(ScopeKey("s"))

    async def teardown() -> None:
        order.append("async")

    scope.ctx.effect(lambda: teardown, "async-effect")

    await scope.dispose()

    assert order == ["async"]


async def test_a_scoped_effect_disposer_is_idempotent() -> None:
    order: list[str] = []
    scope = Context().scope(ScopeKey("s"))
    dispose = scope.ctx.effect(lambda: lambda: order.append("once"), "once")

    await dispose()
    await dispose()

    assert order == ["once"]


async def test_unloading_a_pending_fiber_is_a_noop() -> None:
    fiber = Fiber(name="edge", parent=Context(), plugin=_spec(), config=None)

    await fiber.unload()

    assert fiber.state is FiberState.PENDING


async def test_plugin_registry_exposes_its_fibers() -> None:
    root = Context()
    fiber = await root.plugin(_spec())

    assert root.plugins.fibers == (fiber,)


async def test_scope_reports_its_disposed_state() -> None:
    scope = Context().scope(ScopeKey("s"))

    assert not scope.disposed

    await scope.dispose()

    assert scope.disposed
