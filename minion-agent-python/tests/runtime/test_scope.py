"""Scopes nest, own their registrations, and dispose independently."""

import pytest

from minion_agent.runtime.context import Context
from minion_agent.runtime.errors import InactiveFiberError
from minion_agent.runtime.scope import ScopeKey, scope_of


def test_chain_is_nearest_first() -> None:
    root = ScopeKey("definition")
    instance = ScopeKey("instance", parent=root)
    turn = ScopeKey("turn", parent=instance)

    assert turn.chain() == (turn, instance, root)
    assert root.chain() == (root,)


def test_scope_of_reads_the_tag() -> None:
    ctx = Context()
    key = ScopeKey("agent-1")

    scope = ctx.scope(key)

    assert scope_of(scope.ctx) is key
    assert scope_of(ctx) is None


def test_derived_contexts_inherit_the_tag() -> None:
    ctx = Context()
    scope = ctx.scope(ScopeKey("agent-1"))

    child = scope.ctx.extend(label="derived")

    assert scope_of(child) is scope.key


def test_nested_scope_shadows_to_the_nearest_tag() -> None:
    ctx = Context()
    outer = ctx.scope(ScopeKey("outer"))
    inner_key = ScopeKey("inner", parent=outer.key)

    inner = outer.ctx.scope(inner_key)

    assert scope_of(inner.ctx) is inner_key


async def test_scope_owns_its_effects() -> None:
    order: list[str] = []
    ctx = Context()
    scope = ctx.scope(ScopeKey("agent-1"))

    scope.ctx.effect(lambda: lambda: order.append("scoped"), "scoped")

    await scope.dispose()

    assert order == ["scoped"]


async def test_dispose_is_idempotent() -> None:
    order: list[str] = []
    ctx = Context()
    scope = ctx.scope(ScopeKey("agent-1"))
    scope.ctx.effect(lambda: lambda: order.append("once"), "once")

    await scope.dispose()
    await scope.dispose()

    assert order == ["once"]


async def test_effects_unwind_in_reverse_within_a_scope() -> None:
    order: list[str] = []
    ctx = Context()
    scope = ctx.scope(ScopeKey("agent-1"))
    for label in ("first", "second", "third"):
        scope.ctx.effect(lambda label=label: lambda: order.append(label), label)

    await scope.dispose()

    assert order == ["third", "second", "first"]


async def test_disposing_a_scope_leaves_a_sibling_intact() -> None:
    order: list[str] = []
    ctx = Context()
    left = ctx.scope(ScopeKey("left"))
    right = ctx.scope(ScopeKey("right"))
    left.ctx.effect(lambda: lambda: order.append("left"), "left")
    right.ctx.effect(lambda: lambda: order.append("right"), "right")

    await left.dispose()

    assert order == ["left"]

    await right.dispose()

    assert order == ["left", "right"]


async def test_effect_on_a_disposed_scope_raises() -> None:
    ctx = Context()
    scope = ctx.scope(ScopeKey("agent-1"))
    await scope.dispose()

    with pytest.raises(InactiveFiberError):
        scope.ctx.effect(lambda: None, "too-late")


async def test_an_effect_disposer_can_be_called_before_the_scope() -> None:
    order: list[str] = []
    ctx = Context()
    scope = ctx.scope(ScopeKey("agent-1"))
    dispose = scope.ctx.effect(lambda: lambda: order.append("early"), "early")

    await dispose()
    await scope.dispose()

    assert order == ["early"]


async def test_disposing_a_parent_scope_disposes_a_still_live_child_first() -> None:
    """RT-012: descendant ownership follows nesting, not the caller.

    No `dispose_scope` step ever runs on `inner` -- disposing `outer` alone
    must settle `inner`, deepest first, as part of `Scope.dispose()`'s own
    behavior, not something a caller (or a conformance runner) computes.
    """
    order: list[str] = []
    ctx = Context()
    outer_key = ScopeKey("outer")
    outer = ctx.scope(outer_key)
    inner = outer.ctx.scope(ScopeKey("inner", parent=outer_key))
    outer.ctx.effect(lambda: lambda: order.append("outer"), "outer")
    inner.ctx.effect(lambda: lambda: order.append("inner"), "inner")

    await outer.dispose()

    assert order == ["inner", "outer"]
    assert inner.disposed
    assert outer.disposed


async def test_disposing_a_child_scope_leaves_the_parent_live() -> None:
    ctx = Context()
    outer_key = ScopeKey("outer")
    outer = ctx.scope(outer_key)
    inner = outer.ctx.scope(ScopeKey("inner", parent=outer_key))

    await inner.dispose()

    assert inner.disposed
    assert not outer.disposed


async def test_on_disposed_fires_once_for_direct_and_cascaded_disposal() -> None:
    fired: list[str] = []
    ctx = Context()
    outer_key = ScopeKey("outer")
    outer = ctx.scope(outer_key)
    inner = outer.ctx.scope(ScopeKey("inner", parent=outer_key))
    outer.on_disposed = lambda scope: fired.append(scope.key.name)
    inner.on_disposed = lambda scope: fired.append(scope.key.name)

    await outer.dispose()
    await outer.dispose()  # idempotent: must not fire on_disposed a second time

    assert fired == ["inner", "outer"]
