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
