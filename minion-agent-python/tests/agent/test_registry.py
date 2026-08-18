"""The registry owns instances; a handle owns exactly one instance's teardown."""

import pytest

from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.registry import AgentRegistry, DuplicateInstanceError
from minion_agent.llm import ModelId
from minion_agent.runtime import Context
from minion_agent.session import SessionService


def _definition(name: str = "ada") -> AgentDefinition:
    return AgentDefinition(name=name, model=ModelId("mock", "mock-1"))


def _registry() -> AgentRegistry:
    return AgentRegistry(ctx=Context(), sessions=SessionService())


async def test_create_returns_a_handle_to_a_live_instance() -> None:
    registry = _registry()

    handle = registry.create("room-a", _definition())

    assert handle.instance.id == "room-a"
    assert registry.get("room-a") is handle.instance


async def test_many_instances_share_one_definition() -> None:
    registry = _registry()
    definition = _definition()

    first = registry.create("room-a", definition)
    second = registry.create("room-b", definition)

    assert first.instance.definition is second.instance.definition
    assert first.instance is not second.instance


async def test_each_instance_gets_its_own_session_log() -> None:
    registry = _registry()

    first = registry.create("room-a", _definition())
    second = registry.create("room-b", _definition())

    assert first.instance.log is not second.instance.log
    assert first.instance.log.session_id == "room-a"


async def test_a_duplicate_id_is_rejected() -> None:
    registry = _registry()
    registry.create("room-a", _definition())

    with pytest.raises(DuplicateInstanceError, match="room-a"):
        registry.create("room-a", _definition())


async def test_disposing_a_handle_removes_the_instance() -> None:
    registry = _registry()
    handle = registry.create("room-a", _definition())

    await handle.dispose()

    assert registry.get("room-a") is None
    assert registry.instances() == ()


async def test_disposing_twice_is_harmless() -> None:
    registry = _registry()
    handle = registry.create("room-a", _definition())

    await handle.dispose()
    await handle.dispose()


async def test_disposing_one_instance_leaves_its_siblings_alone() -> None:
    registry = _registry()
    first = registry.create("room-a", _definition())
    registry.create("room-b", _definition())

    await first.dispose()

    assert registry.get("room-b") is not None


async def test_disposing_unwinds_instance_scoped_registrations() -> None:
    registry = _registry()
    handle = registry.create("room-a", _definition())
    order: list[str] = []
    handle.instance.ctx.effect(lambda: lambda: order.append("scoped"), "scoped")

    await handle.dispose()

    assert order == ["scoped"]


async def test_the_id_becomes_reusable_after_disposal() -> None:
    registry = _registry()
    handle = registry.create("room-a", _definition())
    await handle.dispose()

    reborn = registry.create("room-a", _definition())

    assert reborn.instance is not handle.instance
