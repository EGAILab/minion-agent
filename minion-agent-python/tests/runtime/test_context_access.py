"""Attribute access and require() are two views over one resolution mechanism."""

from typing import Protocol

import pytest

from minion_agent.runtime.context import Context
from minion_agent.runtime.errors import ServiceConflictError, ServiceNotFoundError
from minion_agent.runtime.fiber import FiberState


class ToolService(Protocol):
    __service_name__ = "tools"

    def register(self, tool: str) -> None: ...


class FakeOwner:
    name = "owner"
    state = FiberState.ACTIVE


class FakeTools:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register(self, tool: str) -> None:
        self.registered.append(tool)


def test_attribute_access_resolves_a_service() -> None:
    ctx = Context()
    tools = FakeTools()
    ctx.registry.provide("tools", tools, FakeOwner())

    ctx.tools.register("bash")

    assert tools.registered == ["bash"]


def test_require_resolves_the_same_instance() -> None:
    ctx = Context()
    tools = FakeTools()
    ctx.registry.provide("tools", tools, FakeOwner())

    assert ctx.require(ToolService) is ctx.tools


def test_missing_service_raises_by_attribute() -> None:
    ctx = Context()

    with pytest.raises(ServiceNotFoundError, match="tools"):
        _ = ctx.tools


def test_missing_service_raises_by_require() -> None:
    ctx = Context()

    with pytest.raises(ServiceNotFoundError, match="tools"):
        ctx.require(ToolService)


def test_require_rejects_a_protocol_without_a_service_name() -> None:
    class Unnamed(Protocol): ...

    ctx = Context()

    with pytest.raises(TypeError, match="__service_name__"):
        ctx.require(Unnamed)


def test_extend_shares_the_registry_and_bus() -> None:
    root = Context()
    child = root.extend(label="child")

    assert child.registry is root.registry
    assert child.events is root.events
    assert child.root is root
    assert child.label == "child"


def test_child_sees_services_provided_after_extension() -> None:
    root = Context()
    child = root.extend()
    tools = FakeTools()

    root.registry.provide("tools", tools, FakeOwner())

    assert child.tools is tools


def test_child_cannot_shadow_a_parent_service() -> None:
    """Service shadowing is isolation realms, which are deferred. Scoped
    registration (a different mechanism) is not. See spec section 3."""
    root = Context()
    child = root.extend()
    root.registry.provide("tools", FakeTools(), FakeOwner())

    with pytest.raises(ServiceConflictError):
        child.registry.provide("tools", FakeTools(), FakeOwner())


def test_private_and_reserved_attributes_are_not_services() -> None:
    ctx = Context()

    with pytest.raises(AttributeError):
        _ = ctx._not_a_service

    assert ctx.fiber is None
