"""Service registration is exclusive; visibility is narrower than registration."""

from dataclasses import dataclass

import pytest

from minion_agent.runtime.errors import ServiceConflictError
from minion_agent.runtime.fiber import FiberState
from minion_agent.runtime.service import ServiceRegistry


@dataclass
class FakeOwner:
    """Stands in for a Fiber; the registry only reads `state` and `name`."""

    name: str = "owner"
    state: FiberState = FiberState.ACTIVE


def test_provide_then_resolve() -> None:
    registry = ServiceRegistry()
    owner = FakeOwner()

    registry.provide("tools", "tool-service", owner)

    assert registry.resolve("tools") == "tool-service"
    assert registry.has("tools")


def test_second_provider_raises_naming_the_holder() -> None:
    registry = ServiceRegistry()
    registry.provide("tools", "first", FakeOwner(name="holder"))

    with pytest.raises(ServiceConflictError, match="holder"):
        registry.provide("tools", "second", FakeOwner(name="latecomer"))


def test_revoking_frees_the_name_for_a_new_provider() -> None:
    registry = ServiceRegistry()
    revoke = registry.provide("tools", "first", FakeOwner())

    revoke()
    registry.provide("tools", "second", FakeOwner())

    assert registry.resolve("tools") == "second"


def test_no_fallback_to_an_earlier_provider() -> None:
    registry = ServiceRegistry()
    first_revoke = registry.provide("tools", "first", FakeOwner())
    first_revoke()
    second_revoke = registry.provide("tools", "second", FakeOwner())

    second_revoke()

    assert registry.resolve("tools") is None
    assert not registry.has("tools")


def test_inactive_owner_hides_the_service() -> None:
    registry = ServiceRegistry()
    owner = FakeOwner(state=FiberState.LOADING)
    registry.provide("tools", "value", owner)

    assert registry.resolve("tools") is None
    assert registry.resolve("tools", strict=False) == "value"


def test_check_predicate_narrows_visibility() -> None:
    registry = ServiceRegistry()
    visible = False
    registry.provide("tools", "value", FakeOwner(), check=lambda: visible)

    assert registry.resolve("tools") is None

    visible = True
    assert registry.resolve("tools") == "value"


def test_check_predicate_applies_even_when_not_strict() -> None:
    registry = ServiceRegistry()
    registry.provide("tools", "value", FakeOwner(state=FiberState.LOADING), check=lambda: False)

    assert registry.resolve("tools", strict=False) is None


def test_revoking_twice_is_a_no_op() -> None:
    registry = ServiceRegistry()
    revoke = registry.provide("tools", "value", FakeOwner())

    revoke()
    revoke()

    assert registry.resolve("tools") is None


def test_names_lists_registered_services() -> None:
    registry = ServiceRegistry()
    registry.provide("tools", "a", FakeOwner())
    registry.provide("llm", "b", FakeOwner())

    assert registry.names() == frozenset({"tools", "llm"})


def test_impl_of_exposes_the_registration_regardless_of_visibility() -> None:
    registry = ServiceRegistry()
    registry.provide("tools", "value", FakeOwner(state=FiberState.LOADING))

    impl = registry.impl_of("tools")

    assert impl is not None
    assert impl.value == "value"
    assert not impl.is_visible()
    assert registry.impl_of("missing") is None
