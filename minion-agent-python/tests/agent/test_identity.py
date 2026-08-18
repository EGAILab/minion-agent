"""One definition, many live instances -- the distinction section 6 fixes first."""

import pytest

from minion_agent.agent.identity import AgentDefinition, AgentStatus
from minion_agent.llm import ModelId


def _definition(**overrides: object) -> AgentDefinition:
    defaults: dict[str, object] = {
        "name": "ada",
        "model": ModelId("mock", "mock-1"),
        "system": "be helpful",
    }
    return AgentDefinition(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_a_definition_carries_reusable_configuration() -> None:
    definition = _definition()

    assert definition.name == "ada"
    assert definition.model == ModelId("mock", "mock-1")
    assert definition.system == "be helpful"


def test_a_definition_holds_no_conversation_state() -> None:
    """Anything conversational belongs to an instance, so a definition can be
    shared by many of them without coupling them together."""
    fields = set(AgentDefinition.__dataclass_fields__)

    assert fields == {"name", "model", "system", "max_steps"}


def test_max_steps_bounds_a_runaway_turn() -> None:
    assert _definition().max_steps == 16
    assert _definition(max_steps=2).max_steps == 2


def test_the_scope_name_is_derived_from_the_definition_name() -> None:
    """Definition-scoped registrations are shared by every instance of it."""
    assert _definition().scope_name == "agent-definition:ada"


def test_definitions_are_frozen() -> None:
    definition = _definition()

    with pytest.raises(Exception):  # noqa: B017
        definition.name = "changed"  # type: ignore[misc]


def test_status_has_exactly_two_states() -> None:
    """A third state would mean the settle signal has more than one meaning."""
    assert {status.value for status in AgentStatus} == {"idle", "running"}
