"""A definition is what the registry stores and the model is told about."""

from pydantic import BaseModel

from minion_agent.tools.definition import ExecutionMode, ToolDefinition


class EchoParams(BaseModel):
    value: str
    times: int = 1


def _definition(**overrides: object) -> ToolDefinition:
    defaults: dict[str, object] = {
        "name": "echo",
        "description": "repeat a value",
        "parameters": EchoParams,
        "execute": lambda args: "ok",
        "label": "Echo",
    }
    return ToolDefinition(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_a_definition_defaults_to_no_per_tool_mode_override() -> None:
    """`None` -- not a concrete `PARALLEL` -- is the correct default (TOOL-F004): pinned Pi's
    `AgentTool.executionMode?` is optional, and its absence means "defer to whatever run-level
    default applies," which is not observably the same as "this tool explicitly wants parallel."
    Most tools are still safe to overlap in practice; sequential is the claim that carries
    consequences, because one such tool serializes an entire batch."""
    assert _definition().mode is None


def test_a_definition_may_declare_itself_sequential() -> None:
    assert _definition(mode=ExecutionMode.SEQUENTIAL).mode is ExecutionMode.SEQUENTIAL


def test_the_schema_is_derived_from_the_pydantic_model() -> None:
    schema = _definition().schema()

    assert schema.name == "echo"
    assert schema.description == "repeat a value"
    assert schema.parameters["properties"]["value"]["type"] == "string"
    assert schema.parameters["required"] == ["value"]


def test_a_defaulted_field_is_not_required() -> None:
    assert "times" not in _definition().schema().parameters["required"]


def test_a_tool_without_parameters_gets_an_empty_object_schema() -> None:
    """Not a missing schema: a model needs to be told the tool takes nothing,
    or it has no way to call it correctly."""
    schema = _definition(parameters=None).schema()

    assert schema.parameters == {"type": "object", "properties": {}}


def test_execution_modes_are_exactly_two() -> None:
    assert {mode.value for mode in ExecutionMode} == {"parallel", "sequential"}
