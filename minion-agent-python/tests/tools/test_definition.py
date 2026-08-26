"""A definition is what the registry stores and the model is told about."""

import pytest
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
    """Not a missing schema: a model needs to be told the tool takes nothing, or it has no way to
    call it correctly. The tool author supplies the empty schema explicitly (`L05-R005`);
    `None`/missing are not shorthand for it -- see the negative test below."""
    schema = _definition(parameters={"type": "object", "properties": {}}).schema()

    assert schema.parameters == {"type": "object", "properties": {}}


def test_a_tool_with_none_parameters_is_rejected() -> None:
    """`L05-R005`: pinned Pi's `Tool.parameters` is required. `None` is not a semantic alias for
    the empty schema -- it is rejected at construction, not only by typing."""
    with pytest.raises(TypeError, match="parameters"):
        _definition(parameters=None)


@pytest.mark.parametrize("bad", [True, False, [], "schema", 42], ids=lambda v: type(v).__name__)
def test_a_tool_with_a_non_mapping_parameters_value_is_rejected(bad: object) -> None:
    """The "object-valued JSON Schema" boundary means the JSON *representation* of the schema is
    a mapping -- it excludes the JSON-Schema-spec boolean-shorthand forms and any other
    non-mapping value (`L05-R005`)."""
    with pytest.raises(TypeError, match="parameters"):
        _definition(parameters=bad)


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param({"type": "object", "properties": {}}, id="empty-object-instance"),
        pytest.param({"type": "string"}, id="non-object-instance"),
        pytest.param(
            {"oneOf": [{"type": "string"}, {"type": "number"}]}, id="top-level-combinator"
        ),
        pytest.param({"$comment": "kept", "custom-extension": {"x": 1}}, id="arbitrary-members"),
    ],
)
def test_a_tool_accepts_any_object_valued_schema_representation(
    parameters: dict[str, object],
) -> None:
    """`L05-R005`: "object-valued" describes the schema's own JSON representation (a mapping),
    not a requirement that the schema describe an object *instance* via a top-level `"type":
    "object"` keyword. Pinned Pi's `Tool<TParameters extends TSchema>` is generic over TypeBox's
    whole `TSchema` domain, not narrowed to `TObject` -- a non-object-instance schema and a
    top-level combinator are equally valid tool parameter schemas. An earlier repair mistakenly
    required `parameters["type"] == "object"`, which rejected both. Arbitrary object members
    (keywords Layer 05 does not itself interpret) must survive unchanged -- Layer 05 is not a
    JSON Schema validator."""
    schema = _definition(parameters=parameters).schema()

    assert schema.parameters == parameters


def test_execution_modes_are_exactly_two() -> None:
    assert {mode.value for mode in ExecutionMode} == {"parallel", "sequential"}
