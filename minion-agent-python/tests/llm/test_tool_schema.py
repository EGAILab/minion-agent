"""Tool schemas are model-facing vocabulary, not a tools-package detail."""

from minion_agent.llm import (
    GrammarConstrainedSampling,
    JsonSchemaConstrainedSampling,
    ModelId,
    Request,
    ToolSchema,
)


def _schema(name: str = "echo") -> ToolSchema:
    return ToolSchema(
        name=name,
        description="repeat a value",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )


def test_a_schema_carries_what_a_model_needs() -> None:
    schema = _schema()

    assert schema.name == "echo"
    assert schema.description == "repeat a value"
    assert schema.parameters["required"] == ["value"]


def test_a_request_carries_no_tools_by_default() -> None:
    """A request without tools is the ordinary case and must stay expressible."""
    request = Request(model=ModelId("mock", "mock-1"), system="", messages=())

    assert request.tools == ()


def test_a_request_carries_its_visible_tools() -> None:
    request = Request(model=ModelId("mock", "mock-1"), system="", messages=(), tools=(_schema(),))

    assert [tool.name for tool in request.tools] == ["echo"]


def test_the_canonical_form_is_order_independent() -> None:
    """It is content-addressed into the log, so two equal schemas built in
    different key orders must hash the same."""
    first = ToolSchema(name="a", description="d", parameters={"x": 1, "y": 2})
    second = ToolSchema(name="a", description="d", parameters={"y": 2, "x": 1})

    assert first.as_json() == second.as_json()


def test_the_canonical_form_is_json_safe() -> None:
    import json

    assert json.dumps(_schema().as_json())


# --- constrained sampling (TOOL-F005): absent | false | json_schema | grammar ---------------


def test_absent_constrained_sampling_serializes_as_null() -> None:
    assert _schema().as_json()["constrained_sampling"] is None


def test_false_constrained_sampling_is_preserved_exactly() -> None:
    schema = ToolSchema(name="a", description="d", parameters={}, constrained_sampling=False)

    assert schema.as_json()["constrained_sampling"] is False


def test_json_schema_constrained_sampling_round_trips() -> None:
    schema = ToolSchema(
        name="a",
        description="d",
        parameters={},
        constrained_sampling=JsonSchemaConstrainedSampling(strict="require"),
    )

    assert schema.as_json()["constrained_sampling"] == {"type": "json_schema", "strict": "require"}


def test_grammar_constrained_sampling_round_trips() -> None:
    schema = ToolSchema(
        name="a",
        description="d",
        parameters={},
        constrained_sampling=GrammarConstrainedSampling(openai_lark="grammar text"),
    )

    assert schema.as_json()["constrained_sampling"] == {
        "type": "grammar",
        "variants": {"openai_lark": "grammar text"},
    }


def test_grammar_constrained_sampling_openai_regex_round_trips() -> None:
    """The closed GrammarFormat domain has exactly two independently-optional formats
    (`L05-R001`, pinned Pi `packages/ai/src/types.ts::GrammarFormat`) -- confirm the second
    one is accepted too, not only the one exercised above."""
    schema = ToolSchema(
        name="a",
        description="d",
        parameters={},
        constrained_sampling=GrammarConstrainedSampling(openai_regex="^[a-z]+$"),
    )

    assert schema.as_json()["constrained_sampling"] == {
        "type": "grammar",
        "variants": {"openai_regex": "^[a-z]+$"},
    }


def test_grammar_constrained_sampling_both_formats_round_trip() -> None:
    """Pi's `GrammarVariants = Partial<Record<GrammarFormat, string>>` permits both formats
    set simultaneously; each is independently optional, not mutually exclusive."""
    schema = ToolSchema(
        name="a",
        description="d",
        parameters={},
        constrained_sampling=GrammarConstrainedSampling(
            openai_lark="start: WORD+", openai_regex="^[a-z]+$"
        ),
    )

    assert schema.as_json()["constrained_sampling"] == {
        "type": "grammar",
        "variants": {"openai_lark": "start: WORD+", "openai_regex": "^[a-z]+$"},
    }
