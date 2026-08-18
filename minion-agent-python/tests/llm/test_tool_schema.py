"""Tool schemas are model-facing vocabulary, not a tools-package detail."""

from minion_agent.llm import ModelId, Request, ToolSchema


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
