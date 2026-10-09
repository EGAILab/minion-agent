"""L08-D002 schema value snapshot (`L08D002-R003`).

Owner decision: minion-agent#171 issuecomment-6071887505. For one provider request the
authoritative tool-schema state is a value snapshot taken at that request's header-publication
point. The header records it and the provider request is built from the same snapshot, so an
application-owned `parameters` object mutated afterwards -- here, inside a real
`transformContext` listener, between publication and the provider call -- reaches neither. A
later request may see the change when its own snapshot is taken.

The header is read back from the stored tools artifact itself, not through `reconstruct_tools`,
so these witnesses do not depend on `L03-D001`.
"""

import json
from typing import Any

import pytest

from minion_agent.agent.events import AGENT_TRANSFORM_CONTEXT
from minion_agent.llm import StopReason, ToolCallBlock
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.tools import GrammarConstrainedSampling, JsonSchemaConstrainedSampling
from minion_agent.session import EventKind
from minion_agent.tools.definition import ToolDefinition

from .test_single_turn import _loop_with_adapter, _say


def _stored_tools(loop: Any) -> list[list[dict[str, Any]]]:
    """Every request header's tools artifact, decoded from its stored bytes, in log order."""
    return [
        json.loads(loop.artifacts.get(event.data["tools"]).decode("utf-8"))
        for event in loop.instance.log.events
        if event.kind == EventKind.REQUEST_HEADER
    ]


def _sent_tools(adapter: Any) -> list[list[dict[str, Any]]]:
    return [[schema.as_json() for schema in request.tools] for request in adapter.requests]


def _two_tools(loop: Any) -> dict[str, Any]:
    """`alpha` with a nested, application-owned `parameters` mapping; `beta` with sampling metadata.
    Returns alpha's mapping -- the application's own reference to it."""
    alpha_parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "enum": ["a", "b"]},
            "opts": {"type": "object", "properties": {"depth": {"type": "integer"}}},
        },
        "required": ["path"],
    }
    loop.tools.register(
        ToolDefinition(
            name="alpha",
            description="Read a path.",
            parameters=alpha_parameters,
            execute=lambda tool_call_id, args: "ok",
            label="alpha",
            constrained_sampling=JsonSchemaConstrainedSampling(strict="require"),
        )
    )
    loop.tools.register(
        ToolDefinition(
            name="beta",
            description="Write a line.",
            parameters={"type": "object", "properties": {"line": {"type": "string"}}},
            execute=lambda tool_call_id, args: "ok",
            label="beta",
            constrained_sampling=GrammarConstrainedSampling(
                openai_lark="start: WORD", openai_regex="[a-z]+"
            ),
        )
    )
    return alpha_parameters


PRE = {
    "name": "alpha",
    "description": "Read a path.",
    "parameters": {
        "properties": {
            "opts": {"properties": {"depth": {"type": "integer"}}, "type": "object"},
            "path": {"enum": ["a", "b"], "type": "string"},
        },
        "required": ["path"],
        "type": "object",
    },
    "constrained_sampling": {"strict": "require", "type": "json_schema"},
}
POST = {
    **PRE,
    "parameters": {
        "properties": {
            "opts": {"properties": {"depth": {"type": "integer"}}, "type": "object"},
            "path": {"enum": ["a", "b"], "type": "integer"},
        },
        "required": ["path", "opts"],
        "type": "object",
    },
}
BETA = {
    "name": "beta",
    "description": "Write a line.",
    "parameters": {"properties": {"line": {"type": "string"}}, "type": "object"},
    "constrained_sampling": {
        "type": "grammar",
        "variants": {"openai_lark": "start: WORD", "openai_regex": "[a-z]+"},
    },
}


async def test_header_and_request_carry_the_same_complete_schemas_in_order() -> None:
    """G, and F without mutation: two distinguishable tools, nested parameters and sampling
    metadata. Each request's stored header equals what that request sent."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse(
            (ToolCallBlock(id="t1", name="alpha", arguments={"path": "a"}),), StopReason.TOOL_USE
        ),
        ScriptedResponse((), StopReason.STOP),
    )
    _two_tools(loop)

    await loop.prompt(_say("go"))

    assert _sent_tools(adapter) == [[PRE, BETA], [PRE, BETA]]
    assert _stored_tools(loop) == _sent_tools(adapter)


@pytest.mark.xfail(
    strict=True, reason="L08D002-R003: Python schema value snapshot pending implementation"
)
async def test_a_transform_time_mutation_reaches_neither_the_published_header_nor_its_request() -> (
    None
):
    """A-F. A real transformContext listener mutates the application's own nested `parameters`
    mapping (a nested value and a nested list) after the first header is published. That header
    and the first provider request both keep the pre-mutation values, and the sampling metadata.
    The second request takes its own snapshot and may see the change; when it does, its header
    matches it."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse(
            (ToolCallBlock(id="t1", name="alpha", arguments={"path": "a"}),), StopReason.TOOL_USE
        ),
        ScriptedResponse((), StopReason.STOP),
    )
    alpha_parameters = _two_tools(loop)
    calls = [0]

    async def mutate(instance: Any, messages: Any, signal: Any, next_: Any) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            alpha_parameters["properties"]["path"]["type"] = "integer"
            alpha_parameters["required"].append("opts")
        return await next_()

    loop.instance.ctx.events.on(AGENT_TRANSFORM_CONTEXT, mutate)

    await loop.prompt(_say("go"))

    stored, sent = _stored_tools(loop), _sent_tools(adapter)
    assert stored[0] == [PRE, BETA]
    assert sent[0] == [PRE, BETA]
    assert sent[1] == [POST, BETA]
    assert stored[1] == sent[1]
