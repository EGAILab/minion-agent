"""Runner for `prepared_runtime` scenarios
(`conformance/schema/prepared-runtime-scenario.schema.json`, L0506-D001 / `TOOL-041`).

Runs the delta's certification gate (`gate: L0506-D001`, L0506-D001-R003): custom cases only.
Thin by design: one tool call per case through the REAL Layer 06 `execute_call` (resolve,
`prepare_arguments`, validation, the `tools/pre-execute` hook, `execute`). A pre-execute listener
records the prepared value at each observed pointer; a custom tool's `execute` records the same.
The runner decodes `prepare_set` tokens into the values a tool's own shim returns (fixture input)
and renders observed values back to tokens (observation); it never prepares, validates or converts
anything on the pipeline's behalf.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools.builtin._js import number_to_string
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult

SCHEMAS: dict[str, dict[str, Any]] = {
    "number": {
        "type": "object",
        "properties": {"limit": {"type": "number"}},
        "required": ["limit"],
    },
    "integer": {
        "type": "object",
        "properties": {"limit": {"type": "integer"}},
        "required": ["limit"],
    },
    "open": {"type": "object", "properties": {}},
    # L0506-D001-RC002: numeric keywords with no declared type (scenario schema `schema` $comment)
    **{
        kind: {"type": "object", "properties": {"limit": constraint}}
        for kind, constraint in {
            "bound-maximum": {"maximum": 0},
            "bound-minimum": {"minimum": 0},
            "bound-exclusive-maximum": {"exclusiveMaximum": 0},
            "bound-exclusive-minimum": {"exclusiveMinimum": 0},
            "multiple-of": {"multipleOf": 2},
            "one-of-bounds": {"oneOf": [{"maximum": 0}, {"minimum": 1}]},
            "not-bound": {"not": {"maximum": 0}},
            "number-bound": {"type": "number", "maximum": 0},
        }.items()
    },
}
_NAMED = {"+Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan, "-0": -0.0}
_FINITE_LITERAL = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")


def decode(token: str) -> float:
    """A `prepare_set` token as the runtime number a tool's own shim returns."""
    return _NAMED[token] if token in _NAMED else float(token)


def render(value: Any) -> str:
    """The token of an observed prepared value (ECMAScript Number::toString for a finite one)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return repr(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Infinity" if value > 0 else "-Infinity"
        if value == 0 and math.copysign(1.0, value) < 0:
            return "-0"
    return number_to_string(float(value))


def at(document: Any, pointer: str) -> Any:
    node = document
    for part in pointer.split("/")[1:]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def preflight(case: dict[str, Any]) -> None:
    """The schema's language-neutral PREFLIGHT; a violation fails the document."""
    expect = case["expect"]
    if expect["outcome"] == "prepared":
        assert set(expect["observed"]) == set(case["observe"]), f"{case['id']}: observed pointers"
    for token in [*case.get("prepare_set", {}).values(), *expect.get("observed", {}).values()]:
        assert token in _NAMED or (
            _FINITE_LITERAL.fullmatch(token) is not None and math.isfinite(float(token))
        ), f"{case['id']}: token {token!r}"


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one gate-`L0506-D001` (custom) case; returns `{outcome, text, hook, execute,
    raw_unchanged}`. A gate-`WP-13.2` real-`edit` case is not this runner's (L0506-D001-R003)."""
    assert case["tool"] == "custom", f"{case['id']}: not a delta-gate case"
    preflight(case)
    raw = copy.deepcopy(case["arguments"])
    seen: dict[str, list[str]] = {"hook": [], "execute": []}
    values = {key: decode(token) for key, token in case["prepare_set"].items()}

    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> ToolResult:
        seen["execute"] = [render(at(arguments, p)) for p in case["observe"]]
        return ToolResult(
            tool_call_id=tool_call_id, content=(TextBlock(text="ok"),), tool_name="probe"
        )

    definition = ToolDefinition(
        name="probe",
        label="probe",
        description="L0506-D001 probe",
        parameters=SCHEMAS[case["schema"]],
        execute=execute,
        prepare_arguments=lambda arguments: {**arguments, **values},
    )
    registry = ToolRegistry()
    registry.register(definition)
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(call: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        seen["hook"] = [render(at(arguments, p)) for p in case["observe"]]
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    call = ToolCallBlock(id="call-1", name=definition.name, arguments=case["arguments"])
    result = await execute_call(call, registry=registry, ctx=ctx)
    first = result.content[0]
    text = first.text if isinstance(first, TextBlock) else ""
    return {
        "outcome": "argument_validation_failure"
        if result.is_error and "invalid arguments" in text
        else ("error" if result.is_error else "prepared"),
        "text": text,
        "hook": seen["hook"],
        "execute": seen["execute"],
        "raw_unchanged": call.arguments == raw and case["arguments"] == raw,
    }
