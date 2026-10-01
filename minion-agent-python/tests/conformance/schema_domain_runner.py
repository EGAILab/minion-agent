"""Runner for `schema_domain` scenarios (`conformance/schema/schema-domain-scenario.schema.json`,
L05-D001 / `TOOL-016`, `TOOL-003`: JavaScript strings inside the runtime-validation schema).

Thin by design: the case's `schema` becomes ONE tool's `ToolDefinition.parameters` (no
`prepare_arguments`), its `arguments` one call's raw arguments, and the runner observes only the
outcome of the REAL Layer-06 `execute_call` -- `accept` when `execute` ran, `reject` for the
certified immediate argument-validation error. It decodes the value grammar (fixture input) and
never validates, normalizes or converts anything on the pipeline's behalf.
"""

from __future__ import annotations

import struct
from typing import Any

from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult


def string(code_units: list[int]) -> str:
    """UTF-16 code units as the Python string a binding carries (pairs combined, lone kept)."""
    return struct.pack(f"<{len(code_units)}H", *code_units).decode("utf-16-le", "surrogatepass")


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if "utf16" in value:
            return string(value["utf16"])
        return {string(key): decode(item) for key, item in value["$keys"]}
    return value


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    ran: list[bool] = []

    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> ToolResult:
        ran.append(True)
        return ToolResult(
            tool_call_id=tool_call_id, content=(TextBlock(text="ok"),), tool_name="probe"
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="probe",
            label="probe",
            description="L05-D001 probe (no prepare_arguments)",
            parameters=decode(case["schema"]),
            execute=execute,
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)
    call = ToolCallBlock(id="call-1", name="probe", arguments=decode(case["arguments"]))
    result = await execute_call(call, registry=registry, ctx=ctx)
    first = result.content[0]
    text = first.text if isinstance(first, TextBlock) else ""
    if not result.is_error and ran:
        outcome = "accept"
    elif result.is_error and not ran and "invalid arguments" in text:
        outcome = "reject"
    else:
        outcome = "error"
    return {"outcome": outcome, "text": text}


def check(case: dict[str, Any], run: dict[str, Any]) -> None:
    assert run["outcome"] == case["expect"], (case["id"], run)
