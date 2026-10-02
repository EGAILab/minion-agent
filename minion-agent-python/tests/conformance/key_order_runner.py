"""Runner for `key_order` scenarios (`conformance/schema/key-order-scenario.schema.json`,
L0206-D001 / K1: ECMAScript own-property enumeration order of tool-argument objects).

Thin by design. The case's `arguments` insertion sequence becomes a `dict` built by plain
assignment in that order (a repeated key keeps its first position and takes the last value, as
`JSON.parse` does), and ONE `ToolCallBlock` carries it. The runner then only OBSERVES the recursive
key enumeration: the constructed call (`raw`, also the `tools/execution-start` payload), an
`AssistantMessage` appended to a fresh `SessionLog` and replayed (`replay`), the `tools/pre-execute`
listener before any mutation (`hook`), and `execute`. A `mutate` case's listener assigns into the
arguments object in place; a `replace` case's listener returns `Proceed(arguments=...)`. Ordering
is the pipeline's job: the runner never sorts or reorders.
"""

from __future__ import annotations

from typing import Any

from minion_agent.execution import LocalFileSystem
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.llm.messages import AssistantMessage, StopReason, Usage
from minion_agent.runtime import Context
from minion_agent.session.derive import decode_message, encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.tools.builtin import create_edit_tool
from minion_agent.tools.decisions import Proceed
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult


def build(value: Any) -> Any:
    """The fixture grammar as Python values, objects by plain assignment in insertion order."""
    if isinstance(value, list):
        return [build(item) for item in value]
    if isinstance(value, dict):
        built: dict[str, Any] = {}
        for key, item in value["$o"]:
            built[key] = build(item)
        return built
    return value


def observe(value: Any) -> Any:
    """The recursive key enumeration, exactly as the object iterates."""
    if isinstance(value, list):
        return {"a": [observe(item) for item in value]}
    if isinstance(value, dict):
        return {"o": [[key, observe(item)] for key, item in value.items()]}
    return value


async def run_case(case: dict[str, Any], root: str) -> dict[str, Any]:
    call = ToolCallBlock(id="call-1", name="probe", arguments=build(case["arguments"]))
    seen: dict[str, Any] = {"raw": observe(call.arguments)}
    message = AssistantMessage(
        content=(call,),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(),
        model="m",
        provider="p",
        timestamp=0,
    )
    event = SessionLog(session_id="s").append(
        EventKind.ASSISTANT_MESSAGE, {"message": encode_message(message)}
    )
    replayed = decode_message(event.data["message"]).content[0]
    assert isinstance(replayed, ToolCallBlock)
    seen["replay"] = observe(replayed.arguments)

    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> ToolResult:
        seen["execute"] = observe(arguments)
        return ToolResult(
            tool_call_id=tool_call_id, content=(TextBlock(text="ok"),), tool_name="probe"
        )

    edit = create_edit_tool(LocalFileSystem(root))
    schema = edit.parameters if case["schema"] == "edit" else case["schema"]
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="probe",
            label="probe",
            description="L0206-D001 key-order probe",
            parameters=schema,
            execute=execute,
            prepare_arguments=edit.prepare_arguments if case.get("prepare") == "edit" else None,
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(c: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        seen["hook"] = observe(arguments)
        for key, item in case.get("mutate", []):
            arguments[key] = build(item)
        if "replace" in case:
            return Proceed(arguments=build(case["replace"]))
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    result = await execute_call(call, registry=registry, ctx=ctx)
    first = result.content[0]
    seen["result"] = (result.is_error, first.text if isinstance(first, TextBlock) else "")
    return seen


BOUNDARIES = ("raw", "replay", "hook", "execute")


def check(case: dict[str, Any], seen: dict[str, Any]) -> None:
    assert seen["result"] == (False, "ok"), (case["id"], seen["result"])
    for boundary in BOUNDARIES:
        want = case["expect"][boundary]
        assert seen.get(boundary) == want, (case["id"], boundary, seen.get(boundary), want)
