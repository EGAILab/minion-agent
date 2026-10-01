"""Runner for `tool_result_domain` scenarios
(`conformance/schema/tool-result-domain-scenario.schema.json`, L0506-D003 / `AI-006`, `TOOL-005`,
`TOOL-017`, `MINION-002`: the tool-result runtime value domain).

Thin by design, and through the REAL composed stack (`L0506-D003-R002`): a scripted mock provider
answers one turn with ONE tool call, then a final reply. The tool's `execute` returns the case's
`tool.returns` (decoded from the L0206-D002 value grammar, plus `NaN`) or raises `tool.throws`;
the case's `hook` is one after-hook registered through `register_after_tool_call_hook`. The agent
loop itself finalizes the result, builds the `ToolResultMessage`, appends it to the session log and
dispatches its events. The runner only OBSERVES:

    hook                     what the after-hook receives
    execution_end            the live `tools/execution-end` payload (`tool_execution_end`)
    message                  the live `MessageEnd` the loop dispatches for the tool result
    session                  the tool result in `derive_messages(log)`: committed-history replay
    replayed_execution_end   the `ToolExecutionEnd` that `project(log)` rebuilds from the log
                             (Layer 08 event replay; must equal `execution_end`, R001)

Strings render to UTF-16 code units, numbers to tokens, objects to key -> observation maps
(compared as sets: K1 order is L0206-D001's). The runner never decodes, normalizes or converts on
the pipeline's behalf. The gate-WP-13.2 document registers the REAL `edit` tool over the real
`LocalFileSystem` instead of a fixture tool.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from minion_agent.agent.events import AGENT_LIFECYCLE_EVENT
from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.plugin import agents_plugin
from minion_agent.agent.projection import MessageEnd, ToolExecutionEnd, project
from minion_agent.agent_loop import agent_loop_plugin
from minion_agent.execution import LocalFileSystem
from minion_agent.llm import ModelId, TextBlock, ToolCallBlock, ToolResultMessage, UserMessage
from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.llm.plugin import llm_plugin
from minion_agent.runtime import Context
from minion_agent.session import derive_messages
from minion_agent.session.service import session_plugin
from minion_agent.tools.builtin import create_edit_tool
from minion_agent.tools.decisions import AfterToolCallOverride
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_EXECUTION_END
from minion_agent.tools.execute import register_after_tool_call_hook
from minion_agent.tools.plugin import tools_plugin
from minion_agent.tools.result import ToolResult

from .raw_arguments_runner import canonical_finite, expect, observe, string

# L0506-D003: the result domain is the raw token grammar plus NaN (not JSON.parse output).
NAMED = {"+Infinity": math.inf, "-Infinity": -math.inf, "-0": -0.0, "NaN": math.nan}


def preflight(value: Any) -> None:
    """Fail the document on any number token outside the grammar; never defaulted."""
    if isinstance(value, list):
        for item in value:
            preflight(item)
    elif isinstance(value, dict):
        if "number" in value:
            token = value["number"]
            assert token in NAMED or canonical_finite(token), f"number token {token!r}"
        elif "$keys" in value:
            for _, item in value["$keys"]:
                preflight(item)
        elif "utf16" not in value:
            for item in value.values():
                preflight(item)


def number(token: str) -> int | float:
    """A preflighted token as the binding value, through binary64 (CE-L0206-D002-01 N1/N2)."""
    if token in NAMED:
        return NAMED[token]
    value = float(token)
    return int(value) if not any(mark in token for mark in ".eE") else value


def decode(value: Any) -> Any:
    """A scenario value as the runtime value (fixture input only)."""
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if "utf16" in value:
            return string(value["utf16"])
        if "number" in value:
            return number(value["number"])
        if "$keys" in value:
            return {string(key): decode(item) for key, item in value["$keys"]}
        return {key: decode(item) for key, item in value.items()}
    return value


def content(blocks: Any) -> list[Any]:
    return [observe(block.text) for block in blocks]


def observed(result: ToolResult | ToolResultMessage) -> dict[str, Any]:
    return {
        "content": content(result.content),
        "details": observe(result.details),
        "is_error": result.is_error,
    }


def expected(boundary: dict[str, Any]) -> dict[str, Any]:
    """A boundary's expectation computed from the scenario TEXT (CE-L0206-D002-01 N4)."""
    out = {
        "content": [expect(block) for block in boundary["content"]],
        "details": expect(boundary["details"]),
    }
    if "is_error" in boundary:
        out["is_error"] = boundary["is_error"]
    return out


def _after_hook(hook: dict[str, Any], seen: dict[str, Any]) -> Any:
    mode = hook["mode"]

    def after(result: ToolResult) -> AfterToolCallOverride | None:
        seen["hook"] = observed(result)
        if mode == "observe":
            return None
        if mode == "same":
            return AfterToolCallOverride(content=result.content, details=result.details)
        if mode == "null":
            return AfterToolCallOverride(details=None)
        if mode == "throws":
            raise RuntimeError(decode(hook["message"]))
        replaced = tuple(TextBlock(text=decode(b)) for b in hook.get("content", ()))
        return AfterToolCallOverride(
            content=replaced if "content" in hook else None,
            details=decode(hook["details"]) if "details" in hook else None,
        )

    return after


def _fixture_tool(tool: dict[str, Any]) -> ToolDefinition:
    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> ToolResult:
        if "throws" in tool:
            raise RuntimeError(decode(tool["throws"]))
        returns = tool["returns"]
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="probe",
            content=tuple(TextBlock(text=decode(b)) for b in returns["content"]),
            details=decode(returns["details"]),
        )

    return ToolDefinition(
        name="probe",
        label="probe",
        description="probe",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


async def run_case(case: dict[str, Any], root: Path) -> dict[str, Any]:
    """One agent run through every boundary; returns each boundary's observation."""
    seen: dict[str, Any] = {}
    ctx = Context()
    for plugin in (session_plugin, llm_plugin, tools_plugin, agents_plugin, agent_loop_plugin):
        await ctx.plugin(plugin)
    if "edit" in case:
        (root / "f.txt").write_bytes(bytes.fromhex(case["edit"]["file_utf8_hex"]))
        ctx.tools.register(create_edit_tool(LocalFileSystem(str(root))))
        call = ToolCallBlock(id="call-1", name="edit", arguments=decode(case["edit"]["arguments"]))
    else:
        ctx.tools.register(_fixture_tool(case["tool"]))
        call = ToolCallBlock(id="call-1", name="probe", arguments={})
    ctx.llm.register(
        MockAdapter(
            [
                ScriptedResponse((call,), StopReason.TOOL_USE),
                ScriptedResponse((TextBlock(text="done"),), StopReason.STOP),
            ]
        )
    )
    if case["hook"]["mode"] != "none":
        register_after_tool_call_hook(ctx, _after_hook(case["hook"], seen))

    def on_end(call_id: str, name: str, result: ToolResult) -> None:
        seen.setdefault("execution_end", []).append(observed(result))

    def on_lifecycle(instance: Any, event: Any) -> None:
        if isinstance(event, MessageEnd) and isinstance(event.message, ToolResultMessage):
            seen.setdefault("message", []).append(observed(event.message))

    ctx.events.on(TOOLS_EXECUTION_END, on_end)
    handle = ctx.agents.create(
        "probe", AgentDefinition(name="probe", model=ModelId("mock", "mock-1"), system="")
    )
    loop = ctx.agent_loop.for_instance(handle.instance)
    ctx.events.on(AGENT_LIFECYCLE_EVENT, on_lifecycle)
    handle.instance.inbox.followup(UserMessage(content=(TextBlock(text="go"),), timestamp=1))
    await loop.run_until_idle()

    log = handle.instance.log
    seen["session"] = [
        {"content": content(m.content), "details": observe(m.details)}
        for m in derive_messages(log)
        if isinstance(m, ToolResultMessage)
    ]
    seen["replayed_execution_end"] = [
        observed(e.result) for e in project(log) if isinstance(e, ToolExecutionEnd)
    ]
    if "edit" in case:
        seen["file_utf8_hex"] = (root / "f.txt").read_bytes().hex()
    return seen


def check(case: dict[str, Any], seen: dict[str, Any]) -> None:
    want = case["expect"]
    hook = want["hook"]
    boundaries = [
        ("hook", expected(hook) if hook is not None else None),
        ("execution_end", [expected(want["execution_end"])]),
        ("message", [expected(want["message"])]),
        ("session", [expected(want["session"])]),
        # Layer 08 replay of the same event (R001): the scenario's own execution_end expectation.
        ("replayed_execution_end", [expected(want["execution_end"])]),
    ]
    if "file_utf8_hex" in want:
        boundaries.append(("file_utf8_hex", want["file_utf8_hex"]))
    for boundary, wanted in boundaries:
        assert seen.get(boundary) == wanted, (case["id"], boundary, seen.get(boundary))
