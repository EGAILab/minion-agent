"""Runner for `tool_result_domain` scenarios
(`conformance/schema/tool-result-domain-scenario.schema.json`,
L0506-D003 / `AI-006`, `TOOL-005`, `TOOL-017`, `MINION-002`: the tool-result runtime value domain).

Thin by design: a case's `tool.returns` (decoded from the L0206-D002 value grammar, plus `NaN`)
becomes the ONE
`ToolResult` a registered tool's `execute` returns (or `tool.throws` the message it raises); the
case's `hook` becomes
one after-hook registered through `register_after_tool_call_hook`. The runner then only OBSERVES the
result at each
boundary the pipeline itself produces:

    hook            what the after-hook receives
    execution_end   the `tools/execution-end` payload (pinned Pi's `tool_execution_end`)
    message         `ToolResult.to_message()` of the final result (the agent loop's own projection)
    session         that message appended to a fresh `SessionLog` (`encode_message`) and replayed
    (`decode_message`)

Strings render to UTF-16 code units, numbers to tokens, objects to key -> observation maps (compared
as sets: K1
order is L0206-D001's). It never decodes, normalizes or converts on the pipeline's behalf. The
gate-WP-13.2 document
runs the REAL `edit` tool over the real `LocalFileSystem` instead of a fixture tool.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from minion_agent.execution import LocalFileSystem
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.session.derive import decode_message, encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.tools.builtin import create_edit_tool
from minion_agent.tools.decisions import AfterToolCallOverride
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_EXECUTION_END, declare_tools_events
from minion_agent.tools.execute import execute_call, register_after_tool_call_hook
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult

from .raw_arguments_runner import canonical_finite, expect, observe, string

# L0506-D003: the result domain is the raw token grammar plus NaN (a tool result is not JSON.parse
# output).
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


def observed(result: ToolResult) -> dict[str, Any]:
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
        return AfterToolCallOverride(
            content=tuple(TextBlock(text=decode(b)) for b in hook["content"])
            if "content" in hook
            else None,
            details=decode(hook["details"]) if "details" in hook else None,
        )

    return after


async def run_case(case: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """One call through every boundary; returns each boundary's observation."""
    seen: dict[str, Any] = {}
    registry = ToolRegistry()
    if "edit" in case:
        assert root is not None
        (root / "f.txt").write_bytes(bytes.fromhex(case["edit"]["file_utf8_hex"]))
        registry.register(create_edit_tool(LocalFileSystem(str(root))))
        call = ToolCallBlock(id="call-1", name="edit", arguments=decode(case["edit"]["arguments"]))
    else:
        tool = case["tool"]

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

        registry.register(
            ToolDefinition(
                name="probe",
                label="probe",
                description="probe",
                parameters={"type": "object", "properties": {}},
                execute=execute,
            )
        )
        call = ToolCallBlock(id="call-1", name="probe", arguments={})
    ctx = Context()
    declare_tools_events(ctx.events)
    if case["hook"]["mode"] != "none":
        register_after_tool_call_hook(ctx, _after_hook(case["hook"], seen))

    def on_end(call_id: str, name: str, result: ToolResult, *rest: Any) -> None:
        seen.setdefault("execution_end", []).append(observed(result))

    ctx.events.on(TOOLS_EXECUTION_END, on_end)
    result = await execute_call(call, registry=registry, ctx=ctx)
    message = result.to_message()
    seen["message"] = {
        "content": content(message.content),
        "details": observe(message.details),
        "is_error": message.is_error,
    }
    event = SessionLog(session_id="s").append(
        EventKind.TOOL_RESULT, {"message": encode_message(message)}
    )
    replayed = decode_message(event.data["message"])
    seen["session"] = {"content": content(replayed.content), "details": observe(replayed.details)}
    if "edit" in case:
        seen["file_utf8_hex"] = (root / "f.txt").read_bytes().hex()  # type: ignore[operator]
    return seen


def check(case: dict[str, Any], seen: dict[str, Any]) -> None:
    want = case["expect"]
    hook = want["hook"]
    assert seen.get("hook") == (expected(hook) if hook is not None else None), (
        case["id"],
        "hook",
        seen.get("hook"),
    )
    assert seen.get("execution_end") == [expected(want["execution_end"])], (
        case["id"],
        "execution_end",
    )
    assert seen.get("message") == expected(want["message"]), (
        case["id"],
        "message",
        seen.get("message"),
    )
    assert seen.get("session") == expected(want["session"]), (
        case["id"],
        "session",
        seen.get("session"),
    )
    if "file_utf8_hex" in want:
        assert seen.get("file_utf8_hex") == want["file_utf8_hex"], (
            case["id"],
            "file",
            seen.get("file_utf8_hex"),
        )
