"""Runner for `raw_arguments` scenarios (`conformance/schema/raw-arguments-scenario.schema.json`,
L0206-D002 / `AI-003` raw `ToolCall.arguments` JavaScript value domain).

Thin by design: the case's value (fixture input, decoded from the value grammar) becomes ONE
`ToolCallBlock`; the runner then only OBSERVES it at each certified boundary -- the constructed
call, an `AssistantMessage` appended to a fresh `SessionLog` (`encode_message`) and replayed
(`decode_message`), the `tools/execution-start` payload, the `tools/pre-execute` listener,
`execute` of a tool without `prepare_arguments`, and -- the tool reporting one partial result --
the `tools/update` payload and the `on_execution_update` delivery (L0206-D002-R001). Strings render
to UTF-16 code units and numbers to tokens. It never decodes, normalizes or converts on the
pipeline's behalf.
"""

from __future__ import annotations

import math
import re
import struct
from typing import Any

from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.llm.messages import AssistantMessage, StopReason, Usage
from minion_agent.runtime import Context
from minion_agent.session.derive import decode_message, encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.tools.builtin._js import number_to_string
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import (
    TOOLS_EXECUTION_START,
    TOOLS_PRE_EXECUTE,
    TOOLS_UPDATE,
    declare_tools_events,
)
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolPartialResult, ToolResult

from .prepared_runtime_runner import render

# L0206-D002-R002: the raw number domain is JSON.parse's -- no NaN.
NAMED = {"+Infinity": math.inf, "-Infinity": -math.inf, "-0": -0.0}
FINITE_LITERAL = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")


def string(code_units: list[int]) -> str:
    """UTF-16 code units as the Python string a binding carries (pairs combined, lone kept)."""
    return struct.pack(f"<{len(code_units)}H", *code_units).decode("utf-16-le", "surrogatepass")


def units(value: str) -> list[int]:
    data = value.encode("utf-16-le", "surrogatepass")
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def canonical_finite(token: str) -> bool:
    """The schema's numeric PREFLIGHT (L0206-D002-R002): a finite literal must denote a finite
    binary64 value and be that value's ECMAScript Number::toString -- so it names exactly one
    binary64 value and no binding rounds or widens it."""
    if FINITE_LITERAL.fullmatch(token) is None:
        return False
    value = float(token)
    return math.isfinite(value) and number_to_string(value) == token  # "-0" is named, never literal


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
    """A (preflighted) number token as the binding value: an integral literal is an `int`
    (Layer 02's JSON integer decoding, D001's representation rule) -- exactly a binary64 value by
    the preflight; every other number is a `float`."""
    if token in NAMED:
        return NAMED[token]
    if not any(mark in token for mark in ".eE"):
        return int(token)
    return float(token)


def decode(value: Any) -> Any:
    """A scenario value as the runtime value (fixture input)."""
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


def observe(value: Any) -> Any:
    """A runtime value in the grammar, objects as key->observation maps (compared as sets). An
    integer that is not exactly a binary64 value is outside the raw domain and renders as such,
    never through a lossy float conversion (L0206-D002-R002)."""
    if isinstance(value, str):
        return {"utf16": units(value)}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and (abs(value) > 2**1024 or int(float(value)) != value):
        return {"non_binary64_int": str(value)}
    if isinstance(value, int | float):
        return {"number": render(value)}
    if isinstance(value, list):
        return [observe(item) for item in value]
    if isinstance(value, dict):
        return {tuple(units(key)): observe(item) for key, item in value.items()}
    return {"non_json": repr(value)}


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """One call through every boundary; returns each boundary's observation (or its error)."""
    preflight(case["arguments"])
    call = ToolCallBlock(id="call-1", name="probe", arguments=decode(case["arguments"]))
    seen: dict[str, Any] = {"construction": observe(call.arguments)}
    try:
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
    except Exception as error:
        seen["replay"] = {"error": f"{type(error).__name__}: {error}"}

    async def execute(tool_call_id: str, arguments: dict[str, Any], update: Any) -> ToolResult:
        seen["execute"] = observe(arguments)
        update(ToolPartialResult(content=(TextBlock(text="partial"),), details={}))
        return ToolResult(
            tool_call_id=tool_call_id, content=(TextBlock(text="ok"),), tool_name="probe"
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="probe",
            label="probe",
            description="L0206-D002 probe (no prepare_arguments)",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(c: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        seen["hook"] = observe(arguments)
        return await next_()

    def on_start(call_id: str, name: str, arguments: Any, *rest: Any) -> None:
        seen["execution_start"] = observe(arguments)

    def on_update(call_id: str, name: str, arguments: Any, *rest: Any) -> None:
        seen.setdefault("update_event", []).append(observe(arguments))

    async def deliver_update(call_id: str, name: str, arguments: Any, partial: Any) -> None:
        seen.setdefault("update_delivery", []).append(observe(arguments))

    ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    ctx.events.on(TOOLS_EXECUTION_START, on_start)
    ctx.events.on(TOOLS_UPDATE, on_update)
    result = await execute_call(
        call, registry=registry, ctx=ctx, on_execution_update=deliver_update
    )
    first = result.content[0]
    seen["result"] = (result.is_error, first.text if isinstance(first, TextBlock) else "")
    return seen


BOUNDARIES = ("construction", "replay", "execution_start", "hook", "execute")
UPDATE_BOUNDARIES = ("update_event", "update_delivery")  # exactly one partial result each


def check(case: dict[str, Any], seen: dict[str, Any]) -> None:
    want = observe(decode(case["arguments"]))
    assert seen["result"] == (False, "ok"), (case["id"], seen["result"])
    for boundary in BOUNDARIES:
        assert seen.get(boundary) == want, (case["id"], boundary, seen.get(boundary))
    for boundary in UPDATE_BOUNDARIES:
        assert seen.get(boundary) == [want], (case["id"], boundary, seen.get(boundary))
