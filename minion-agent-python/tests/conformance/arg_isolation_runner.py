"""Runner for `arg_isolation` scenarios (`conformance/schema/arg-isolation-scenario.schema.json`,
L0506-D005 / `TOOL-003`: raw vs validated tool-argument isolation, minion-agent#129).

Thin by design: ONE `ToolCallBlock` whose arguments are the case's `raw_text` as `JSON.parse` yields
it (fixture construction; `-0` stays a float, as `raw_arguments_runner` maps it), a tool with the
case's JSON-Schema parameters and named `prepare_arguments` shim, and one `tools/pre-execute`
listener per `hooks` program. Each listener records its arguments at entry, then runs its program IN
PLACE on the object it received; `execute` records its arguments and reports one partial result;
each `tools/update` payload is recorded; finally the raw arguments. Isolation is the pipeline's job:
the runner never copies, clones or orders anything on its behalf.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any

from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools.decisions import Block
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, TOOLS_UPDATE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolPartialResult, ToolResult

from .prepared_runtime_runner import render
from .raw_arguments_runner import is_binary64_int, number


def _units(value: str) -> list[int]:
    data = value.encode("utf-16-le", "surrogatepass")
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def observe(value: Any, stack: list[Any] | None = None) -> Any:
    """A runtime value in the observation grammar; a reference to an ancestor on the current path
    observes as `{cycle: levels up}`."""
    stack = [] if stack is None else stack
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and not is_binary64_int(value):
        # Strict and total (CE-L0206-D002-01 N3'): never rounded into a matching token.
        return {"non_binary64_int": hex(value)}
    if isinstance(value, int | float):
        # The certified prepared-runtime token: Number::toString, -0/NaN/+-Infinity named.
        return {"n": render(value)}
    if isinstance(value, str):
        return {"u": _units(value)}
    for depth, ancestor in enumerate(stack):
        if ancestor is value:
            return {"cycle": len(stack) - depth}
    if isinstance(value, list):
        return {"a": [observe(item, [*stack, value]) for item in list.__iter__(value)]}
    if isinstance(value, dict):
        return {"o": [[key, observe(item, [*stack, value])] for key, item in dict.items(value)]}
    return {"non_json": repr(value)}


def build(value: Any) -> Any:
    """An inserted value from the grammar, objects by plain assignment in insertion order."""
    if not isinstance(value, dict):
        return value
    if "n" in value:
        # The certified binary64 decoder (raw_arguments_runner.number); NaN: prepared domain.
        return math.nan if value["n"] == "NaN" else number(value["n"])
    if "u" in value:
        return struct.pack(f"<{len(value['u'])}H", *value["u"]).decode("utf-16-le", "surrogatepass")
    if "a" in value:
        return [build(item) for item in value["a"]]
    built: dict[str, Any] = {}
    for key, item in value["o"]:
        built[key] = build(item)
    return built


def _at(root: Any, path: list[Any]) -> Any:
    for key in path:
        root = root[key]
    return root


def _run(arguments: Any, program: list[dict[str, Any]]) -> None:
    for op in program:
        target = _at(arguments, op["path"])
        if op["op"] == "set":
            target[op["key"]] = build(op["value"])
        elif op["op"] == "push":
            target.append(build(op["value"]))
        else:
            del target[op["key"]]


def _alias(_raw: dict[str, Any]) -> dict[str, Any]:
    shared = {"k": 1}
    return {"p": shared, "q": shared}


def _cycle(_raw: dict[str, Any]) -> dict[str, Any]:
    node: dict[str, Any] = {"k": 1}
    node["self"] = node
    return node


def _non_finite(raw: dict[str, Any]) -> dict[str, Any]:
    return {**raw, "nan": math.nan, "inf": math.inf, "ninf": -math.inf, "nz": -0.0}


def _reuse_raw_child(raw: dict[str, Any]) -> dict[str, Any]:
    return {"o": raw["o"], "extra": 1}


SHIMS = {
    "alias": _alias,
    "cycle": _cycle,
    "non-finite": _non_finite,
    "reuse-raw-child": _reuse_raw_child,
}


def parse_raw(text: str) -> Any:
    """`JSON.parse`'s value (fixture construction): every number literal through binary64 with the
    certified decoder (`raw_arguments_runner.number`), so `-0` stays a float and an integral literal
    is its binary64 value's exact integer, never the spelled digits."""
    return json.loads(text, parse_int=number, parse_float=number)


async def run_case(scenario: dict[str, Any]) -> dict[str, Any]:
    raw = parse_raw(scenario["raw_text"])
    call = ToolCallBlock(id="c1", name="probe", arguments=raw)
    seen: dict[str, Any] = {"hook_entries": [], "facts": None, "execute": None}

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
            description="L0506-D005 isolation probe",
            parameters=scenario.get("schema", {"type": "object", "properties": {}}),
            execute=execute,
            prepare_arguments=SHIMS[scenario["prepare"]] if "prepare" in scenario else None,
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)
    programs = scenario["hooks"]
    for index, program in enumerate(programs):

        async def listener(
            c: Any,
            d: Any,
            arguments: Any,
            signal: Any,
            next_: Any,
            index: int = index,
            program: list[dict[str, Any]] = program,
        ) -> Any:
            if index == 0:
                seen["facts"] = [
                    _at(arguments, f["same"][0]) is _at(arguments, f["same"][1])
                    if "same" in f
                    else _at(arguments, f["distinct_from_raw"])
                    is not _at(call.arguments, f["distinct_from_raw"])
                    for f in scenario.get("facts", [])
                ]
            seen["hook_entries"].append(observe(arguments))
            _run(arguments, program)
            if scenario.get("block") and index == len(programs) - 1:
                return Block(reason="blocked")
            return await next_()

        ctx.events.on(TOOLS_PRE_EXECUTE, listener)
    updates: list[Any] = []
    ctx.events.on(
        TOOLS_UPDATE, lambda call_id, name, arguments, *rest: updates.append(observe(arguments))
    )
    await execute_call(call, registry=registry, ctx=ctx)
    return {
        "outcome": "executed" if seen["execute"] is not None else "immediate_error",
        "hook_entries": seen["hook_entries"],
        "facts": seen["facts"],
        "execute": seen["execute"],
        "updates": updates,
        "raw_after": observe(call.arguments),
    }
