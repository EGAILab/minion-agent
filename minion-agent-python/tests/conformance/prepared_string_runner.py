"""Runner for `prepared_string` scenarios
(`conformance/schema/prepared-string-scenario.schema.json`, L0506-D002 / `TOOL-041`).

Runs the delta's certification gate (`gate: L0506-D002`): custom cases only. Thin by design: one
tool call per case through the REAL Layer 06 `execute_call` (resolve, `prepare_arguments`,
validation, the `tools/pre-execute` hook, `execute`). A pre-execute listener records the prepared
value at each observed pointer (and object keys at each `observe_keys` pointer); a custom tool's
`execute` records the same. The runner decodes `prepare_set` UTF-16 code units into the strings a
tool's own shim returns (fixture input) and renders observed strings back to code units
(observation); it never prepares, validates or converts anything on the pipeline's behalf.

Python's representation of a JavaScript string (`spec/tools.md`, TOOL-041 string domain): a `str`
in which a valid surrogate pair is one astral code point and every unpaired surrogate code unit is
a surrogate code point. `utf-16-le` with `surrogatepass` maps exactly between the two.
"""

from __future__ import annotations

import copy
import struct
from typing import Any

from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools.decisions import Proceed
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult


def string(units: list[int]) -> str:
    """UTF-16 code units as the Python string a binding carries (pairs combined, lone kept)."""
    return struct.pack(f"<{len(units)}H", *units).decode("utf-16-le", "surrogatepass")


def units(value: Any) -> Any:
    """An observed value's UTF-16 code units (a non-string renders as a never-matching marker)."""
    if not isinstance(value, str):
        return {"non_string": repr(value)}
    data = value.encode("utf-16-le", "surrogatepass")
    return list(struct.unpack(f"<{len(data) // 2}H", data))


PAIR = string([0xD83D, 0xDE00])
LONE_HIGH = string([0xD800])
SCHEMAS: dict[str, dict[str, Any]] = {
    "open": {"type": "object", "properties": {}},
    "string": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "min-length-2": {"type": "object", "properties": {"text": {"type": "string", "minLength": 2}}},
    "max-length-1": {"type": "object", "properties": {"text": {"type": "string", "maxLength": 1}}},
    "pattern-one-char": {
        "type": "object",
        "properties": {"text": {"type": "string", "pattern": "^.$"}},
    },
    "pattern-two-chars": {
        "type": "object",
        "properties": {"text": {"type": "string", "pattern": "^..$"}},
    },
    "const-pair": {"type": "object", "properties": {"text": {"const": PAIR}}},
    "enum-lone": {"type": "object", "properties": {"text": {"enum": [LONE_HIGH]}}},
}


def decode(value: Any) -> Any:
    """A `prepare_set` value as the runtime value a tool's own shim returns."""
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if "utf16" in value:
            return string(value["utf16"])
        if "$key" in value:
            return {string(value["$key"]): decode(value["value"])}
        return {key: decode(item) for key, item in value.items()}
    return value


def at(document: Any, pointer: str) -> Any:
    node = document
    for part in pointer.split("/")[1:]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def observe(arguments: Any, values: list[str], keys: list[str]) -> dict[str, Any]:
    return {
        "values": {p: units(at(arguments, p)) for p in values},
        "keys": {p: [units(k) for k in at(arguments, p)] for p in keys},
    }


def preflight(case: dict[str, Any]) -> None:
    """The schema's language-neutral PREFLIGHT; a violation fails the document."""
    expect = case["expect"]
    if expect["outcome"] != "prepared":
        return
    ident = case["id"]
    assert set(expect["observed"]) == set(case["observe"]), f"{ident}: observed pointers"
    assert set(expect.get("observed_keys", {})) == set(case.get("observe_keys", [])), ident
    assert ("execute_observed" in expect) == ("hook_replace_set" in case), ident
    if "hook_replace_set" in case:
        assert set(expect["execute_observed"]) == set(case["execute_observe"]), ident
    assert set(expect.get("execute_observed_keys", {})) == set(
        case.get("execute_observe_keys", [])
    ), ident


def _set(arguments: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    return {**arguments, **{pointer[1:]: decode(value) for pointer, value in values.items()}}


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one gate-`L0506-D002` (custom) case; returns `{outcome, text, hook, execute,
    raw_unchanged}`. A gate-`WP-13.2` real-`edit` case is not this runner's."""
    assert case["tool"] == "custom", f"{case['id']}: not a delta-gate case"
    preflight(case)
    raw = copy.deepcopy(case["arguments"])
    seen: dict[str, Any] = {}
    replace = case.get("hook_replace_set")
    execute_values = case.get("execute_observe", case["observe"])
    execute_keys = case.get("execute_observe_keys", [] if replace else case.get("observe_keys", []))

    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> ToolResult:
        seen["execute"] = observe(arguments, execute_values, execute_keys)
        return ToolResult(
            tool_call_id=tool_call_id, content=(TextBlock(text="ok"),), tool_name="probe"
        )

    definition = ToolDefinition(
        name="probe",
        label="probe",
        description="L0506-D002 probe",
        parameters=SCHEMAS[case["schema"]],
        execute=execute,
        prepare_arguments=lambda arguments: _set(arguments, case["prepare_set"]),
    )
    registry = ToolRegistry()
    registry.register(definition)
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(call: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        seen["hook"] = observe(arguments, case["observe"], case.get("observe_keys", []))
        if replace is not None:
            return Proceed(arguments=_set(arguments, replace))
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
        "hook": seen.get("hook"),
        "execute": seen.get("execute"),
        "raw_unchanged": call.arguments == raw and case["arguments"] == raw,
    }


def check(case: dict[str, Any], run: dict[str, Any]) -> None:
    expect = case["expect"]
    ident = case["id"]
    assert run["raw_unchanged"], f"{ident}: raw ToolCall arguments changed"
    assert run["outcome"] == expect["outcome"], (ident, run["text"])
    if expect["outcome"] != "prepared":
        assert run["hook"] is None and run["execute"] is None, ident
        return
    hook = {"values": expect["observed"], "keys": expect.get("observed_keys", {})}
    if "hook_replace_set" in case:
        execute = {
            "values": expect["execute_observed"],
            "keys": expect.get("execute_observed_keys", {}),
        }
    else:
        execute = hook
    assert run["hook"] == hook, (ident, "hook", run["hook"])
    assert run["execute"] == execute, (ident, "execute", run["execute"])
