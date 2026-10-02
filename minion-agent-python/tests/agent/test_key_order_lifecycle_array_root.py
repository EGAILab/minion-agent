"""`L0206-D001-R004`, K1 targeted closure 1 (Codex, docs #228 comment `5949233413`): the raw
arguments domain admits an ARRAY root, and agent lifecycle listeners (`ToolExecutionStart` /
`ToolExecutionUpdate` on `agent/lifecycle-event`) must each receive it in ECMAScript order --
including a native child the previous listener attached and then mutated through its retained
reference (control has returned to the framework, so the Owner's bounded interval cannot apply).
Observations are NATIVE (`json.dumps` of the retained child), so no graph read repairs them."""

from __future__ import annotations

import json
from typing import Any

import pytest

from minion_agent.agent import events as agent_events_module
from minion_agent.agent.events import AGENT_LIFECYCLE_EVENT, declare_agent_events
from minion_agent.agent.projection import ToolExecutionStart, ToolExecutionUpdate
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.llm.js_object import order_in_place
from minion_agent.runtime import Context
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolPartialResult, ToolResult

ORDERED = '{"1": 3, "2": 2, "b": 1}'


def _listeners(ctx: Context, seen: list[str]) -> None:
    retained: dict[str, Any] = {}

    def first(_instance: Any, event: Any) -> None:
        child: dict[str, Any] = {"b": 1}
        event.arguments[0]["o"] = child  # through the adopted graph's own seam
        child["2"] = 2  # then natively, through the retained reference
        child["1"] = 3
        retained["child"] = child

    def second(_instance: Any, _event: Any) -> None:
        seen.append(json.dumps(retained["child"]))

    ctx.events.on(AGENT_LIFECYCLE_EVENT, first)
    ctx.events.on(AGENT_LIFECYCLE_EVENT, second)


async def _start_through_the_pipeline() -> list[str]:
    seen: list[str] = []
    ctx = Context()
    declare_tools_events(ctx.events)
    declare_agent_events(ctx.events)
    _listeners(ctx, seen)

    async def on_start(call_id: str, name: str, arguments: Any) -> None:
        # As the agent-loop driver does: the live start delivery dispatches the lifecycle event.
        event = ToolExecutionStart(tool_call_id=call_id, tool_name=name, arguments=arguments)
        await ctx.events.serial(AGENT_LIFECYCLE_EVENT, None, event)

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="t",
            label="t",
            description="t",
            parameters={"type": "object", "properties": {}},
            execute=lambda tool_call_id, args: ToolResult(
                tool_call_id=tool_call_id, content=(TextBlock(text="ok"),), tool_name="t"
            ),
        )
    )
    call = ToolCallBlock(id="c", name="t", arguments=[{}])  # type: ignore[arg-type]
    await execute_call(call, registry=registry, ctx=ctx, on_execution_start=on_start)
    return seen


async def _update_at_the_lifecycle_seam() -> list[str]:
    seen: list[str] = []
    ctx = Context()
    declare_agent_events(ctx.events)
    _listeners(ctx, seen)
    call = ToolCallBlock(id="c", name="t", arguments=[{}])  # type: ignore[arg-type]
    event = ToolExecutionUpdate(
        tool_call_id="c",
        tool_name="t",
        arguments=call.arguments,
        partial_result=ToolPartialResult(content=(TextBlock(text="p"),), details={}),
    )
    await ctx.events.serial(AGENT_LIFECYCLE_EVENT, None, event)
    return seen


async def test_an_array_root_reaches_the_next_start_listener_ordered() -> None:
    assert await _start_through_the_pipeline() == [ORDERED]


async def test_an_array_root_reaches_the_next_update_listener_ordered() -> None:
    assert await _update_at_the_lifecycle_seam() == [ORDERED]


async def test_control_the_dict_only_gate_fails_both_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reinstating the closure-1 candidate's dict-only gate must fail both witnesses."""

    def dict_only(args: tuple[object, ...]) -> None:
        arguments = getattr(args[-1], "arguments", None) if args else None
        if isinstance(arguments, dict):
            order_in_place(arguments)

    monkeypatch.setattr(agent_events_module, "_order_event_arguments", dict_only)
    assert await _start_through_the_pipeline() != [ORDERED]
    assert await _update_at_the_lifecycle_seam() != [ORDERED]


def test_a_primitive_root_is_left_untouched() -> None:
    agent_events_module._order_event_arguments((None, ToolExecutionStart("c", "t", 5)))  # type: ignore[arg-type]
    agent_events_module._order_event_arguments(())
