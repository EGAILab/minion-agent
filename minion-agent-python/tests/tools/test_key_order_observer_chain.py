"""`L0206-D001` (K1), convergence `CE-L0206-D001-01`: a tool call's arguments are ONE shared,
mutable object, and every observer -- each event listener, each live callback, `execute` -- sees it
in ECMAScript key order, including after an EARLIER observer mutated it (Codex CHECKPOINT3's
`R004` refinement). Each witness mutates through a real observer and is observed by the next one,
through the real `execute_call` pipeline; each delivery seam also has a control that removes only
its own ordering and must make the witness fail."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.llm import js_object as js_module
from minion_agent.runtime import Context, DispatchMode, EventBus
from minion_agent.tools import events as tools_events_module
from minion_agent.tools import execute as execute_module
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

ORDERED = ["1", "2", "b"]


def _registry(seen: dict[str, Any]) -> ToolRegistry:
    def execute(tool_call_id: str, arguments: dict[str, Any], update: Any) -> ToolResult:
        seen["execute"] = list(arguments)
        update(ToolPartialResult(content=(TextBlock(text="partial"),), details={}))
        return ToolResult(tool_call_id=tool_call_id, content=(TextBlock(text="ok"),), tool_name="t")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="t",
            label="t",
            description="t",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )
    return registry


def _mutate(_id: str, _name: str, arguments: dict[str, Any], *_rest: Any) -> None:
    arguments["2"] = 2
    arguments["1"] = 3


async def _run(
    *, on_start_event: Any = None, on_update_event: Any = None, pre_execute: Any = None
) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    ctx = Context()
    declare_tools_events(ctx.events)
    if on_start_event:
        ctx.events.on(TOOLS_EXECUTION_START, on_start_event)
    if on_update_event:
        ctx.events.on(TOOLS_UPDATE, on_update_event)
    for listener in pre_execute or ():
        ctx.events.on(TOOLS_PRE_EXECUTE, listener)

    async def start(call_id: str, name: str, arguments: dict[str, Any]) -> None:
        seen["start"] = list(arguments)

    async def update(call_id: str, name: str, arguments: dict[str, Any], partial: Any) -> None:
        seen["update"] = list(arguments)

    result = await execute_call(
        ToolCallBlock(id="c", name="t", arguments={"b": 1}),
        registry=_registry(seen),
        ctx=ctx,
        on_execution_start=start,
        on_execution_update=update,
    )
    seen["is_error"] = result.is_error
    return seen


async def test_a_start_listener_mutation_reaches_the_start_delivery_ordered() -> None:
    seen = await _run(on_start_event=_mutate)
    assert seen["start"] == ORDERED and not seen["is_error"]


async def test_an_update_listener_mutation_reaches_the_update_delivery_ordered() -> None:
    seen = await _run(on_update_event=_mutate)
    assert seen["update"] == ORDERED and not seen["is_error"]


async def test_a_plain_child_assigned_by_one_listener_reaches_the_next_ordered() -> None:
    later: dict[str, Any] = {}

    async def first(c: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        child: dict[str, Any] = {"b": 1}
        arguments["o"] = child
        child["2"] = 2  # through the retained reference, after assignment
        return await next_()

    async def second(c: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        later["child"] = list(arguments["o"])
        return await next_()

    seen = await _run(pre_execute=[first, second])
    assert later["child"] == ["2", "b"]
    assert seen["execute"] == ["b", "o"]


async def test_an_observer_sees_its_own_mutation_of_a_pipeline_object_ordered_at_once() -> None:
    """Objects the pipeline owns are `JsObject`s from construction: a listener's own assignment is
    in the rule's order immediately, before any later boundary."""
    inside: dict[str, Any] = {}

    def listener(_id: str, _name: str, arguments: dict[str, Any], *_rest: Any) -> None:
        _mutate(_id, _name, arguments)
        inside["own"] = list(arguments)

    await _run(on_start_event=listener)
    assert inside["own"] == ORDERED


@pytest.mark.parametrize("mode", [DispatchMode.EMIT, DispatchMode.SERIAL, DispatchMode.PARALLEL])
async def test_before_each_runs_before_every_listener_in_every_mode(mode: DispatchMode) -> None:
    bus = EventBus()
    bus.declare("e", mode)
    seen: list[list[str]] = []
    bus.before_each("e", lambda args: js_module.order_in_place(args[0]))

    def mutate_then_record(arguments: dict[str, Any]) -> None:
        seen.append(list(arguments))
        arguments.update({"1": 1})

    def record(arguments: dict[str, Any]) -> None:
        seen.append(list(arguments))

    bus.on("e", mutate_then_record)
    bus.on("e", record)
    payload: dict[str, Any] = {"b": 1}
    if mode is DispatchMode.EMIT:
        bus.emit("e", payload)
    elif mode is DispatchMode.SERIAL:
        await bus.serial("e", payload)
    else:
        await bus.parallel("e", payload)
    assert seen == [["b"], ["1", "b"]]


def test_an_undeclared_event_cannot_take_a_before_each() -> None:
    from minion_agent.runtime import EventModeError

    with pytest.raises(EventModeError):
        EventBus().before_each("undeclared", lambda args: None)


# --- single-seam controls: removing one delivery seam's ordering must fail its witness ----------


def _no_start_delivery_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = execute_module.order_raw

    def skip_second(value: Any) -> Any:  # the emit's call is the first; the delivery's the second
        calls["n"] += 1
        return value if calls["n"] == 2 else real(value)

    monkeypatch.setattr(execute_module, "order_raw", skip_second)
    monkeypatch.setattr(js_module.JsObject, "__setitem__", dict.__setitem__)


def _no_update_delivery_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Order the raw object for the start emit, its delivery and the update emit, but not for the
    update delivery (the fourth `order_raw` of the call)."""
    calls = {"n": 0}
    real = execute_module.order_raw

    def skip_fourth(value: Any) -> Any:
        calls["n"] += 1
        return value if calls["n"] == 4 else real(value)

    monkeypatch.setattr(execute_module, "order_raw", skip_fourth)
    monkeypatch.setattr(js_module.JsObject, "__setitem__", dict.__setitem__)


def _no_before_each(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools_events_module, "_order_arguments", lambda args: None)
    monkeypatch.setattr(EventBus, "before_each", lambda self, name, prepare: None)


CONTROLS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], dict[str, Any], str]] = {
    "start-delivery-unordered": (_no_start_delivery_ordering, {"on_start_event": _mutate}, "start"),
    "update-delivery-unordered": (
        _no_update_delivery_ordering,
        {"on_update_event": _mutate},
        "update",
    ),
}


@pytest.mark.parametrize("name", sorted(CONTROLS))
async def test_control_a_delivery_seam_without_ordering_fails(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    install, kwargs, boundary = CONTROLS[name]
    install(monkeypatch)
    seen = await _run(**kwargs)
    assert seen[boundary] != ORDERED


async def test_control_without_before_each_the_next_listener_sees_an_unordered_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_before_each(monkeypatch)
    later: dict[str, Any] = {}

    async def first(c: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        child: dict[str, Any] = {"b": 1}
        arguments["o"] = child
        child["2"] = 2
        return await next_()

    async def second(c: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        later["child"] = list(arguments["o"])
        return await next_()

    ctx = Context()
    declare_tools_events(ctx.events)
    ctx.events.on(TOOLS_PRE_EXECUTE, first)
    ctx.events.on(TOOLS_PRE_EXECUTE, second)
    monkeypatch.setattr(execute_module, "order_in_place", lambda value: value)
    await execute_call(
        ToolCallBlock(id="c", name="t", arguments={"b": 1}),
        registry=_registry({}),
        ctx=ctx,
    )
    assert later["child"] == ["b", "2"]
