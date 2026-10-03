"""`L0206-D001-R007`, K1 final complete review 2 (Codex, docs #228 comment `5965316121`): argument
containers the FRAMEWORK produces (typed-model validation's rebuild, a `prepare_arguments` shim's
graph) are pipeline-owned, like the raw arguments a provider decoded. A before-hook that mutates one
through its own API sees ECMAScript order at once, as in pinned Pi, where every object is ordered
intrinsically. The Owner's Q1/Q2 interval covers only containers an observer introduced; none is
introduced here. Each witness drives the real `execute_call` pipeline and observes natively
(`list(child)`, no graph read)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from minion_agent.llm import ToolCallBlock
from minion_agent.llm.js_object import JsArray, JsObject, order_in_place
from minion_agent.runtime import Context
from minion_agent.tools import execute as execute_module
from minion_agent.tools.decisions import Block
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

ORDERED = ["1", "2", "b"]


class ListParams(BaseModel):
    a: list[Any]


class NestedParams(BaseModel):
    a: list[dict[str, Any]]


class DefaultedParams(BaseModel):
    a: list[Any] = Field(default_factory=list)


async def _run(
    *,
    parameters: Any,
    raw: dict[str, Any],
    mutate: Any,
    prepare: Any = None,
) -> dict[str, Any]:
    """Run one call whose before-hook applies `mutate(args)`, which returns the native child it
    built. The hook blocks unless the child's own keys are already in ECMAScript order."""
    seen: dict[str, Any] = {}
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="t",
            label="t",
            description="t",
            parameters=parameters,
            prepare_arguments=prepare,
            execute=lambda tool_call_id, args: "ok",
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(call: Any, definition: Any, args: Any, signal: Any, next_: Any) -> Any:
        child = mutate(args)
        seen["order"] = list(child)
        seen["args"] = args
        if seen["order"] != ORDERED:
            return Block(reason="order-based block")
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    result = await execute_call(
        ToolCallBlock(id="c", name="t", arguments=raw), registry=registry, ctx=ctx
    )
    seen["blocked"] = result.is_error
    return seen


def _append_child(args: Any) -> dict[str, Any]:
    parent = args["a"]  # obtained once; every later access is through this alias, natively
    child: dict[str, Any] = {"b": 1, "2": 2, "1": 3}
    parent.append(child)
    assert list.__getitem__(parent, -1) is child  # identity kept: the seam orders in place
    return child


def _append_into_nested(args: Any) -> dict[str, Any]:
    inner = args["a"][0]
    inner["b"] = 1
    inner["2"] = 2
    inner["1"] = 3
    return inner  # type: ignore[no-any-return]


async def _typed() -> dict[str, Any]:
    return await _run(parameters=ListParams, raw={"a": []}, mutate=_append_child)


async def _prepared() -> dict[str, Any]:
    return await _run(
        parameters={"type": "object"},
        raw={"a": []},
        mutate=_append_child,
        prepare=lambda args: {"a": []},
    )


# --- the two R007 observations -----------------------------------------------------------------


async def test_a_typed_model_array_carries_the_graph_seams() -> None:
    seen = await _typed()
    assert seen["order"] == ORDERED
    assert isinstance(seen["args"]["a"], JsArray)
    assert not seen["blocked"]


async def test_a_prepare_produced_array_carries_the_graph_seams() -> None:
    seen = await _prepared()
    assert seen["order"] == ORDERED
    assert isinstance(seen["args"]["a"], JsArray)
    assert not seen["blocked"]


# --- neighbors -----------------------------------------------------------------------------------


async def test_a_typed_model_nested_object_in_an_array_orders_on_assignment() -> None:
    seen = await _run(parameters=NestedParams, raw={"a": [{}]}, mutate=_append_into_nested)
    assert seen["order"] == ORDERED
    assert not seen["blocked"]


async def test_a_typed_model_default_filled_array_carries_the_graph_seams() -> None:
    seen = await _run(parameters=DefaultedParams, raw={}, mutate=_append_child)
    assert seen["order"] == ORDERED
    assert not seen["blocked"]


async def test_a_prepare_produced_nested_object_orders_on_assignment() -> None:
    def assign(args: Any) -> dict[str, Any]:
        target = args["o"]
        target["b"] = 1
        target["2"] = 2
        target["1"] = 3
        return target  # type: ignore[no-any-return]

    seen = await _run(
        parameters={"type": "object"},
        raw={},
        mutate=assign,
        prepare=lambda args: {"o": {}},
    )
    assert seen["order"] == ORDERED
    assert not seen["blocked"]


async def test_a_container_the_shim_places_twice_stays_one_container() -> None:
    shared: list[Any] = []

    def touch(args: Any) -> dict[str, Any]:
        assert args["x"] is args["y"]  # aliasing inside the prepared graph survives adoption
        return _append_child({"a": args["x"]})

    seen = await _run(
        parameters={"type": "object"},
        raw={},
        mutate=touch,
        prepare=lambda args: {"x": shared, "y": shared},
    )
    assert seen["order"] == ORDERED
    assert seen["args"]["y"][-1] is seen["args"]["x"][-1]


async def test_raw_arguments_without_a_shim_or_model_are_unchanged() -> None:
    seen = await _run(parameters={"type": "object"}, raw={"a": []}, mutate=_append_child)
    assert seen["order"] == ORDERED
    assert not seen["blocked"]


# --- controls: reinstate each pre-R007 seam; its witness must fail -------------------------------


def _native_rebuild(delivered: Any, given: Any) -> Any:
    """Pre-R007 `_in_input_order`: arrays rebuilt as plain lists."""
    if isinstance(delivered, dict):
        source = given if isinstance(given, dict) else {}
        ordered = JsObject()
        for key in source:
            if key in delivered:
                ordered[key] = _native_rebuild(delivered[key], source[key])
        for key, value in delivered.items():
            if key not in ordered:
                ordered[key] = _native_rebuild(value, None)
        return ordered
    if isinstance(delivered, list):
        source_list = given if isinstance(given, list) else []
        return [
            _native_rebuild(item, source_list[i] if i < len(source_list) else None)
            for i, item in enumerate(delivered)
        ]
    return delivered


async def test_control_a_native_typed_rebuild_fails_the_typed_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execute_module, "_in_input_order", _native_rebuild)
    seen = await _typed()
    assert seen["order"] != ORDERED
    assert seen["blocked"]


async def test_control_ordering_the_shim_graph_without_adopting_fails_the_prepare_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execute_module, "adopt", order_in_place)  # pre-R007 `_prepare`
    seen = await _prepared()
    assert seen["order"] != ORDERED
    assert seen["blocked"]


# --- R4-C001: references that cross the adopted-native / retained-graph frontier -----------------


async def _crossing(retained_first: bool) -> dict[str, Any]:
    """A native child reachable through a retained graph container is ALSO returned by the shim
    through a new native parent. Pinned Pi keeps one object (structuredClone keeps aliasing), so a
    hook's change through one path shows through the other."""
    native: list[Any] = []
    retained = JsObject()
    retained["a"] = native  # attached through a seam: stays native (Owner Q2)
    seen: dict[str, Any] = {}
    registry = ToolRegistry()

    def prepare(args: Any) -> dict[str, Any]:
        old = dict.__getitem__(args, "old")
        same = {"inner": dict.__getitem__(old, "a")}  # a NEW native parent of the retained child
        return {"old": old, "same": same} if retained_first else {"same": same, "old": old}

    registry.register(
        ToolDefinition(
            name="t",
            label="t",
            description="t",
            parameters={"type": "object"},
            prepare_arguments=prepare,
            execute=lambda tool_call_id, args: "ok",
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(call: Any, definition: Any, args: Any, signal: Any, next_: Any) -> Any:
        via_old = dict.__getitem__(dict.__getitem__(args, "old"), "a")
        via_same = dict.__getitem__(dict.__getitem__(args, "same"), "inner")
        seen["equal"] = via_old is via_same
        seen["kept"] = via_old is native
        list.append(via_same, "changed")
        seen["through_old"] = list(via_old)
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    call = ToolCallBlock(id="c", name="t", arguments={})
    call.arguments["old"] = retained
    await execute_call(call, registry=registry, ctx=ctx)
    return seen


@pytest.mark.parametrize("retained_first", [True, False])
async def test_a_reference_crossing_the_frontier_stays_one_object(retained_first: bool) -> None:
    seen = await _crossing(retained_first)
    assert seen == {"equal": True, "kept": True, "through_old": ["changed"]}


def _single_pass_adopt(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Revision-4 `adopt`: memoized over the native frontier, blind to retained-graph contents."""
    memo = {} if memo is None else memo
    if isinstance(value, (JsObject, JsArray)) or not isinstance(value, (dict, list)):
        return value
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        js_object = JsObject()
        memo[id(value)] = js_object
        for key, item in dict.items(value):
            dict.__setitem__(js_object, key, _single_pass_adopt(item, memo))
        return order_in_place(js_object)
    array = JsArray()
    memo[id(value)] = array
    list.extend(array, [_single_pass_adopt(item, memo) for item in value])
    return array


async def test_control_a_frontier_blind_adopt_splits_the_crossing_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execute_module, "adopt", _single_pass_adopt)
    seen = await _crossing(retained_first=False)
    assert seen["equal"] is False
    assert seen["through_old"] == []
