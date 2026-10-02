"""`L0206-D001` (K1): `JsObject` enumerates its keys in ECMAScript `OrdinaryOwnPropertyKeys`
order -- array-index keys ascending, then other keys in insertion order -- after every
construction and mutation. The pipeline-level evidence is `conformance/agent/key-order/`; these
tests pin the type."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from minion_agent.llm import ToolCallBlock
from minion_agent.llm.js_object import JsObject, is_array_index, order_in_place
from minion_agent.runtime import Context
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry


@pytest.mark.parametrize(
    ("key", "index"),
    [
        ("0", True),
        ("7", True),
        ("10", True),
        ("4294967294", True),
        ("4294967295", False),
        ("00", False),
        ("01", False),
        ("-0", False),
        ("-1", False),
        ("+1", False),
        ("1.0", False),
        (" 1", False),
        ("0x1", False),
        ("", False),
        (chr(0x661), False),  # ARABIC-INDIC DIGIT ONE: not a canonical decimal string
        ("10000000000", False),  # eleven digits: beyond 2**32 - 2 by length alone
        ("9" * 5000, False),  # R003: beyond the int-string conversion limit, still total
    ],
)
def test_array_index_is_the_canonical_decimal_of_0_to_2_pow_32_minus_2(
    key: str, index: bool
) -> None:
    assert is_array_index(key) is index


def test_construction_orders_indices_first_ascending_then_insertion() -> None:
    obj = JsObject([("b", 1), ("2", 2), ("a", 3), ("1", 4), ("01", 5)])
    assert list(obj) == ["1", "2", "b", "a", "01"]


def test_a_new_index_key_takes_its_ascending_position() -> None:
    obj = JsObject({"1": 1, "3": 3, "b": "b"})
    obj["2"] = 2
    obj["0"] = 0
    obj["c"] = "c"
    assert list(obj) == ["0", "1", "2", "3", "b", "c"]


def test_an_overwrite_keeps_the_position() -> None:
    obj = JsObject({"b": 1, "a": 2})
    obj["b"] = 9
    assert list(obj.items()) == [("b", 9), ("a", 2)]


def test_update_setdefault_and_the_merge_operators_keep_the_rule() -> None:
    obj = JsObject({"b": 1})
    obj.update({"1": 1}, z=2)
    obj.update([("0", 0)])
    assert obj.setdefault("a", 3) == 3
    assert obj.setdefault("a", 4) == 3
    obj |= {"5": 5}
    merged = obj | {"2": 2}
    assert list(obj) == ["0", "1", "5", "b", "z", "a"]
    assert isinstance(merged, JsObject)
    assert list(merged) == ["0", "1", "2", "5", "b", "z", "a"]
    copied = obj.copy()
    assert isinstance(copied, JsObject)
    assert list(copied) == list(obj)


def test_assigned_values_are_stored_as_given_and_ordered_in_place() -> None:
    """R002: an assigned object is the very object assigned (never a copy); `order_in_place`
    orders it, and everything nested, without replacing anything."""
    child: dict[str, Any] = {"y": 1, "1": 1}
    obj = JsObject({"o": {"z": 1, "2": 2}, "list": [{"b": 1, "0": 0}, 5]})
    obj["n"] = child
    assert obj["n"] is child
    child["0"] = 0
    inner, items = obj["o"], obj["list"]
    assert order_in_place(obj) is obj
    assert obj["n"] is child and obj["o"] is inner and obj["list"] is items
    assert list(child) == ["0", "1", "y"]
    assert list(obj["o"]) == ["2", "z"]
    assert list(obj["list"][0]) == ["0", "b"]


def test_json_encoding_emits_the_rule() -> None:
    assert json.dumps(JsObject({"b": 1, "1": 2})) == '{"1": 2, "b": 1}'


def test_order_in_place_visits_shared_and_cyclic_objects_once() -> None:
    shared: dict[str, Any] = {"b": 1, "1": 1}
    cyclic: dict[str, Any] = {"z": 1, "0": 0}
    cyclic["self"] = cyclic
    value = [shared, shared, cyclic, 5]
    assert order_in_place(value) is value
    assert list(shared) == ["1", "b"]
    assert list(cyclic) == ["0", "z", "self"]
    assert order_in_place(5) == 5


def test_a_long_decimal_key_is_an_ordinary_key_at_construction() -> None:
    """R003: construction is total over every string key."""
    key = "9" * 5000
    call = ToolCallBlock(id="c", name="t", arguments={"b": 1, key: 2, "1": 3})
    assert list(call.arguments) == ["1", "b", key]


def test_a_tool_call_carries_ordered_arguments() -> None:
    """CE-L0206-D001-01: construction ADOPTS the decoded value -- every object in it becomes a
    `JsObject`, so a later mutation of any object the pipeline owns keeps the rule at once."""
    given: dict[str, Any] = {"b": 1, "2": 2, "o": {"z": 1, "0": 0}, "l": [{"y": 1}]}
    call = ToolCallBlock(id="c", name="t", arguments=given)
    assert isinstance(call.arguments, JsObject)
    assert isinstance(call.arguments["o"], JsObject)
    assert isinstance(call.arguments["l"][0], JsObject)
    assert call.arguments["l"] is given["l"]  # lists keep identity
    call.arguments["o"]["1"] = 1
    call.arguments["0"] = 0
    assert list(call.arguments["o"]) == ["0", "1", "z"]
    assert list(call.arguments) == ["0", "2", "b", "o", "l"]


class _Typed(BaseModel):
    z: str
    a: str
    d: str = "default"
    inner: dict[str, int] = {}
    items: list[dict[str, int]] = []


async def test_a_typed_model_tool_receives_input_order_then_defaults() -> None:
    """K1-F1: the model's values in the INPUT's order (by the rule), defaulted keys after, in
    declared order -- never the model's declared order imposed on supplied keys."""
    seen: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="typed",
            label="typed",
            description="typed",
            parameters=_Typed,
            execute=lambda tool_call_id, args: seen.append(args) or "ok",
        )
    )
    ctx = Context()
    declare_tools_events(ctx.events)
    arguments = {"items": [{"b": 1, "0": 0}], "a": "x", "inner": {"q": 1, "1": 2}, "z": "y"}
    call = ToolCallBlock(id="c", name="typed", arguments=arguments)
    await execute_call(call, registry=registry, ctx=ctx)
    assert list(seen[0]) == ["items", "a", "inner", "z", "d"]
    assert list(seen[0]["inner"]) == ["1", "q"]
    assert list(seen[0]["items"][0]) == ["0", "b"]
