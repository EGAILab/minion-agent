"""`L0206-D001` (K1), Owner Q2 decision (`minion-agent#100` comment `5948712829`): containers a hook
introduces after construction stay NATIVE (identity, R002); a mutation through their own native
API is not Minion-mediated; any later graph-mediated read or framework boundary (next listener,
execute, events, serialization) is exact. Matrix M-V (decision section 13); the direct-alias
observations are the approved bounded Python divergence (documentary); controls (section 14)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from minion_agent.agent import events as agent_events_module
from minion_agent.llm import ToolCallBlock
from minion_agent.llm import js_object as js_module
from minion_agent.llm.messages import AssistantMessage, StopReason, Usage
from minion_agent.session import derive as derive_module
from minion_agent.tools import events as tools_events_module
from minion_agent.tools import execute as execute_module

from .test_key_order_observer_chain import (
    MATRIX,
    Hook,
    _identity,
    _keys,
    _only_before_execute,
    _only_before_next_listener,
    _pipeline,
)

CHILD = {"b": 1, "2": 2, "1": 3}
ORDERED = ["1", "2", "b"]


def _native_parent_hook(
    record: dict[str, Any], parent: Any, key: str, attach: Callable[[Any, Any], None]
) -> Hook:
    """Attach a hook-introduced native `parent`, then attach a child THROUGH its native API."""

    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        args[key] = parent
        record["parent_same"] = args[key] is parent
        child = dict(CHILD)
        attach(args[key], child)  # native API: list.append / dict.__setitem__ ...
        record["alias_direct"] = list(child)  # M/R: the approved bounded divergence
        record["child"] = child
        return await next_()

    return hook


def _assert_native_case(record: dict[str, Any]) -> None:
    assert record["parent_same"] and record["identity_kept"]
    assert record["alias_direct"] == ["b", "2", "1"]  # approved divergence (documentary)
    assert record["graph_read"] == ORDERED
    assert record["next_listener"] == ORDERED
    assert record["execute"] == ORDERED


async def _native_list_case(
    attach: Callable[[Any, Any], None], read: Callable[[Any], Any]
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    items: list[Any] = [{"x": 0}]
    hook = _native_parent_hook(record, items, "a", attach)

    async def graph_read(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        record["graph_read"] = list(read(args))  # N/O/P/Q: exact
        return await next_()

    async def next_listener(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        found = [x for x in list.__iter__(dict.__getitem__(args, "a")) if x is record["child"]]
        record["next_listener"] = _keys(found[0])  # T: exact
        return await next_()

    seen = await _pipeline([hook, graph_read, next_listener])
    array = dict.__getitem__(seen["execute"], "a")
    record["identity_kept"] = array is items and any(
        item is record["child"] for item in list.__iter__(array)
    )
    record["execute"] = _keys(record["child"])  # U: exact (ordered in place before execute)
    return record


async def witness_m_n_t_u_native_list_append() -> None:
    _assert_native_case(
        await _native_list_case(lambda a, c: a.append(c), lambda args: args["a"][1])
    )


async def witness_o_native_list_insert() -> None:
    _assert_native_case(
        await _native_list_case(lambda a, c: a.insert(0, c), lambda args: args["a"][0])
    )


async def witness_p_native_list_replacement() -> None:
    def replace(array: Any, child: Any) -> None:
        array[0] = child

    _assert_native_case(await _native_list_case(replace, lambda args: args["a"][0]))


async def witness_q_native_list_extend() -> None:
    _assert_native_case(
        await _native_list_case(lambda a, c: a.extend([c]), lambda args: args["a"][1])
    )


async def witness_r_s_t_u_native_dict_nested() -> None:
    record: dict[str, Any] = {}
    outer: dict[str, Any] = {}

    def assign(parent: Any, child: Any) -> None:
        parent["n"] = child

    hook = _native_parent_hook(record, outer, "o", assign)

    async def graph_read(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        record["graph_read"] = list(args["o"]["n"])  # S: exact
        return await next_()

    async def next_listener(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        record["next_listener"] = _keys(dict.__getitem__(dict.__getitem__(args, "o"), "n"))
        return await next_()

    seen = await _pipeline([hook, graph_read, next_listener])
    record["identity_kept"] = dict.__getitem__(seen["execute"], "o") is outer
    record["execute"] = _keys(record["child"])
    _assert_native_case(record)


async def witness_deep_native_descendants() -> None:
    """Deeper descendants attached through native containers are repaired by the graph read."""
    record: dict[str, Any] = {}

    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        outer: dict[str, Any] = {}
        args["o"] = outer
        args["o"]["n"] = {"p": [{"q": dict(CHILD)}]}
        record["deep"] = list(args["o"]["n"]["p"][0]["q"])
        return await next_()

    await _pipeline([hook])
    assert record["deep"] == ORDERED


async def witness_v_event_and_serialization_boundary() -> None:
    """V: a start-event listener introduces native containers on the RAW object and attaches
    through them; the live start delivery is exact, and so is a later session encoding."""
    seen_raw: dict[str, Any] = {}

    def listener(_id: str, _name: str, args: dict[str, Any], *_rest: Any) -> None:
        items: list[Any] = []
        args["a"] = items
        args["a"].append(dict(CHILD))
        outer: dict[str, Any] = {}
        args["o"] = outer
        args["o"]["n"] = dict(CHILD)
        seen_raw["args"] = args

    seen = await _pipeline([], on_start_event=listener)
    raw = seen_raw["args"]
    assert seen["start"] == ["b", "a", "o"]  # no index keys: insertion order
    assert _keys(list.__getitem__(dict.__getitem__(raw, "a"), 0)) == ORDERED
    assert _keys(dict.__getitem__(dict.__getitem__(raw, "o"), "n")) == ORDERED

    call = ToolCallBlock(id="c", name="t", arguments={"b": 1})
    items2: list[Any] = []
    call.arguments["a"] = items2
    items2.append(dict(CHILD))  # native, after construction, never read through the graph
    message = AssistantMessage(
        content=(call,),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(),
        model="m",
        provider="p",
        timestamp=0,
    )
    encoded = derive_module.encode_message(message)["content"][0]["arguments"]
    assert _keys(list.__getitem__(dict.__getitem__(encoded, "a"), 0)) == ORDERED


MATRIX_Q2: dict[str, Callable[[], Any]] = {
    "M-N-T-U-native-list-append": witness_m_n_t_u_native_list_append,
    "O-native-list-insert": witness_o_native_list_insert,
    "P-native-list-replacement": witness_p_native_list_replacement,
    "Q-native-list-extend": witness_q_native_list_extend,
    "R-S-T-U-native-dict-nested": witness_r_s_t_u_native_dict_nested,
    "deep-native-descendants": witness_deep_native_descendants,
    "V-event-and-serialization": witness_v_event_and_serialization_boundary,
}


@pytest.mark.parametrize("name", sorted(MATRIX_Q2))
async def test_matrix_q2(name: str) -> None:
    await MATRIX_Q2[name]()


# --- negative controls (decision section 14) ----------------------------------------------------


def _reorder(item: dict[str, Any]) -> None:
    ordered = {k: dict.__getitem__(item, k) for k in js_module.es_order(list(dict.__iter__(item)))}
    dict.clear(item)
    dict.update(item, ordered)


def _copy_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    original = js_module.JsObject.__setitem__

    def setitem(self: Any, key: str, value: Any) -> None:
        original(self, key, copy.copy(value) if isinstance(value, dict | list) else value)

    monkeypatch.setattr(js_module.JsObject, "__setitem__", setitem)


def _wrapper_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    original = js_module.JsObject.__getitem__

    def getitem(self: Any, key: str) -> Any:
        value = original(self, key)
        return js_module.adopt(copy.copy(value)) if isinstance(value, dict | list) else value

    monkeypatch.setattr(js_module.JsObject, "__getitem__", getitem)


def _shallow_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph reads order only the returned value itself, not its native descendants."""

    def getitem(self: Any, key: str) -> Any:
        value = dict.__getitem__(self, key)
        if isinstance(value, dict):
            _reorder(value)
        return value

    monkeypatch.setattr(js_module.JsObject, "__getitem__", getitem)


def _no_execute_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disorder leaks from a hook into execute: no ordering before execute or between listeners."""
    monkeypatch.setattr(execute_module, "order_in_place", _identity)
    monkeypatch.setattr(tools_events_module, "_order_arguments", lambda args: None)


def _install_order(monkeypatch: pytest.MonkeyPatch, order: Callable[[Any], Any]) -> None:
    for module in (js_module, execute_module, tools_events_module, agent_events_module):
        monkeypatch.setattr(module, "order_in_place", order)
    monkeypatch.setattr(derive_module, "order_raw", order)


def _dicts_not_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_lists(value: Any) -> Any:
        seen: set[int] = set()
        pending = [value]
        while pending:
            item = pending.pop()
            if not isinstance(item, dict) or id(item) in seen:
                continue
            seen.add(id(item))
            _reorder(item)
            pending.extend(v for v in dict.values(item) if isinstance(v, dict))
        return value

    _install_order(monkeypatch, no_lists)


def _direct_children_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def one_level(value: Any) -> Any:
        if isinstance(value, dict):
            _reorder(value)
            for child in dict.values(value):
                if isinstance(child, dict):
                    _reorder(child)
        return value

    _install_order(monkeypatch, one_level)


CONTROLS_Q2: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "copy-hook-introduced-parent": _copy_parent,
    "identity-replacing-wrappers": _wrapper_reads,
    "normalize-only-at-execute": _only_before_execute,
    "normalize-only-between-listeners": _only_before_next_listener,
    "no-recursive-normalization-on-graph-read": _shallow_read,
    "disorder-leaks-into-execute": _no_execute_boundary,
    "dicts-but-not-lists": _dicts_not_lists,
    "direct-children-not-deeper-descendants": _direct_children_only,
}


@pytest.mark.parametrize("name", sorted(CONTROLS_Q2))
async def test_control_q2_fails_the_matrix(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    CONTROLS_Q2[name](monkeypatch)
    failed = []
    for witness, run in {**MATRIX, **MATRIX_Q2}.items():
        try:
            await run()
        except Exception:  # an assertion mismatch or a broken pipeline both fail the witness
            failed.append(witness)
    assert failed, f"{name} survived the observer-chain matrix"
