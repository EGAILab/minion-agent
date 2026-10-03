"""`L0206-D001` (K1), convergence `CE-L0206-D001-01`, Owner Q1 decision (`minion-agent#100` comment
`5947071963`): objects reachable through the tool-argument graph are in ECMAScript own-property
order at every Minion-mediated attachment and observation; an attached object is the SAME object
(never a copy). Evidence matrix A-L (decision section 12) through real hooks of the real
`execute_call` pipeline, the approved bounded divergence witness (section 13), and the negative
controls (section 14), each of which must make at least one matrix witness fail."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from minion_agent.agent import events as agent_events_module
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.llm import content as content_module
from minion_agent.llm import js_object as js_module
from minion_agent.runtime import Context, DispatchMode, EventBus, EventModeError
from minion_agent.session import derive as derive_module
from minion_agent.tools import batch as batch_module
from minion_agent.tools import events as tools_events_module
from minion_agent.tools import execute as execute_module
from minion_agent.tools.decisions import Block, Proceed
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

Hook = Callable[..., Any]


async def _pipeline(
    hooks: list[Hook],
    *,
    arguments: dict[str, Any] | None = None,
    on_start_event: Any = None,
    on_update_event: Any = None,
) -> dict[str, Any]:
    """One real call: `hooks` are tools/pre-execute listeners; execute records its arguments."""
    seen: dict[str, Any] = {}

    def execute(tool_call_id: str, args: dict[str, Any], update: Any) -> ToolResult:
        seen["execute"] = args
        seen["execute_keys"] = {
            k: list(dict.__iter__(v)) for k, v in dict.items(args) if isinstance(v, dict)
        }
        seen["execute_top"] = list(dict.__iter__(args))
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
    ctx = Context()
    declare_tools_events(ctx.events)
    if on_start_event:
        ctx.events.on(TOOLS_EXECUTION_START, on_start_event)
    if on_update_event:
        ctx.events.on(TOOLS_UPDATE, on_update_event)
    for hook in hooks:
        ctx.events.on(TOOLS_PRE_EXECUTE, hook)

    async def start(call_id: str, name: str, args: dict[str, Any]) -> None:
        seen["start"] = list(dict.__iter__(args))

    async def update(call_id: str, name: str, args: dict[str, Any], partial: Any) -> None:
        seen["update"] = list(dict.__iter__(args))

    result = await execute_call(
        ToolCallBlock(id="c", name="t", arguments=arguments if arguments is not None else {"b": 1}),
        registry=registry,
        ctx=ctx,
        on_execution_start=start,
        on_execution_update=update,
    )
    seen["is_error"] = result.is_error
    first = result.content[0] if result.content else None
    seen["text"] = first.text if isinstance(first, TextBlock) else ""
    return seen


def _keys(value: dict[str, Any]) -> list[str]:
    """A dict's own enumeration WITHOUT any Minion mediation (the native order it holds)."""
    return list(dict.__iter__(value))


# --- the evidence matrix (decision section 12) --------------------------------------------------


async def witness_a_attach_then_read_through_args() -> None:
    """A: the C001 witness -- read-back through args decides block/proceed."""
    record: dict[str, Any] = {}

    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        child = {"b": 1, "2": 2, "1": 3}
        args["o"] = child
        record["keys"] = list(args["o"].keys())
        if record["keys"][0] != "1":
            return Block(reason="order-based block")
        return await next_()

    seen = await _pipeline([hook])
    assert record["keys"] == ["1", "2", "b"]
    assert not seen["is_error"], seen["text"]


async def witness_b_identity_preserved() -> None:
    record: dict[str, Any] = {}

    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        child = {"b": 1, "1": 1}
        args["o"] = child
        record["same"] = args["o"] is child
        record["child_keys"] = _keys(child)  # ordered at attachment, in place
        child["c"] = 2  # a later alias mutation stays visible downstream
        return await next_()

    seen = await _pipeline([hook])
    assert record["same"] and record["child_keys"] == ["1", "b"]
    assert seen["execute_keys"]["o"] == ["1", "b", "c"]


async def witness_c_mutated_alias_read_through_args() -> None:
    record: dict[str, Any] = {}

    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        child: dict[str, Any] = {"b": 1}
        args["o"] = child
        child["2"] = 2  # direct native mutation through the retained alias
        child["1"] = 3
        record["read_back"] = list(args["o"])  # the next graph-mediated read repairs it
        return await next_()

    await _pipeline([hook])
    assert record["read_back"] == ["1", "2", "b"]


async def _attach_mutate_no_read(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
    child: dict[str, Any] = {"b": 1}
    args["o"] = child
    child["2"] = 2
    child["1"] = 3
    return await next_()


async def witness_d_mutated_alias_next_listener() -> None:
    record: dict[str, Any] = {}

    async def second(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        record["next"] = _keys(dict.__getitem__(args, "o"))
        return await next_()

    await _pipeline([_attach_mutate_no_read, second])
    assert record["next"] == ["1", "2", "b"]


async def witness_e_mutated_alias_execute() -> None:
    seen = await _pipeline([_attach_mutate_no_read])
    assert seen["execute_keys"]["o"] == ["1", "2", "b"]


async def witness_f_nested_attachment() -> None:
    record: dict[str, Any] = {}

    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        args["o"] = {"z": 1, "p": {"b": 1, "1": 1}}
        record["nested_native"] = _keys(dict.__getitem__(dict.__getitem__(args, "o"), "p"))
        record["nested"] = list(args["o"]["p"])
        return await next_()

    await _pipeline([hook])
    assert record["nested_native"] == ["1", "b"] and record["nested"] == ["1", "b"]


def _array_hook(record: dict[str, Any], operate: Callable[[Any, dict[str, Any]], None]) -> Hook:
    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        element = {"b": 1, "2": 2, "1": 3}
        operate(dict.__getitem__(args, "a"), element)
        record["native"] = _keys(element)  # ordered by the seam before it returned
        record["same"] = any(item is element for item in list.__iter__(dict.__getitem__(args, "a")))
        return await next_()

    return hook


async def _array_witness(operate: Callable[[Any, dict[str, Any]], None]) -> None:
    record: dict[str, Any] = {}
    await _pipeline([_array_hook(record, operate)], arguments={"a": [{"x": 0}]})
    assert record["native"] == ["1", "2", "b"]
    assert record["same"]


async def witness_g_array_append() -> None:
    await _array_witness(lambda array, element: array.append(element))


async def witness_h_array_insert() -> None:
    await _array_witness(lambda array, element: array.insert(0, element))


async def witness_i_array_replacement() -> None:
    def replace(array: Any, element: dict[str, Any]) -> None:
        array[0] = element

    await _array_witness(replace)


async def witness_j_array_extend() -> None:
    await _array_witness(lambda array, element: array.extend([element]))


async def witness_k_hook_replacement() -> None:
    record: dict[str, Any] = {}

    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        replacement = {"b": 1, "1": 1, "o": {"z": 1, "0": 0}}
        record["replacement"] = replacement
        return Proceed(arguments=replacement)

    seen = await _pipeline([hook])
    assert seen["execute"] is record["replacement"]
    assert seen["execute_top"] == ["1", "b", "o"]
    assert seen["execute_keys"]["o"] == ["0", "z"]


def _mutate_raw(_id: str, _name: str, args: dict[str, Any], *_rest: Any) -> None:
    args["2"] = 2
    args["1"] = 3


async def witness_l_raw_start_delivery() -> None:
    seen = await _pipeline([], on_start_event=_mutate_raw)
    assert seen["start"] == ["1", "2", "b"]


async def witness_l_raw_update_delivery() -> None:
    seen = await _pipeline([], on_update_event=_mutate_raw)
    assert seen["update"] == ["1", "2", "b"]


MATRIX: dict[str, Callable[[], Any]] = {
    "A-attach-then-read-through-args": witness_a_attach_then_read_through_args,
    "B-identity-preserved": witness_b_identity_preserved,
    "C-mutated-alias-read-through-args": witness_c_mutated_alias_read_through_args,
    "D-mutated-alias-next-listener": witness_d_mutated_alias_next_listener,
    "E-mutated-alias-execute": witness_e_mutated_alias_execute,
    "F-nested-attachment": witness_f_nested_attachment,
    "G-array-append": witness_g_array_append,
    "H-array-insert": witness_h_array_insert,
    "I-array-replacement": witness_i_array_replacement,
    "J-array-extend": witness_j_array_extend,
    "K-hook-replacement": witness_k_hook_replacement,
    "L-raw-start-delivery": witness_l_raw_start_delivery,
    "L-raw-update-delivery": witness_l_raw_update_delivery,
}


@pytest.mark.parametrize("name", sorted(MATRIX))
async def test_matrix(name: str) -> None:
    await MATRIX[name]()


# --- the approved bounded Python divergence (decision sections 7, 8, 13) ------------------------


async def test_approved_divergence_retained_native_alias_enumerated_directly() -> None:
    """DOCUMENTARY. Pi: `1, 2, b` everywhere. Python: a plain dict already attached, mutated
    directly through the hook's retained alias and enumerated directly through that same alias
    before any further graph operation, shows native insertion order -- the approved
    INTENTIONAL_BOUNDED_DIVERGENCE (Python binding only). The next graph-mediated read repairs the
    SAME object in place, so the divergence does not leak past it (section 9)."""
    record: dict[str, Any] = {}

    async def hook(c: Any, d: Any, args: Any, signal: Any, next_: Any) -> Any:
        child: dict[str, Any] = {"b": 1}
        args["o"] = child
        child["2"] = 2
        child["1"] = 3
        record["alias_direct"] = list(child.keys())
        record["through_args"] = list(args["o"])
        record["alias_after"] = list(child.keys())
        return await next_()

    await _pipeline([hook])
    assert record["alias_direct"] == ["b", "2", "1"]  # the approved gap, exactly bounded
    assert record["through_args"] == ["1", "2", "b"]
    assert record["alias_after"] == ["1", "2", "b"]


# --- EventBus.before_each (the additive Layer-04 seam) ------------------------------------------


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


def test_an_event_without_before_each_dispatches_unchanged() -> None:
    bus = EventBus()
    bus.declare("e", DispatchMode.EMIT)
    seen: list[list[str]] = []
    bus.on("e", lambda arguments: seen.append(list(arguments)))
    bus.emit("e", {"b": 1, "1": 1})
    assert seen == [["b", "1"]]


def test_an_undeclared_event_cannot_take_a_before_each() -> None:
    with pytest.raises(EventModeError):
        EventBus().before_each("undeclared", lambda args: None)


def test_construction_adopts_arrays_and_objects() -> None:
    call = ToolCallBlock(id="c", name="t", arguments={"a": [{"b": 1, "0": 0}], "o": {"z": 1}})
    assert isinstance(call.arguments, js_module.JsObject)
    assert isinstance(dict.__getitem__(call.arguments, "a"), js_module.JsArray)
    assert content_module.adopt is js_module.adopt


# --- negative controls (decision section 14) ----------------------------------------------------


def _plain_setitem(self: Any, key: str, value: Any) -> None:
    """JsObject's own index ordering kept; attachment ordering removed."""
    if key in self or not js_module.is_array_index(key):
        dict.__setitem__(self, key, value)
        return
    position = int(key)
    later = [k for k in dict.__iter__(self) if not js_module.is_array_index(k) or int(k) > position]
    moved = [(k, dict.pop(self, k)) for k in later]
    dict.__setitem__(self, key, value)
    for k, v in moved:
        dict.__setitem__(self, k, v)


def _strip_graph_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """No attachment or read normalization on the graph's own types."""
    monkeypatch.setattr(js_module.JsObject, "__setitem__", _plain_setitem)
    monkeypatch.setattr(js_module.JsObject, "__getitem__", dict.__getitem__)
    monkeypatch.setattr(js_module.JsObject, "get", dict.get)
    monkeypatch.setattr(js_module.JsObject, "values", dict.values)
    monkeypatch.setattr(js_module.JsObject, "items", dict.items)
    for name in ("append", "insert", "extend", "__setitem__", "__getitem__", "__iter__"):
        monkeypatch.setattr(js_module.JsArray, name, getattr(list, name))
    monkeypatch.setattr(js_module.JsArray, "__iadd__", list.__iadd__)


def _identity(value: Any) -> Any:
    return value


def _keep_only(monkeypatch: pytest.MonkeyPatch, kept: str) -> None:
    """Strip every ordering mechanism except one family of sites."""
    _strip_graph_seams(monkeypatch)
    if kept != "listener":
        monkeypatch.setattr(tools_events_module, "_order_arguments", lambda args: None)
        monkeypatch.setattr(agent_events_module, "_order_event_arguments", lambda args: None)
    if kept != "execute":
        monkeypatch.setattr(execute_module, "order_in_place", _identity)
    if kept != "serialization":
        monkeypatch.setattr(derive_module, "order_raw", _identity)
    for module in (execute_module, batch_module):
        monkeypatch.setattr(module, "order_raw", _identity)


def _only_before_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    _keep_only(monkeypatch, "execute")


def _only_before_next_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    _keep_only(monkeypatch, "listener")


def _only_during_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    _keep_only(monkeypatch, "serialization")


def _copy_on_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    original = js_module.JsObject.__setitem__

    def setitem(self: Any, key: str, value: Any) -> None:
        original(self, key, copy.deepcopy(value))

    monkeypatch.setattr(js_module.JsObject, "__setitem__", setitem)


def _identity_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attachment converts: an ordered JsObject COPY replaces the attached dict."""
    original = js_module.JsObject.__setitem__

    def setitem(self: Any, key: str, value: Any) -> None:
        converted = js_module.adopt(copy.copy(value)) if isinstance(value, dict) else value
        original(self, key, converted)

    monkeypatch.setattr(js_module.JsObject, "__setitem__", setitem)


def _top_level_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def top_only(value: Any) -> Any:
        if isinstance(value, dict):
            ordered = {
                k: dict.__getitem__(value, k)
                for k in js_module.es_order(list(dict.__iter__(value)))
            }
            dict.clear(value)
            dict.update(value, ordered)
        return value

    for module in (js_module, execute_module, tools_events_module, agent_events_module):
        monkeypatch.setattr(module, "order_in_place", top_only)


def _array_insertion_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("append", "insert", "extend", "__setitem__"):
        monkeypatch.setattr(js_module.JsArray, name, getattr(list, name))


def _same_hook_read_unordered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(js_module.JsObject, "__setitem__", _plain_setitem)
    monkeypatch.setattr(js_module.JsObject, "__getitem__", dict.__getitem__)


CONTROLS: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "copy-on-assignment": _copy_on_assignment,
    "normalize-only-before-execute": _only_before_execute,
    "normalize-only-before-next-listener": _only_before_next_listener,
    "normalize-only-during-serialization": _only_during_serialization,
    "top-level-not-nested": _top_level_only,
    "array-insertion-bypass": _array_insertion_bypass,
    "same-hook-args-read-insertion-order": _same_hook_read_unordered,
    "identity-loss": _identity_loss,
}


@pytest.mark.parametrize("name", sorted(CONTROLS))
async def test_control_fails_the_matrix(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    CONTROLS[name](monkeypatch)
    failed = []
    for witness, run in MATRIX.items():
        try:
            await run()
        except Exception:  # an assertion mismatch or a broken pipeline both fail the witness
            failed.append(witness)
    assert failed, f"{name} survived the observer-chain matrix"
