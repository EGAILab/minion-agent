"""`L0506-D005` (`TOOL-003`; minion-agent#190, provenance #129) binding witnesses beside the
canonical `arg-isolation` corpus: identity of the ONE validated graph across listeners and
`execute` (Owner F), the pydantic path's isolation (AUDITED -- NO CHANGE), and the clone itself."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

from minion_agent.llm import ToolCallBlock
from minion_agent.llm.js_object import JsArray, JsObject, structured_clone
from minion_agent.runtime import Context
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry


async def _run(parameters: Any, raw: dict[str, Any], listeners: int = 2) -> dict[str, Any]:
    seen: dict[str, Any] = {"listener_args": []}

    def execute(tool_call_id: str, arguments: Any) -> str:
        seen["execute_args"] = arguments
        return "ok"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="t", label="t", description="t", parameters=parameters, execute=execute)
    )
    ctx = Context()
    declare_tools_events(ctx.events)
    for _ in range(listeners):

        async def listener(c: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
            seen["listener_args"].append(arguments)
            arguments["o"]["seen"] = len(seen["listener_args"])
            return await next_()

        ctx.events.on(TOOLS_PRE_EXECUTE, listener)
    call = ToolCallBlock(id="c", name="t", arguments=raw)
    await execute_call(call, registry=registry, ctx=ctx)
    seen["raw"] = call.arguments
    return seen


async def test_every_listener_and_execute_receive_the_one_validated_graph() -> None:
    seen = await _run({"type": "object"}, {"o": {"z": 1}})
    first, second = seen["listener_args"]
    assert first is second is seen["execute_args"]
    assert seen["execute_args"]["o"] == {"z": 1, "seen": 2}
    assert seen["raw"] == {"o": {"z": 1}}
    assert seen["execute_args"]["o"] is not seen["raw"]["o"]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


async def test_the_pydantic_path_is_already_isolated_from_the_raw_arguments() -> None:
    # AUDITED -- NO CHANGE: `model_dump` rebuilds every container.
    seen = await _run(_Open, {"o": {"z": 1}})
    assert seen["execute_args"]["o"] == {"z": 1, "seen": 2}
    assert seen["raw"] == {"o": {"z": 1}}


def test_structured_clone_copies_every_container_once_and_keeps_order() -> None:
    shared = {"b": 1, "2": 2}
    source: dict[str, Any] = {"p": shared, "q": [shared, shared]}
    source["self"] = source
    clone = structured_clone(source)
    assert isinstance(clone, JsObject) and isinstance(clone["q"], JsArray)
    assert clone["p"] is clone["q"][0] is clone["q"][1]
    assert clone["self"] is clone
    assert clone["p"] is not shared and clone is not source
    assert list(dict.keys(clone["p"])) == ["2", "b"]  # ECMAScript order


def test_structured_clone_carries_values_unchanged() -> None:
    values = [-0.0, math.inf, -math.inf, "\ud800x", 10**300, None, True]
    clone = structured_clone({"v": values, "nan": math.nan})
    assert math.copysign(1.0, clone["v"][0]) < 0
    assert clone["v"][1:] == values[1:]
    assert math.isnan(clone["nan"])


def test_structured_clone_of_a_non_container_is_the_value() -> None:
    marker = object()
    assert structured_clone(marker) is marker
    assert structured_clone("s") == "s"


def test_structured_clone_is_not_bounded_by_the_interpreter_stack() -> None:
    deep: Any = []
    for _ in range(100_000):
        deep = [deep]
    clone = structured_clone({"d": deep})
    depth, node = 0, clone["d"]
    while node:
        node = list.__getitem__(node, 0)
        depth += 1
    assert depth == 100_000
