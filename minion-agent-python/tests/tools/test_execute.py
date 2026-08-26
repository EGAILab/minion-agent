"""One call through the pipeline. Every path produces exactly one result."""

from typing import Any

from pydantic import BaseModel

from minion_agent.llm import ToolCallBlock, text_of
from minion_agent.runtime import Context
from minion_agent.tools.decisions import Block, Proceed
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult


class EchoParams(BaseModel):
    value: str


def _call(name: str = "echo", **arguments: Any) -> ToolCallBlock:
    return ToolCallBlock(id="t1", name=name, arguments=dict(arguments))


def _ctx() -> Context:
    ctx = Context()
    declare_tools_events(ctx.events)
    return ctx


def _registry(*definitions: ToolDefinition) -> ToolRegistry:
    registry = ToolRegistry()
    for definition in definitions:
        registry.register(definition)
    return registry


def _echo(**overrides: Any) -> ToolDefinition:
    defaults: dict[str, Any] = {
        "name": "echo",
        "description": "repeat",
        "parameters": EchoParams,
        "execute": lambda tool_call_id, args: str(args["value"]),
        "label": "Echo",
    }
    return ToolDefinition(**{**defaults, **overrides})


async def test_a_tool_runs_and_returns_its_output() -> None:
    result = await execute_call(_call(value="pong"), registry=_registry(_echo()), ctx=_ctx())

    assert text_of(result.to_message()) == "pong"
    assert not result.is_error


async def test_execute_receives_the_real_tool_call_id() -> None:
    """Pinned Pi's `execute(toolCallId, params, ...)`: the pipeline's own call id, not one the
    tool invents (`TOOL-017`)."""
    seen: list[str] = []
    definition = _echo(execute=lambda tool_call_id, args: seen.append(tool_call_id) or "ok")

    await execute_call(
        ToolCallBlock(id="unique-id-1", name="echo", arguments={"value": "x"}),
        registry=_registry(definition),
        ctx=_ctx(),
    )

    assert seen == ["unique-id-1"]


async def test_a_string_return_is_normalized_into_a_result() -> None:
    """Tools may return a bare string; the pipeline needs a full result."""
    result = await execute_call(_call(value="x"), registry=_registry(_echo()), ctx=_ctx())

    assert isinstance(result, ToolResult)
    assert result.tool_call_id == "t1"


async def test_a_tool_may_return_a_full_result() -> None:
    definition = _echo(
        execute=lambda tool_call_id, args: ToolResult(
            tool_call_id="t1", content=(), tool_name="echo", terminate=True, details={"k": 1}
        )
    )

    result = await execute_call(_call(value="x"), registry=_registry(definition), ctx=_ctx())

    assert result.terminate
    assert result.details == {"k": 1}


async def test_an_async_tool_is_awaited() -> None:
    async def slow(tool_call_id: str, args: dict[str, Any]) -> str:
        return "async"

    result = await execute_call(
        _call(value="x"), registry=_registry(_echo(execute=slow)), ctx=_ctx()
    )

    assert text_of(result.to_message()) == "async"


async def test_an_unknown_tool_produces_an_error_result() -> None:
    result = await execute_call(_call("missing"), registry=_registry(), ctx=_ctx())

    assert result.is_error
    assert "missing" in text_of(result.to_message())
    assert result.tool_call_id == "t1"


async def test_invalid_arguments_produce_an_error_result_the_model_can_act_on() -> None:
    """The model chose the arguments, so it is the one that must be told."""
    result = await execute_call(_call(wrong="field"), registry=_registry(_echo()), ctx=_ctx())

    assert result.is_error
    assert "value" in text_of(result.to_message())


async def test_defaults_are_filled_in_before_the_tool_runs() -> None:
    class Params(BaseModel):
        value: str = "default"

    seen: list[dict[str, Any]] = []
    definition = _echo(
        parameters=Params, execute=lambda tool_call_id, args: seen.append(args) or "ok"
    )

    await execute_call(_call(), registry=_registry(definition), ctx=_ctx())

    assert seen == [{"value": "default"}]


async def test_a_raising_tool_produces_an_error_result() -> None:
    def broken(tool_call_id: str, args: dict[str, Any]) -> str:
        raise RuntimeError("disk on fire")

    result = await execute_call(
        _call(value="x"), registry=_registry(_echo(execute=broken)), ctx=_ctx()
    )

    assert result.is_error
    assert "disk on fire" in text_of(result.to_message())


async def test_a_listener_may_block_the_call() -> None:
    ctx = _ctx()
    ran: list[str] = []

    async def veto(call: Any, definition: Any, arguments: Any, next_: Any) -> Block:
        return Block(reason="not permitted")

    ctx.events.on(TOOLS_PRE_EXECUTE, veto)
    definition = _echo(execute=lambda tool_call_id, args: ran.append("ran") or "ok")

    result = await execute_call(_call(value="x"), registry=_registry(definition), ctx=ctx)

    assert ran == []
    assert result.is_error
    assert "not permitted" in text_of(result.to_message())


async def test_a_blocked_call_may_also_terminate_the_turn() -> None:
    ctx = _ctx()
    ran: list[str] = []

    async def veto(call: Any, definition: Any, arguments: Any, next_: Any) -> Block:
        return Block(reason="stop now", terminate=True)

    ctx.events.on(TOOLS_PRE_EXECUTE, veto)
    definition = _echo(execute=lambda tool_call_id, args: ran.append("ran") or "ok")

    result = await execute_call(_call(value="x"), registry=_registry(definition), ctx=ctx)

    assert ran == []
    assert result.terminate


async def test_a_listener_may_narrow_the_arguments() -> None:
    """How sandboxing pins a path without the tool knowing a policy exists."""
    ctx = _ctx()
    seen: list[dict[str, Any]] = []

    async def pin(call: Any, definition: Any, arguments: Any, next_: Any) -> Proceed:
        return Proceed(arguments={"value": "pinned"})

    ctx.events.on(TOOLS_PRE_EXECUTE, pin)
    definition = _echo(execute=lambda tool_call_id, args: seen.append(args) or "ok")

    await execute_call(_call(value="original"), registry=_registry(definition), ctx=ctx)

    assert seen == [{"value": "pinned"}]


async def test_an_abstaining_listener_leaves_the_call_alone() -> None:
    ctx = _ctx()
    calls: list[str] = []

    async def abstain(call: Any, definition: Any, arguments: Any, next_: Any) -> Any:
        calls.append("abstained")
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, abstain)

    result = await execute_call(_call(value="through"), registry=_registry(_echo()), ctx=ctx)

    assert calls == ["abstained"]
    assert text_of(result.to_message()) == "through"


async def test_prepare_arguments_runs_before_validation() -> None:
    """Pinned Pi's `AgentTool.prepareArguments` always runs before schema validation
    (`TOOL-017`): raw arguments that would fail validation on their own must still succeed once
    `prepare_arguments` repairs them first."""
    definition = _echo(prepare_arguments=lambda raw: {**raw, "value": "repaired"})

    result = await execute_call(_call(), registry=_registry(definition), ctx=_ctx())

    assert not result.is_error
    assert text_of(result.to_message()) == "repaired"


async def test_prepare_arguments_does_not_mutate_the_original_call() -> None:
    """Nonmutation discipline (design spec section 6, matching XFORM): a shim that mutates its
    input in place must not corrupt the source `ToolCallBlock.arguments`."""

    def mutate_in_place(raw: dict[str, Any]) -> dict[str, Any]:
        raw["value"] = "mutated"
        return raw

    call = _call(value="original")
    definition = _echo(prepare_arguments=mutate_in_place)

    await execute_call(call, registry=_registry(definition), ctx=_ctx())

    assert call.arguments == {"value": "original"}


async def test_a_prepare_arguments_failure_produces_an_error_result() -> None:
    """`TOOL-017`: a `prepare_arguments` exception becomes an error result, same as any other
    pre-execute-stage failure -- `execute` is skipped entirely."""
    ran: list[str] = []

    def broken_prepare(raw: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("cannot repair")

    definition = _echo(
        prepare_arguments=broken_prepare,
        execute=lambda tool_call_id, args: ran.append("ran") or "ok",
    )

    result = await execute_call(_call(value="x"), registry=_registry(definition), ctx=_ctx())

    assert ran == []
    assert result.is_error
    assert "cannot repair" in text_of(result.to_message())


async def test_a_raising_before_hook_listener_produces_an_error_result() -> None:
    """Pinned Pi wraps `prepareToolCallArguments` + `validateToolArguments` +
    `config.beforeToolCall` in one try/catch: a hook that throws (rather than returning a
    structured `Block`) collapses to the same generic error result (`TOOL-017`)."""
    ctx = _ctx()
    ran: list[str] = []

    async def exploding(call: Any, definition: Any, arguments: Any, next_: Any) -> Any:
        raise RuntimeError("hook exploded")

    ctx.events.on(TOOLS_PRE_EXECUTE, exploding)
    definition = _echo(execute=lambda tool_call_id, args: ran.append("ran") or "ok")

    result = await execute_call(_call(value="x"), registry=_registry(definition), ctx=ctx)

    assert ran == []
    assert result.is_error
    assert "hook exploded" in text_of(result.to_message())
