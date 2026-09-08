"""One call through the pipeline. Every path produces exactly one result."""

from typing import Any

from pydantic import BaseModel

from minion_agent.llm import ToolCallBlock, text_of
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.decisions import Block, Proceed
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_POST_EXECUTE, TOOLS_PRE_EXECUTE, declare_tools_events
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
    """`IR-L06-003`: pinned Pi's `prepareToolCall` text is exact --
    `` `Tool ${toolCall.name} not found` `` -- not a host-specific rewording."""
    result = await execute_call(_call("missing"), registry=_registry(), ctx=_ctx())

    assert result.is_error
    assert text_of(result.to_message()) == "Tool missing not found"
    assert result.tool_call_id == "t1"


async def test_generated_error_details_and_tool_supplied_details_are_distinct() -> None:
    """`CA-L06-007`: pinned Pi's error helper (`createErrorToolResult`) always supplies
    `details: {}` -- but a SUCCESSFUL result's `details` is whatever the tool itself returned.
    Layer 06 does not synthesize a `details` value for an undeclared successful result as a
    shared, Pi-parity rule; `{}` only appears there as an unrelated Python `ToolResult` default,
    never something pinned Pi requires. Proven through the real pipeline: an unknown-tool call (a
    generated error) carries `{}`, and a tool that explicitly returns its own `details` carries
    that value unchanged, not merged with or replaced by any host default."""
    unknown = await execute_call(_call("missing"), registry=_registry(), ctx=_ctx())
    assert unknown.is_error
    assert unknown.to_message().details == {}

    definition = _echo(
        execute=lambda tool_call_id, args: ToolResult(
            tool_call_id=tool_call_id, content=(), tool_name="echo", details={"source": "tool"}
        )
    )
    successful = await execute_call(_call(value="x"), registry=_registry(definition), ctx=_ctx())
    assert not successful.is_error
    assert successful.to_message().details == {"source": "tool"}


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
    """`L06-R002`: pinned Pi surfaces `error.message`, never a runtime type name -- a JS
    `Error("disk on fire")` becomes `disk on fire`, not `Error: disk on fire`. Asserted by exact
    equality, not substring containment: `"disk on fire" in text` would also pass under the old,
    incorrect `RuntimeError: disk on fire` output."""

    def broken(tool_call_id: str, args: dict[str, Any]) -> str:
        raise RuntimeError("disk on fire")

    result = await execute_call(
        _call(value="x"), registry=_registry(_echo(execute=broken)), ctx=_ctx()
    )

    assert result.is_error
    assert text_of(result.to_message()) == "disk on fire"


async def test_a_listener_may_block_the_call() -> None:
    ctx = _ctx()
    ran: list[str] = []

    async def veto(call: Any, definition: Any, arguments: Any, signal: Any, next_: Any) -> Block:
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

    async def veto(call: Any, definition: Any, arguments: Any, signal: Any, next_: Any) -> Block:
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

    async def pin(call: Any, definition: Any, arguments: Any, signal: Any, next_: Any) -> Proceed:
        return Proceed(arguments={"value": "pinned"})

    ctx.events.on(TOOLS_PRE_EXECUTE, pin)
    definition = _echo(execute=lambda tool_call_id, args: seen.append(args) or "ok")

    await execute_call(_call(value="original"), registry=_registry(definition), ctx=ctx)

    assert seen == [{"value": "pinned"}]


async def test_an_abstaining_listener_leaves_the_call_alone() -> None:
    ctx = _ctx()
    calls: list[str] = []

    async def abstain(call: Any, definition: Any, arguments: Any, signal: Any, next_: Any) -> Any:
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
    assert text_of(result.to_message()) == "cannot repair"  # L06-R002: no class-name prefix


async def test_a_raising_before_hook_listener_produces_an_error_result() -> None:
    """Pinned Pi wraps `prepareToolCallArguments` + `validateToolArguments` +
    `config.beforeToolCall` in one try/catch: a hook that throws (rather than returning a
    structured `Block`) collapses to the same generic error result (`TOOL-017`)."""
    ctx = _ctx()
    ran: list[str] = []

    async def exploding(call: Any, definition: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        raise RuntimeError("hook exploded")

    ctx.events.on(TOOLS_PRE_EXECUTE, exploding)
    definition = _echo(execute=lambda tool_call_id, args: ran.append("ran") or "ok")

    result = await execute_call(_call(value="x"), registry=_registry(definition), ctx=ctx)

    assert ran == []
    assert result.is_error
    assert text_of(result.to_message()) == "hook exploded"  # L06-R002: no class-name prefix


_RAW_SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
"""A plain, object-valued JSON Schema mapping -- the actual Layer-05 shared `ToolDefinition.
parameters` representation, not a Pydantic model (`L06-R001`)."""


async def test_a_raw_object_schema_accepts_valid_arguments() -> None:
    """`L06-R001`: pinned Pi's `validateToolArguments` validates every `Tool.parameters:
    TSchema`, with no exemption for a raw-object-schema representation -- valid arguments must
    reach `execute` exactly as for a pydantic-backed tool."""
    definition = _echo(
        parameters=_RAW_SCHEMA, execute=lambda tool_call_id, args: f"got {args['x']}"
    )

    result = await execute_call(_call(x="hello"), registry=_registry(definition), ctx=_ctx())

    assert not result.is_error
    assert text_of(result.to_message()) == "got hello"


async def test_a_raw_object_schema_rejects_invalid_arguments() -> None:
    """`L06-R001`: an earlier, uncertified revision let a raw JSON-Schema `dict` bypass
    validation entirely (a genuine `PI_PARITY_DEFECT`) -- arguments that violate the schema must
    now fail before `execute` runs, same as a pydantic-backed tool. `L06-R002`: the failure text
    is the validator's own clean message, never a Python exception class name."""
    ran: list[str] = []
    definition = _echo(
        parameters=_RAW_SCHEMA, execute=lambda tool_call_id, args: ran.append("ran") or "ok"
    )

    result = await execute_call(_call(x=123), registry=_registry(definition), ctx=_ctx())

    assert ran == []
    assert result.is_error
    message = text_of(result.to_message())
    assert message.startswith("invalid arguments: ")
    assert "ValidationError" not in message
    assert "jsonschema" not in message


async def test_prepare_arguments_repairs_raw_schema_arguments_before_validation() -> None:
    """`L06-R001`/`TOOL-018`: `prepare_arguments` runs before validation regardless of which
    schema representation the tool uses -- raw arguments that would fail the JSON Schema on
    their own must still succeed once `prepare_arguments` repairs them first."""
    definition = _echo(
        parameters=_RAW_SCHEMA,
        prepare_arguments=lambda raw: {**raw, "x": str(raw["x"])},
        execute=lambda tool_call_id, args: f"got {args['x']!r}",
    )

    result = await execute_call(_call(x=123), registry=_registry(definition), ctx=_ctx())

    assert not result.is_error
    assert text_of(result.to_message()) == "got '123'"


async def test_prepare_arguments_can_invalidate_previously_valid_raw_schema_arguments() -> None:
    """The inverse of the above: `prepare_arguments` running before validation means it can also
    break arguments that started out valid -- validation still sees only the prepared value."""
    ran: list[str] = []
    definition = _echo(
        parameters=_RAW_SCHEMA,
        prepare_arguments=lambda raw: {**raw, "x": 999},
        execute=lambda tool_call_id, args: ran.append("ran") or "ok",
    )

    result = await execute_call(_call(x="valid"), registry=_registry(definition), ctx=_ctx())

    assert ran == []
    assert result.is_error


# -- Layer 09, `L09-C002`: preflight abort/error priority ---------------------


async def test_unknown_tool_wins_over_an_aborted_signal() -> None:
    """Checked before the try block and before any signal read (`prepareToolCall`,
    `agent-loop.ts:607-613`) -- wins unconditionally, even already aborted."""
    controller = RunAbortController()
    controller.abort()

    result = await execute_call(
        _call(name="missing", value="x"), registry=_registry(), ctx=_ctx(), signal=controller.signal
    )

    assert result.is_error
    assert "not found" in text_of(result.to_message())


async def test_argument_validation_failure_wins_over_an_aborted_signal() -> None:
    """Prepare/validate runs before the before-hook, which is the only place abort is checked --
    a validation failure keeps its own message regardless of abort state."""
    controller = RunAbortController()
    controller.abort()
    definition = _echo(execute=lambda tool_call_id, args: "should not run")

    result = await execute_call(
        _call(missing_field="x"),
        registry=_registry(definition),
        ctx=_ctx(),
        signal=controller.signal,
    )

    assert result.is_error
    assert "Operation aborted" not in text_of(result.to_message())


async def test_a_throwing_before_hook_wins_over_an_aborted_signal() -> None:
    """A before-hook that THROWS is caught by the same outer catch prepare/validate exceptions
    use, before any abort check is reached -- its own message wins, not "Operation aborted"."""
    ctx = _ctx()
    controller = RunAbortController()
    controller.abort()

    async def exploding(
        call: Any, definition: Any, arguments: Any, signal: Any, next_: Any
    ) -> Block:
        raise RuntimeError("hook exploded")

    ctx.events.on(TOOLS_PRE_EXECUTE, exploding)
    definition = _echo(execute=lambda tool_call_id, args: "should not run")

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=ctx, signal=controller.signal
    )

    assert result.is_error
    assert "hook exploded" in text_of(result.to_message())


async def test_an_aborted_signal_wins_over_a_returned_block_decision() -> None:
    """The discriminating witness (`L09-C002`): a before-hook that RETURNS `Block(True)` -- not
    one that throws -- loses to an already-aborted signal. `block: False` would not distinguish
    this from the ordinary "proceed" path, since it would reach `execute()` regardless."""
    ctx = _ctx()
    controller = RunAbortController()
    controller.abort()
    ran: list[str] = []

    async def veto(call: Any, definition: Any, arguments: Any, signal: Any, next_: Any) -> Block:
        return Block(reason="not permitted", terminate=True)

    ctx.events.on(TOOLS_PRE_EXECUTE, veto)
    definition = _echo(execute=lambda tool_call_id, args: ran.append("ran") or "ok")

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=ctx, signal=controller.signal
    )

    assert ran == []
    assert result.is_error
    assert text_of(result.to_message()) == "Operation aborted"
    assert not result.terminate  # the hook's own terminate:True never took effect


async def test_no_before_hook_and_an_aborted_signal_produces_operation_aborted() -> None:
    """Pi's own SECOND abort check covers "no beforeToolCall configured" -- Minion's single
    checkpoint (right after the always-running `TOOLS_PRE_EXECUTE` waterfall, whether or not any
    listener is registered) covers the same case."""
    controller = RunAbortController()
    controller.abort()
    ran: list[str] = []
    definition = _echo(execute=lambda tool_call_id, args: ran.append("ran") or "ok")

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=_ctx(), signal=controller.signal
    )

    assert ran == []
    assert result.is_error
    assert text_of(result.to_message()) == "Operation aborted"


async def test_a_call_proceeds_normally_when_the_signal_is_not_aborted() -> None:
    """No consumer forces a stop merely because a signal object exists and is unset -- the
    all-consumers-ignore-abort case, at the smallest possible scale."""
    signal = RunAbortController().signal

    result = await execute_call(
        _call(value="pong"), registry=_registry(_echo()), ctx=_ctx(), signal=signal
    )

    assert not result.is_error
    assert text_of(result.to_message()) == "pong"


# -- Layer 09: `execute()`'s own cooperative signal parameter -----------------


async def test_a_four_parameter_tool_receives_the_active_signal() -> None:
    """Matches pinned Pi's own `execute(toolCallId, params, signal, onUpdate)` positional order
    -- a tool declaring `wants_signal=True` and a fourth parameter receives `signal`, `update`
    cooperatively; Minion never inspects or acts on it itself."""
    signal = RunAbortController().signal
    seen: list[Any] = []

    def execute(tool_call_id: str, args: Any, signal: Any, update: Any) -> str:
        seen.append(signal)
        return "ok"

    definition = _echo(execute=execute, wants_signal=True)

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=_ctx(), signal=signal
    )

    assert not result.is_error
    assert seen == [signal]


async def test_a_signal_only_tool_receives_no_update_slot() -> None:
    """`L09-R003`: a tool wanting cancellation but not live updates -- Pi's own signal-only
    `execute(toolCallId, params, signal)` -- is representable without an unused fourth
    parameter."""
    signal = RunAbortController().signal
    seen: list[Any] = []

    def execute(tool_call_id: str, args: Any, signal: Any) -> str:
        seen.append(signal)
        return "ok"

    definition = _echo(execute=execute, wants_signal=True)

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=_ctx(), signal=signal
    )

    assert not result.is_error
    assert seen == [signal]


async def test_a_three_parameter_tools_own_meaning_is_unchanged() -> None:
    """A pre-existing 3-parameter tool with `wants_signal` left at its default (`False`) still
    means `(tool_call_id, arguments, update)`, not `(tool_call_id, arguments, signal)` -- Layer
    09 does not reinterpret any existing tool's own established arity."""
    seen: list[Any] = []

    def execute(tool_call_id: str, args: Any, update: Any) -> str:
        seen.append(update)
        return "ok"

    definition = _echo(execute=execute)  # wants_signal defaults to False

    result = await execute_call(
        _call(value="x"),
        registry=_registry(definition),
        ctx=_ctx(),
        signal=RunAbortController().signal,
    )

    assert not result.is_error
    assert len(seen) == 1
    assert callable(seen[0])  # the update callback, not a RunSignal


async def test_execute_is_not_forcibly_interrupted_by_an_aborted_signal() -> None:
    """Pi never forcibly interrupts `execute()` -- a tool that ignores the signal runs to
    completion and its result is used normally."""
    controller = RunAbortController()

    def execute(tool_call_id: str, args: Any, signal: Any, update: Any) -> str:
        controller.abort()  # aborts mid-execution; this tool does not check its own `signal`
        return "finished anyway"

    definition = _echo(execute=execute, wants_signal=True)

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=_ctx(), signal=controller.signal
    )

    assert not result.is_error
    assert text_of(result.to_message()) == "finished anyway"


async def test_after_hook_runs_unconditionally_despite_an_aborted_signal() -> None:
    """`finalizeExecutedToolCall` has no abort short-circuit at all -- an already-executed call
    is always finalized, and its own after-hook always runs, regardless of abort state."""
    from minion_agent.tools.execute import register_after_tool_call_hook

    ctx = _ctx()
    controller = RunAbortController()
    ran: list[str] = []

    def after_hook(result: ToolResult) -> None:
        ran.append("after")

    register_after_tool_call_hook(ctx, after_hook)

    def execute(tool_call_id: str, args: Any, signal: Any, update: Any) -> str:
        controller.abort()
        return "ok"

    definition = _echo(execute=execute, wants_signal=True)

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=ctx, signal=controller.signal
    )

    assert not result.is_error
    assert ran == ["after"]


async def test_after_hook_receives_the_active_signal() -> None:
    """`L09-R001`: pinned Pi's own `afterToolCall(context, signal)` passes the active run's
    signal as the hook's own second parameter -- a raw listener registered directly against
    `tools/post-execute` must receive the same signal `execute()` itself did, not merely see the
    after-hook run (which alone does not prove signal delivery)."""
    ctx = _ctx()
    signal = RunAbortController().signal
    seen: list[Any] = []

    async def after(result: Any, received_signal: Any, next_: Any) -> Any:
        seen.append(received_signal)
        return await next_()

    ctx.events.on(TOOLS_POST_EXECUTE, after)
    definition = _echo()

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=ctx, signal=signal
    )

    assert not result.is_error
    assert seen == [signal]


async def test_before_hook_receives_the_active_signal() -> None:
    """`L09-R001`: pinned Pi's own `beforeToolCall(context, signal)` passes the active run's
    signal as the hook's own second parameter."""
    ctx = _ctx()
    signal = RunAbortController().signal
    seen: list[Any] = []

    async def before(
        call: Any, definition: Any, arguments: Any, received_signal: Any, next_: Any
    ) -> Any:
        seen.append(received_signal)
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, before)
    definition = _echo()

    result = await execute_call(
        _call(value="x"), registry=_registry(definition), ctx=ctx, signal=signal
    )

    assert not result.is_error
    assert seen == [signal]
