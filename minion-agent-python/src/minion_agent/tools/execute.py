"""One tool call, start to finish.

The pipeline is `tools/pre-execute` -> execute -> `tools/post-execute` (design
spec section 7), all of it here.

One rule governs every branch: **the call produces exactly one result.** An
unknown tool, arguments the model got wrong, a listener's refusal, and a tool
that raises all become error *results*. A missing result leaves the transcript
incoherent, and the model is then asked to continue from a conversation that
does not make sense.
"""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import ValidationError

from ..llm import ToolCallBlock
from ..runtime import Context, Scope, ScopeKey
from .decisions import Block, PreExecuteDecision, Proceed
from .definition import ToolDefinition
from .events import TOOLS_POST_EXECUTE, TOOLS_PRE_EXECUTE, TOOLS_UPDATE
from .registry import ToolRegistry
from .result import ToolResult, text_result


def _validate(definition: ToolDefinition, arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce `arguments` through the tool's parameter model.

    Defaults are filled in here rather than inside the tool, so a listener
    inspecting the arguments sees what the tool will actually receive. A raw
    JSON Schema `dict` parameters value (`TOOL-F010`) has no Python validator
    to run -- arguments pass through unchanged, including for a no-parameters
    tool's explicit empty-object schema (`L05-R005`).
    """
    if isinstance(definition.parameters, dict):
        return dict(arguments)
    return definition.parameters.model_validate(arguments).model_dump()


def _wants_update(execute: Any) -> bool:
    """Whether the tool declared a second parameter for partial output."""
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):  # pragma: no cover - builtins are not tools
        return False
    return len(signature.parameters) >= 2


async def _finalize(result: ToolResult, ctx: Context, scope: ScopeKey | None) -> ToolResult:
    """Run the result through `tools/post-execute`.

    The terminal is computed from the current arguments, because this event's
    terminal is "the result as currently transformed" (design spec section 3).
    A constant terminal would discard a lone listener's work -- the exact
    failure the terminal rule exists to prevent.

    Every path reaches here, error paths included: a blocked or failed call is
    the result most worth annotating, and a listener that only sometimes runs
    is a listener nobody can reason about.
    """
    transformed: ToolResult = await ctx.events.waterfall(
        TOOLS_POST_EXECUTE,
        result,
        terminal=lambda current, *_: current,
        scope=scope,
    )
    return transformed


async def execute_call(
    call: ToolCallBlock,
    *,
    registry: ToolRegistry,
    ctx: Context,
    scope: ScopeKey | Scope | None = None,
) -> ToolResult:
    """Run `call` and return its result, whatever happens."""
    definition = registry.resolve(call.name, scope)
    # Events want a bare ScopeKey (design spec section 3), not a live Scope --
    # normalize once; the registry lookup above already used the richer value
    # for its own disposed-scope check (L05-R002).
    scope = scope.key if isinstance(scope, Scope) else scope
    if definition is None:
        return await _finalize(
            text_result(call.id, f"unknown tool {call.name!r}", call.name, is_error=True),
            ctx,
            scope,
        )

    try:
        arguments = _validate(definition, call.arguments)
    except ValidationError as error:
        # Surfaced to the model, which chose these arguments and is the only
        # party that can choose better ones.
        return await _finalize(
            text_result(call.id, f"invalid arguments: {error}", call.name, is_error=True),
            ctx,
            scope,
        )

    decision: PreExecuteDecision = await ctx.events.waterfall(
        TOOLS_PRE_EXECUTE,
        call,
        definition,
        arguments,
        terminal=Proceed(arguments=arguments),
        scope=scope,
    )
    if isinstance(decision, Block):
        return await _finalize(
            text_result(
                call.id, decision.reason, call.name, is_error=True, terminate=decision.terminate
            ),
            ctx,
            scope,
        )

    def update(partial: str) -> None:
        ctx.events.emit(TOOLS_UPDATE, call.id, partial, scope=scope)

    try:
        outcome = (
            definition.execute(decision.arguments, update)
            if _wants_update(definition.execute)
            else definition.execute(decision.arguments)
        )
        value = await outcome if inspect.isawaitable(outcome) else outcome
    except Exception as error:  # surfaced to the model, not raised
        return await _finalize(
            text_result(call.id, f"{type(error).__name__}: {error}", call.name, is_error=True),
            ctx,
            scope,
        )

    if isinstance(value, ToolResult):
        # Tools identify their own call only by accident; the pipeline knows.
        return await _finalize(
            ToolResult(
                tool_call_id=call.id,
                content=value.content,
                tool_name=call.name,
                is_error=value.is_error,
                details=value.details,
                terminate=value.terminate,
                added_tool_names=value.added_tool_names,
            ),
            ctx,
            scope,
        )
    return await _finalize(text_result(call.id, str(value), call.name), ctx, scope)
