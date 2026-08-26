"""One tool call, start to finish.

Stages, matching pinned Pi's `prepareToolCall` + `executePreparedToolCall` +
`finalizeExecutedToolCall` exactly (`packages/agent/src/agent-loop.ts`, Layer 06):

    resolve -> prepare -> validate -> before-hook -> execute (+ live updates) -> after-hook

An outcome decided before `execute()` runs (unknown tool, a prepare/validate/before-hook
exception, or an explicit before-hook block) is "immediate" and never reaches `execute()` or the
after-hook (`tools/post-execute`) at all -- pinned Pi's `finalizeExecutedToolCall` (the after-hook)
is only ever invoked for an outcome that actually reached `execute()`, success or failure alike
(`TOOL-017`). One rule governs every branch regardless: **the call produces exactly one result.**
An unknown tool, arguments the model got wrong, a listener's refusal, and a tool that raises all
become error *results*. A missing result leaves the transcript incoherent, and the model is then
asked to continue from a conversation that does not make sense.
"""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import ValidationError

from ..llm import ToolCallBlock
from ..runtime import Context, Scope, ScopeKey
from .decisions import Block, PreExecuteDecision, Proceed
from .definition import ToolDefinition
from .events import (
    TOOLS_EXECUTION_END,
    TOOLS_EXECUTION_START,
    TOOLS_POST_EXECUTE,
    TOOLS_PRE_EXECUTE,
    TOOLS_UPDATE,
)
from .registry import ToolRegistry
from .result import ToolResult, text_result


def _prepare(definition: ToolDefinition, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the tool's `prepare_arguments` compatibility shim, if any, before validation.

    Pinned Pi's `AgentTool.prepareArguments?: (args) => Static<TParameters>` always runs
    before schema validation, never after (`TOOL-017`; field/signature only until now --
    `TOOL-F002`). A fresh `dict` is passed in so a shim that mutates its argument in place
    cannot corrupt the original tool call's arguments (design spec section 6 nonmutation
    discipline, matching XFORM's own rule).
    """
    if definition.prepare_arguments is None:
        return arguments
    return definition.prepare_arguments(dict(arguments))


def _validate(definition: ToolDefinition, arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce `arguments` through the tool's parameter model.

    Defaults are filled in here rather than inside the tool, so a listener
    inspecting the arguments sees what the tool will actually receive. A raw
    JSON Schema `dict` parameters value (`TOOL-F010`) has no Python validator
    to run -- arguments pass through unchanged, including for a no-parameters
    tool's explicit empty-object schema (`L05-R005`). This is a disclosed,
    intentional Python-only limitation, not a silently-skipped contract:
    pinned Pi always validates via TypeBox regardless of representation,
    but Layer 05 deliberately declined to make Layer 06 into a general
    JSON Schema validator (`TOOL-017`).
    """
    if isinstance(definition.parameters, dict):
        return dict(arguments)
    return definition.parameters.model_validate(arguments).model_dump()


def _wants_update(execute: Any) -> bool:
    """Whether the tool declared a third parameter (after `tool_call_id`, `arguments`) for
    partial output."""
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):  # pragma: no cover - builtins are not tools
        return False
    return len(signature.parameters) >= 3


async def _finalize(result: ToolResult, ctx: Context, scope: ScopeKey | None) -> ToolResult:
    """Run the result through `tools/post-execute` (pinned Pi's `afterToolCall`).

    The terminal is computed from the current arguments, because this event's
    terminal is "the result as currently transformed" (design spec section 3).
    A constant terminal would discard a lone listener's work -- the exact
    failure the terminal rule exists to prevent.

    Called only for an outcome that reached `execute()` -- see `execute_call`.
    """
    transformed: ToolResult = await ctx.events.waterfall(
        TOOLS_POST_EXECUTE,
        result,
        terminal=lambda current, *_: current,
        scope=scope,
    )
    return transformed


def _immediate(
    call: ToolCallBlock, ctx: Context, scope: ScopeKey | None, result: ToolResult
) -> ToolResult:
    """An outcome decided before `execute()` runs: emit `tool_execution_end` directly, skipping
    the after-hook entirely (pinned Pi never invokes `afterToolCall` for these)."""
    ctx.events.emit(TOOLS_EXECUTION_END, call.id, call.name, result, scope=scope)
    return result


async def execute_call(
    call: ToolCallBlock,
    *,
    registry: ToolRegistry,
    ctx: Context,
    scope: ScopeKey | Scope | None = None,
) -> ToolResult:
    """Run `call` and return its result, whatever happens."""
    # Events want a bare ScopeKey (design spec section 3), not a live Scope --
    # normalize once; the registry lookup below already used the richer value
    # for its own disposed-scope check (L05-R002).
    scope = scope.key if isinstance(scope, Scope) else scope
    ctx.events.emit(TOOLS_EXECUTION_START, call.id, call.name, call.arguments, scope=scope)

    definition = registry.resolve(call.name, scope)
    if definition is None:
        return _immediate(
            call,
            ctx,
            scope,
            text_result(call.id, f"unknown tool {call.name!r}", call.name, is_error=True),
        )

    try:
        prepared_arguments = _prepare(definition, call.arguments)
        validated_arguments = _validate(definition, prepared_arguments)
        decision: PreExecuteDecision = await ctx.events.waterfall(
            TOOLS_PRE_EXECUTE,
            call,
            definition,
            validated_arguments,
            terminal=Proceed(arguments=validated_arguments),
            scope=scope,
        )
    except ValidationError as error:
        # Surfaced to the model, which chose these arguments and is the only
        # party that can choose better ones.
        return _immediate(
            call,
            ctx,
            scope,
            text_result(call.id, f"invalid arguments: {error}", call.name, is_error=True),
        )
    except Exception as error:  # prepare_arguments or a before-hook listener raised
        return _immediate(
            call,
            ctx,
            scope,
            text_result(call.id, f"{type(error).__name__}: {error}", call.name, is_error=True),
        )

    if isinstance(decision, Block):
        return _immediate(
            call,
            ctx,
            scope,
            text_result(
                call.id, decision.reason, call.name, is_error=True, terminate=decision.terminate
            ),
        )

    # From here the call has reached "prepared": execute() will run, and whatever it
    # produces -- success or failure -- goes through the after-hook (pinned Pi's
    # finalizeExecutedToolCall is invoked uniformly for both).
    accepting_updates = True

    def update(partial: str) -> None:
        # Pinned Pi: "Calls made after the tool promise settles are ignored"
        # (`AgentToolUpdateCallback`). `ctx.events.emit` is synchronous here, so there is no
        # promise-drain queue to manage -- the flag alone decides late vs. live.
        if not accepting_updates:
            return
        ctx.events.emit(TOOLS_UPDATE, call.id, partial, scope=scope)

    try:
        outcome = (
            definition.execute(call.id, decision.arguments, update)
            if _wants_update(definition.execute)
            else definition.execute(call.id, decision.arguments)
        )
        value = await outcome if inspect.isawaitable(outcome) else outcome
    except Exception as error:  # surfaced to the model, not raised
        accepting_updates = False
        executed = text_result(
            call.id, f"{type(error).__name__}: {error}", call.name, is_error=True
        )
    else:
        accepting_updates = False
        if isinstance(value, ToolResult):
            # Tools identify their own call only by accident; the pipeline knows.
            executed = ToolResult(
                tool_call_id=call.id,
                content=value.content,
                tool_name=call.name,
                is_error=value.is_error,
                details=value.details,
                terminate=value.terminate,
                added_tool_names=value.added_tool_names,
                usage=value.usage,
            )
        else:
            executed = text_result(call.id, str(value), call.name)

    try:
        finalized = await _finalize(executed, ctx, scope)
    except Exception as error:
        # Pinned Pi's finalizeExecutedToolCall: an after-hook exception REPLACES the entire
        # prior result -- success or failure alike -- with a plain error result. Nothing from
        # `executed` (content, details, usage, terminate) survives; this is not a merge.
        finalized = text_result(
            call.id, f"{type(error).__name__}: {error}", call.name, is_error=True
        )
    ctx.events.emit(TOOLS_EXECUTION_END, call.id, call.name, finalized, scope=scope)
    return finalized
