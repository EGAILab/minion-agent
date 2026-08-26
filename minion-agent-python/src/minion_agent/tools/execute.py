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

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from ..llm import ToolCallBlock
from ..runtime import Context, Scope, ScopeKey
from .decisions import AfterToolCallOverride, Block, PreExecuteDecision, Proceed
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


class ArgumentValidationError(Exception):
    """Arguments do not satisfy the tool's parameter schema.

    Raised uniformly whether the schema is a pydantic model or a raw, object-valued JSON-Schema
    mapping (`L06-R001`) -- `execute_call`'s error-conversion boundary does not need to know
    which validator produced it. `str(error)` is always the clean semantic message (never a
    Python exception class name -- `L06-R002`)."""


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
    """Validate `arguments` against the tool's parameter schema.

    Pinned Pi's `validateToolArguments` validates every `Tool.parameters: TSchema`, with no
    exemption for a raw-object-schema representation (`L06-R001`, corrected this pass -- an
    earlier revision skipped validation entirely for a raw JSON-Schema `dict`, a genuine
    `PI_PARITY_DEFECT`). A pydantic-model-backed `ToolDefinition.parameters` gets real pydantic
    validation (and its default-filling); a raw, object-valued JSON-Schema `dict` (`TOOL-F010`)
    is validated for real too, via the general `jsonschema` library against the exact schema
    Layer 05 already approved -- this is deliberately NOT reproducing pinned Pi's TypeBox-specific
    coercion algorithm (`packages/ai/src/utils/validation.ts`), which remains a disclosed,
    intentional divergence: the shared contract is "arguments conform to the supplied JSON
    Schema," not "TypeBox's exact clone/convert/coerce pipeline." Neither path fills in
    JSON-Schema-only defaults the way pydantic does for its own models; a raw-schema tool sees
    its arguments unchanged when they already validate.
    """
    if isinstance(definition.parameters, dict):
        try:
            Draft202012Validator(definition.parameters).validate(arguments)
        except JsonSchemaValidationError as error:
            raise ArgumentValidationError(error.message) from error
        return dict(arguments)
    try:
        return definition.parameters.model_validate(arguments).model_dump()
    except PydanticValidationError as error:
        raise ArgumentValidationError(str(error)) from error


def _wants_update(execute: Any) -> bool:
    """Whether the tool declared a third parameter (after `tool_call_id`, `arguments`) for
    partial output."""
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):  # pragma: no cover - builtins are not tools
        return False
    return len(signature.parameters) >= 3


def _merge_override(current: ToolResult, override: AfterToolCallOverride | None) -> ToolResult:
    """Apply an after-hook's Pi-shaped partial override to `current`.

    Field-by-field, matching pinned Pi's `finalizeExecutedToolCall` merge exactly: an omitted
    (`None`) field keeps `current`'s value; a supplied field replaces it wholesale (no deep
    merge). Fields outside `AfterToolCallOverride`'s five (`tool_call_id`, `tool_name`,
    `added_tool_names`) are never touched -- they cannot be, since the override type has no
    slot for them (`L06-R003`).
    """
    if override is None:
        return current
    return ToolResult(
        tool_call_id=current.tool_call_id,
        tool_name=current.tool_name,
        added_tool_names=current.added_tool_names,
        content=current.content if override.content is None else override.content,
        details=current.details if override.details is None else override.details,
        is_error=current.is_error if override.is_error is None else override.is_error,
        usage=current.usage if override.usage is None else override.usage,
        terminate=current.terminate if override.terminate is None else override.terminate,
    )


type AfterToolCallHook = Any
"""`Callable[[ToolResult], AfterToolCallOverride | None]` (sync or async) -- see
`register_after_tool_call_hook`. Spelled `Any` rather than a `Callable[...]` alias so a hook may
freely be a plain function, a bound method, or an async function without fighting `Awaitable`
variance; `_finalize` awaits the result only when it actually is one."""


def register_after_tool_call_hook(
    ctx: Context, hook: AfterToolCallHook, *, scope: ScopeKey | None = None
) -> Any:
    """The recommended way to extend `tools/post-execute` (`L06-R003`/`L06-R006`).

    `hook` receives the current, already-merged `ToolResult` (read-only) and may return an
    `AfterToolCallOverride` (or `None`/nothing for no change) -- never the whole result, so a
    hook written against this API cannot even attempt to replace execution identity or
    `added_tool_names`. Multiple hooks compose as a deterministic, registration-ordered fold
    (`TOOL-005`): each sees the result exactly as merged by every earlier hook, mirroring pinned
    Pi's own single-callback semantics for the zero/one-hook cases and extending it, for N hooks,
    as an intentional Minion architectural divergence -- not something pinned Pi itself defines.

    This helper's own constraint is a convenience, not the authoritative boundary:
    `tools/post-execute` remains a public Runtime event, so a caller may also register a raw
    listener directly via `ctx.events.on(TOOLS_POST_EXECUTE, ...)` and return a whole,
    differently-identified `ToolResult`. `_finalize`'s restoration of `tool_call_id`/`tool_name`/
    `added_tool_names` -- at every listener-to-listener handoff, not only once the whole chain
    finishes (`L06-R003`) -- is what actually makes identity/`added_tool_names` replacement
    impossible, regardless of which registration path produced a given listener's output, and
    regardless of whether another listener runs afterward to observe it.

    Returns the same disposer `EventBus.on` returns.
    """

    async def listener(result: ToolResult, next_: Any) -> ToolResult:
        outcome = hook(result)
        override = await outcome if inspect.isawaitable(outcome) else outcome
        merged: ToolResult = await next_(_merge_override(result, override))
        return merged

    return ctx.events.on(TOOLS_POST_EXECUTE, listener, scope=scope)


async def _finalize(result: ToolResult, ctx: Context, scope: ScopeKey | None) -> ToolResult:
    """Run the result through every registered `tools/post-execute` hook (pinned Pi's
    `afterToolCall`, extended to N listeners -- see `register_after_tool_call_hook`).

    The terminal is computed from the current arguments, because this event's
    terminal is "the result as currently transformed" (design spec section 3).
    A constant terminal would discard a lone listener's work -- the exact
    failure the terminal rule exists to prevent.

    Called only for an outcome that reached `execute()` -- see `execute_call`.

    Execution identity (`tool_call_id`, `tool_name`) and `added_tool_names` are restored from
    `result` -- the pristine, pre-hook value -- at **every** listener-to-listener handoff, not only
    once the whole waterfall completes (`L06-R003`, second closure). A first fix restored them only
    after `ctx.events.waterfall` returned: correct for the *final* result, but a listener that
    delegates via `next(replacement)` with a forged `replacement` still handed that forgery,
    unrestored, to whichever listener ran next -- observable (and provably exploitable: a later
    listener can read the forged fields and copy them into an allowed field like `details`) even
    though the finally-returned result looked clean. `waterfall`'s `normalize_step` (see
    `EventBus.waterfall`) closes that gap generically: it runs on whatever a listener passes to
    `next` before the *next* listener ever sees it, so no listener -- helper-registered or raw,
    anywhere in the chain -- can ever observe a predecessor's unauthorized replacement. A listener
    that short-circuits instead of delegating has no next listener to protect from, so its direct
    return is not run through `normalize_step`; the restoration below still covers that value
    (and every other final value) once the waterfall as a whole returns. `ToolResult` is itself
    frozen, so in-place mutation of the passed-in object remains structurally impossible,
    independent of either restoration point.
    """
    tool_call_id = result.tool_call_id
    tool_name = result.tool_name
    added_tool_names = result.added_tool_names

    def _restore(current: tuple[Any, ...]) -> tuple[Any, ...]:
        (candidate,) = current
        if (
            candidate.tool_call_id == tool_call_id
            and candidate.tool_name == tool_name
            and candidate.added_tool_names == added_tool_names
        ):
            return current
        return (
            ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                added_tool_names=added_tool_names,
                content=candidate.content,
                details=candidate.details,
                is_error=candidate.is_error,
                usage=candidate.usage,
                terminate=candidate.terminate,
            ),
        )

    transformed: ToolResult = await ctx.events.waterfall(
        TOOLS_POST_EXECUTE,
        result,
        terminal=lambda current, *_: current,
        scope=scope,
        normalize_step=_restore,
    )
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        added_tool_names=added_tool_names,
        content=transformed.content,
        details=transformed.details,
        is_error=transformed.is_error,
        usage=transformed.usage,
        terminate=transformed.terminate,
    )


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
    except ArgumentValidationError as error:
        # Surfaced to the model, which chose these arguments and is the only
        # party that can choose better ones.
        return _immediate(
            call,
            ctx,
            scope,
            text_result(call.id, f"invalid arguments: {error}", call.name, is_error=True),
        )
    except Exception as error:  # prepare_arguments or a before-hook listener raised
        # Pinned Pi surfaces error.message, never a runtime type name (L06-R002).
        return _immediate(
            call, ctx, scope, text_result(call.id, str(error), call.name, is_error=True)
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
        executed = text_result(call.id, str(error), call.name, is_error=True)
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
        finalized = text_result(call.id, str(error), call.name, is_error=True)
    ctx.events.emit(TOOLS_EXECUTION_END, call.id, call.name, finalized, scope=scope)
    return finalized
