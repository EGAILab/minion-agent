"""Executing a batch of calls: contagion, ordering, and the terminate fold.

Two orders are normative and different (design spec section 6):

* `tool_execution_end` is emitted in **completion** order.
* Tool-result **messages** are emitted in **assistant source** order.

A batch therefore reports both, and the caller writes both into the log --
which is what lets one log reconstruct each without the projection guessing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from ..llm import ToolCallBlock
from ..runtime import Context, Scope, ScopeKey
from .definition import ExecutionMode
from .events import TOOLS_EXECUTION_END, TOOLS_EXECUTION_START
from .execute import (
    OnExecutionEnd,
    OnExecutionStart,
    _execute_and_finalize,
    _preflight,
    _Prepared,
    execute_call,
)
from .registry import ToolRegistry
from .result import ToolResult, text_result


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """Everything the loop needs to log and fold one batch."""

    results: tuple[ToolResult, ...]
    """Assistant source order -- what the model sees."""

    completion_order: tuple[str, ...]
    """Call ids in the order their tools finished."""

    terminate: bool
    """Whether *every* result asked to end the turn."""

    def completion_index(self, call_id: str) -> int:
        """Where `call_id` finished relative to the rest of its batch."""
        return self.completion_order.index(call_id)


def _is_sequential(
    calls: Sequence[ToolCallBlock],
    registry: ToolRegistry,
    scope: ScopeKey | Scope | None,
    default_mode: ExecutionMode,
) -> bool:
    """Pi's contagion rule: the run-level default, or one exclusive tool, serializes the
    whole batch (`config.toolExecution === "sequential" || hasSequentialToolCall`,
    `packages/agent/src/agent-loop.ts::executeToolCalls`).

    An unresolvable name is not exclusive. It never runs, so it has no
    exclusivity to spread, and treating it as exclusive would let a model's
    typo serialize an otherwise parallel batch.

    DSH instead groups calls so parallel-safe ones overlap around exclusive
    barriers. That is arguably better and produces different traces; preserving
    pi's semantics is the stated goal.
    """
    if default_mode is ExecutionMode.SEQUENTIAL:
        return True
    for call in calls:
        definition = registry.resolve(call.name, scope)
        if definition is not None and definition.mode is ExecutionMode.SEQUENTIAL:
            return True
    return False


async def execute_batch(
    calls: Sequence[ToolCallBlock],
    *,
    registry: ToolRegistry,
    ctx: Context,
    scope: ScopeKey | Scope | None = None,
    default_mode: ExecutionMode = ExecutionMode.PARALLEL,
    on_execution_start: OnExecutionStart | None = None,
    on_execution_end: OnExecutionEnd | None = None,
) -> BatchOutcome:
    """Run every call in `calls`, returning results in source order.

    `default_mode` is the run-level execution-mode default (pinned Pi's
    `AgentLoopConfig.toolExecution?`, "Default: parallel") -- the effective default belongs to
    execution, not to `ToolDefinition` itself (`execution_mode: None` on a tool means "no
    per-tool preference," never "parallel"; only the batch decides what `None` falls back to).

    Parallel mode is NOT "preflight every call concurrently too" (`IR-L06-001`): pinned Pi's
    `executeToolCallsParallel` resolves/prepares/validates/before-hooks every call strictly
    sequentially, in source order -- `tool_execution_start` for call 2 never fires before call 1's
    entire preflight has settled -- and only starts `execute()`/the after-hook concurrently once
    *every* call in the batch has survived preflight. An immediate outcome (unknown tool, prepare/
    validate/before-hook failure or block) finalizes right there in the sequential phase, before
    any prepared call's `execute()` begins; only calls that survive preflight run concurrently
    afterward. `_preflight`/`_execute_and_finalize` (`execute.py`) are the same two functions
    `execute_call` itself is built from -- no stage's rules are duplicated here.

    `on_execution_start`/`on_execution_end`, when supplied, are awaited at the exact points
    described in `execute.py` -- for a sequential batch, per call, in order; for a parallel batch,
    `on_execution_start` still runs sequentially across the whole preflight barrier (never
    concurrently with itself), and `on_execution_end` runs as each call's own concurrent
    `execute()`+after-hook phase finishes (`L08-R002`, PASS 6). A listener that raises propagates
    out of this function immediately, preventing any call not yet past that point from proceeding
    -- additive: `None` (every existing caller) preserves this function's own certified behavior.
    """
    completion: list[str] = []
    scope_key = scope.key if isinstance(scope, Scope) else scope

    if _is_sequential(calls, registry, scope, default_mode):

        async def run(call: ToolCallBlock) -> ToolResult:
            result = await execute_call(
                call,
                registry=registry,
                ctx=ctx,
                scope=scope,
                on_execution_start=on_execution_start,
                on_execution_end=on_execution_end,
            )
            completion.append(result.tool_call_id)
            return result

        results = [await run(call) for call in calls]
    else:
        outcomes: list[_Prepared | ToolResult] = []
        for call in calls:
            outcome = await _preflight(
                call,
                registry=registry,
                ctx=ctx,
                scope=scope_key,
                on_execution_start=on_execution_start,
                on_execution_end=on_execution_end,
            )
            # An immediate outcome already produced its final result -- and emitted
            # tools/execution-end -- during preflight, strictly before the barrier below, so its
            # completion is recorded here, not deferred into the concurrent phase.
            if isinstance(outcome, ToolResult):
                completion.append(outcome.tool_call_id)
            outcomes.append(outcome)

        async def resolve(outcome: _Prepared | ToolResult) -> ToolResult:
            if isinstance(outcome, ToolResult):
                return outcome
            result = await _execute_and_finalize(
                outcome, ctx=ctx, scope=scope_key, on_execution_end=on_execution_end
            )
            completion.append(result.tool_call_id)
            return result

        # The barrier: every call above has already finished preflight before any of these
        # execute()+after-hook stages starts.
        results = list(await asyncio.gather(*(resolve(outcome) for outcome in outcomes)))

    return BatchOutcome(
        results=tuple(results),
        completion_order=tuple(completion),
        # Non-empty is part of the rule, not a guard: an empty batch has no
        # result asking to stop, and must not end a turn by vacuous agreement.
        terminate=bool(results) and all(result.terminate for result in results),
    )


async def execute_length_stop_batch(
    calls: Sequence[ToolCallBlock],
    *,
    ctx: Context,
    scope: ScopeKey | Scope | None = None,
    on_execution_start: OnExecutionStart | None = None,
    on_execution_end: OnExecutionEnd | None = None,
) -> BatchOutcome:
    """Every call in `calls` becomes the same length-stop error result; none of them run.

    Pinned Pi's `failToolCallsFromTruncatedMessage` (`packages/agent/src/agent-loop.ts`): a
    `length` stop reason means the assistant's output was cut off by the token limit, so every
    tool call it carries may itself carry truncated arguments -- none are safe to execute. This
    unconditionally skips resolution, `prepare_arguments`, validation, the before-hook, `execute`,
    and the after-hook for every call; each becomes the identical error result, in source order.
    `tool_execution_start`/`tool_execution_end` still fire for each call, matching pinned Pi
    exactly -- including, when supplied, awaiting `on_execution_start`/`on_execution_end` at the
    same points `execute_batch` does (`L08-R002`, PASS 6). `terminate` is always `False` here --
    pinned Pi never folds these results through `shouldTerminateToolBatch` at all.
    """
    scope_key = scope.key if isinstance(scope, Scope) else scope
    results: list[ToolResult] = []
    completion: list[str] = []
    for call in calls:
        ctx.events.emit(TOOLS_EXECUTION_START, call.id, call.name, call.arguments, scope=scope_key)
        if on_execution_start is not None:
            await on_execution_start(call.id, call.name, call.arguments)
        result = text_result(
            call.id,
            f'Tool call "{call.name}" was not executed: the response hit the output token '
            "limit, so its arguments may be truncated. Re-issue the tool call with complete "
            "arguments.",
            call.name,
            is_error=True,
        )
        ctx.events.emit(TOOLS_EXECUTION_END, call.id, call.name, result, scope=scope_key)
        if on_execution_end is not None:
            await on_execution_end(call.id, call.name, result)
        results.append(result)
        completion.append(call.id)

    return BatchOutcome(results=tuple(results), completion_order=tuple(completion), terminate=False)
