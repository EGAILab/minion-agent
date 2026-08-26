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
from .execute import execute_call
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
) -> BatchOutcome:
    """Run every call in `calls`, returning results in source order.

    `default_mode` is the run-level execution-mode default (pinned Pi's
    `AgentLoopConfig.toolExecution?`, "Default: parallel") -- the effective default belongs to
    execution, not to `ToolDefinition` itself (`execution_mode: None` on a tool means "no
    per-tool preference," never "parallel"; only the batch decides what `None` falls back to).
    """
    completion: list[str] = []

    async def run(call: ToolCallBlock) -> ToolResult:
        result = await execute_call(call, registry=registry, ctx=ctx, scope=scope)
        completion.append(call.id)
        return result

    if _is_sequential(calls, registry, scope, default_mode):
        results = [await run(call) for call in calls]
    else:
        results = list(await asyncio.gather(*(run(call) for call in calls)))

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
) -> BatchOutcome:
    """Every call in `calls` becomes the same length-stop error result; none of them run.

    Pinned Pi's `failToolCallsFromTruncatedMessage` (`packages/agent/src/agent-loop.ts`): a
    `length` stop reason means the assistant's output was cut off by the token limit, so every
    tool call it carries may itself carry truncated arguments -- none are safe to execute. This
    unconditionally skips resolution, `prepare_arguments`, validation, the before-hook, `execute`,
    and the after-hook for every call; each becomes the identical error result, in source order.
    `tool_execution_start`/`tool_execution_end` still fire for each call, matching pinned Pi
    exactly. `terminate` is always `False` here -- pinned Pi never folds these results through
    `shouldTerminateToolBatch` at all.
    """
    scope_key = scope.key if isinstance(scope, Scope) else scope
    results: list[ToolResult] = []
    completion: list[str] = []
    for call in calls:
        ctx.events.emit(TOOLS_EXECUTION_START, call.id, call.name, call.arguments, scope=scope_key)
        result = text_result(
            call.id,
            f'Tool call "{call.name}" was not executed: the response hit the output token '
            "limit, so its arguments may be truncated. Re-issue the tool call with complete "
            "arguments.",
            call.name,
            is_error=True,
        )
        ctx.events.emit(TOOLS_EXECUTION_END, call.id, call.name, result, scope=scope_key)
        results.append(result)
        completion.append(call.id)

    return BatchOutcome(results=tuple(results), completion_order=tuple(completion), terminate=False)
