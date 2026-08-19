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
from ..runtime import Context, ScopeKey
from .definition import ExecutionMode
from .execute import execute_call
from .registry import ToolRegistry
from .result import ToolResult


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
    calls: Sequence[ToolCallBlock], registry: ToolRegistry, scope: ScopeKey | None
) -> bool:
    """Pi's contagion rule: one exclusive tool serializes the whole batch.

    An unresolvable name is not exclusive. It never runs, so it has no
    exclusivity to spread, and treating it as exclusive would let a model's
    typo serialize an otherwise parallel batch.

    DSH instead groups calls so parallel-safe ones overlap around exclusive
    barriers. That is arguably better and produces different traces; preserving
    pi's semantics is the stated goal.
    """
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
    scope: ScopeKey | None = None,
) -> BatchOutcome:
    """Run every call in `calls`, returning results in source order."""
    completion: list[str] = []

    async def run(call: ToolCallBlock) -> ToolResult:
        result = await execute_call(call, registry=registry, ctx=ctx, scope=scope)
        completion.append(call.id)
        return result

    if _is_sequential(calls, registry, scope):
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
