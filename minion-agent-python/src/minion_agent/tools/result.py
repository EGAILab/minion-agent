"""What a tool returns.

Distinct from `ToolResultMessage` on purpose. The message is the model-visible
artifact; the result is what the pipeline carries, and it holds three things
that must never reach the model:

* `details` — structured annotation for listeners and telemetry.
* `terminate` — an instruction to the loop, not something the model said.
* `added_tool_names` — tools this result introduced, available from this
  transcript point onward (design spec section 7).

Frozen, so `tools/post-execute` transforms by replacement. A mutable result
would let one listener change a value another already returned.

`ToolPartialResult` (`L08-R011`) is the DIFFERENT, narrower type for a tool's own LIVE partial
report -- pinned Pi's `AgentToolResult<T>` exactly, with no `tool_call_id`/`tool_name`/`is_error`
of its own (those live on the enclosing event). `ToolResult` is Minion's own pipeline-level
FINALIZED-outcome type, a superset that adds those three as pipeline bookkeeping -- do not reuse it
for a partial value; a partial that reports `is_error`/spoofed identity would be observably
different from pinned Pi and certified Rust, which have no such fields on this type at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..llm import TextBlock, ToolResultContentBlock, ToolResultMessage, Usage


def _now_ms() -> int:
    """Wall-clock milliseconds, matching Pi's `Date.now()` for a synthesized
    `ToolResultMessage`'s timestamp (`createToolResultMessage`,
    `packages/agent/src/agent-loop.ts`) -- same convention already established
    for XFORM's synthesized results (`llm/transform_messages.py::_now_ms`)."""
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One finalized tool outcome."""

    tool_call_id: str
    content: tuple[ToolResultContentBlock, ...]
    tool_name: str
    """The tool that produced this result. Required, matching
    `ToolResultMessage.tool_name` (LLM-F0-delta finding A) -- the pipeline
    supplies it from the call it dispatched, since "tools identify their own
    call only by accident; the pipeline knows" (`tools/execute.py`)."""
    is_error: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    terminate: bool = False
    added_tool_names: tuple[str, ...] = ()
    usage: Usage | None = None
    """Usage from the tool execution itself (pinned Pi `AgentToolResult.usage?`). Preserved
    end to end into `ToolResultMessage.usage`; never folded into main LLM context token
    accounting (`TOOL-017`, closing a gap `ToolResult` did not represent before Layer 06)."""

    def to_message(self) -> ToolResultMessage:
        """The model-visible projection of this result.

        `details`/`added_tool_names` ride alongside `content` as structured
        metadata (design spec section 4) -- distinct from "must never reach
        the model" above, which is about `content`, the readable payload.
        """
        return ToolResultMessage(
            tool_call_id=self.tool_call_id,
            content=self.content,
            timestamp=_now_ms(),
            tool_name=self.tool_name,
            is_error=self.is_error,
            # Pass through unchanged (IR-L06-004): pinned Pi's `AgentToolResult.details: T` is
            # REQUIRED, and `createToolResultMessage` copies it verbatim -- `createErrorToolResult`
            # sets it to `{}`, not absent, and that `{}` survives all the way to
            # `ToolResultMessage.details`. An earlier revision wrote `self.details or None`, which
            # collapsed the empty-dict default every non-annotating result already carries into
            # `None` -- observably different from Pi for every error result and every tool that
            # never sets details. `added_tool_names` is the opposite case and is unaffected: Pi's
            # own `createToolResultMessage` conditionally omits that key entirely when the array is
            # empty (`addedToolNames?.length ? {...} : {}`), which `self.added_tool_names or None`
            # already matches.
            details=self.details,
            usage=self.usage,
            added_tool_names=self.added_tool_names or None,
        )


def text_result(
    call_id: str, text: str, tool_name: str, *, is_error: bool = False, terminate: bool = False
) -> ToolResult:
    """A result whose whole content is one block of text."""
    return ToolResult(
        tool_call_id=call_id,
        content=(TextBlock(text=text),),
        tool_name=tool_name,
        is_error=is_error,
        terminate=terminate,
    )


@dataclass(frozen=True, slots=True)
class ToolPartialResult:
    """A tool's own live partial-output report (`L08-R011`) -- pinned Pi's own
    `AgentToolResult<T>` (`packages/agent/src/types.ts:361-375`) EXACTLY: `content`/`details`
    required, `usage`/`added_tool_names`/`terminate` genuinely optional (`None` when the tool did
    not set them, distinguishable from an explicit falsy/empty value the same way Pi's own
    `usage?`/`addedToolNames?`/`terminate?` are `undefined`, not defaulted, when omitted).

    Deliberately NOT `ToolResult`: this type carries NO `tool_call_id`, `tool_name`, or `is_error`
    -- pinned Pi's own `AgentToolResult<T>` has none of those either. Call identity already lives
    on the enclosing `tool_execution_update` event (`ToolExecutionUpdate.tool_call_id`/
    `.tool_name`), and Pi has no `isError` concept on this type at all: "Execute the tool call.
    Throw on failure instead of encoding errors in `content`" (`AgentTool.execute`'s own
    docstring) -- a thrown/rejected `execute()` becomes an error OUTCOME the pipeline itself
    produces, never a field a tool sets on its own returned/reported value. An earlier revision
    reused `ToolResult` (this module's own pipeline-level FINALIZED-outcome type, which
    additionally carries `tool_call_id`/`tool_name`/`is_error` as Minion-specific pipeline
    bookkeeping) for the partial-update value too -- observably larger than Pi's/certified Rust's
    own `AgentToolResult`, and a genuine parity defect, not a superset with no cost."""

    content: tuple[ToolResultContentBlock, ...]
    details: dict[str, Any]
    """Genuinely REQUIRED (no default), matching pinned Pi's own `AgentToolResult<T>.details: T`
    exactly (`L08-R011`, contract-convergence characterization): an earlier revision defaulted
    this to `{}` via `field(default_factory=dict)`, letting `ToolPartialResult(content=())`
    construct successfully without ever supplying it -- optional-in-practice despite this type's
    own docstring calling it required, and the paired canonical encoder then omitted an explicit
    `{}` from observed evidence as if it had never been set, collapsing "required and empty" into
    "absent" (the identical hazard `ToolResult.to_message()`'s own docstring already documents and
    guards against for the FINAL result -- `details=self.details`, never `self.details or None`).
    A convenience constructor (`text_partial_result`) may still supply `{}` explicitly; the type
    itself no longer lets a caller omit it."""
    usage: Usage | None = None
    added_tool_names: tuple[str, ...] | None = None
    terminate: bool | None = None


def text_partial_result(text: str) -> ToolPartialResult:
    """A partial result whose whole content is one block of text -- the common case for a tool
    that only ever streams incremental text, matching `text_result`'s own convenience shape."""
    return ToolPartialResult(content=(TextBlock(text=text),), details={})
