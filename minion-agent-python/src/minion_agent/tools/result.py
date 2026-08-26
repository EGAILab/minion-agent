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
            details=self.details or None,
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
