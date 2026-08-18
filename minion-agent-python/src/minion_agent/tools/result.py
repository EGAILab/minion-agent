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

from dataclasses import dataclass, field
from typing import Any

from ..llm import ContentBlock, TextBlock, ToolResultMessage


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One finalized tool outcome."""

    tool_call_id: str
    content: tuple[ContentBlock, ...]
    is_error: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    terminate: bool = False
    added_tool_names: tuple[str, ...] = ()

    def to_message(self) -> ToolResultMessage:
        """The model-visible projection of this result."""
        return ToolResultMessage(
            tool_call_id=self.tool_call_id,
            content=self.content,
            timestamp=0,
            is_error=self.is_error,
        )


def text_result(
    call_id: str, text: str, *, is_error: bool = False, terminate: bool = False
) -> ToolResult:
    """A result whose whole content is one block of text."""
    return ToolResult(
        tool_call_id=call_id,
        content=(TextBlock(text=text),),
        is_error=is_error,
        terminate=terminate,
    )
