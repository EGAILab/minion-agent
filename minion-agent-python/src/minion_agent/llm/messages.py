"""Messages, stop reasons, and token accounting.

Mirrors Pi's semantics: the same stop-reason vocabulary, the same treatment of
reasoning tokens as a subset of output rather than an extra class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .content import ContentBlock, TextBlock


class StopReason(StrEnum):
    """Why a provider stopped generating."""

    PENDING = "pending"
    """Still streaming; never a settled message's reason."""

    STOP = "stop"
    """The model ended its turn."""

    LENGTH = "length"
    """The output token cap was reached."""

    TOOL_USE = "tool_use"
    """The model requested tools and expects results."""

    ERROR = "error"
    """The request failed. `error_message` says how."""

    ABORTED = "aborted"
    """The caller cancelled."""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one request.

    `reasoning` is a *subset* of `output` where a provider reports it, so it is
    deliberately excluded from `total` — counting it separately would
    double-count the same tokens.
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int | None = None

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write


@dataclass(frozen=True, slots=True)
class UserMessage:
    """Input from a user or an application."""

    content: tuple[ContentBlock, ...]
    timestamp: int


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """One settled model response."""

    content: tuple[ContentBlock, ...]
    stop_reason: StopReason
    usage: Usage
    model: str
    provider: str
    timestamp: int
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """The result of one tool call, linked by `tool_call_id`."""

    tool_call_id: str
    content: tuple[ContentBlock, ...]
    timestamp: int
    is_error: bool = False


type Message = UserMessage | AssistantMessage | ToolResultMessage


def text_of(message: Message) -> str:
    """Concatenate a message's text blocks, ignoring every other kind."""
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))
