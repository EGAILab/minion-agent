"""Streaming chunk vocabulary and the never-raises collection helper.

Every chunk carries `partial`, the message as assembled so far, so a consumer
can render any prefix without tracking state of its own.

Images do not stream: there are no image delta chunks by design. An image is
present in a message or it is not.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from .content import ToolCallBlock
from .errors import AdapterProtocolError
from .messages import AssistantMessage, StopReason


@dataclass(frozen=True, slots=True)
class StreamStart:
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class TextStart:
    content_index: int
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class TextDelta:
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class TextEnd:
    content_index: int
    text: str
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class ThinkingStart:
    content_index: int
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class ThinkingEnd:
    content_index: int
    thinking: str
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    content_index: int
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    content_index: int
    tool_call: ToolCallBlock
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class StreamDone:
    message: AssistantMessage
    partial: AssistantMessage


@dataclass(frozen=True, slots=True)
class StreamError:
    reason: StopReason
    message: AssistantMessage
    partial: AssistantMessage


type StreamChunk = (
    StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | StreamDone
    | StreamError
)

type AssistantStream = AsyncIterator[StreamChunk]


async def collect(
    stream: AssistantStream,
    on_chunk: Callable[[StreamChunk], None] | None = None,
) -> AssistantMessage:
    """Drain `stream` and return its settled message.

    `on_chunk` observes every chunk as it arrives, which is how the loop logs
    streaming fidelity without a second traversal — and without the session
    layer being visible from here.

    Never raises for model, network, or cancellation failures — those arrive
    as a `StreamError` whose message carries the reason. Raises only when an
    adapter breaks its contract by ending without a terminal chunk.
    """
    async for chunk in stream:
        if on_chunk is not None:
            on_chunk(chunk)
        if isinstance(chunk, StreamDone | StreamError):
            return chunk.message
    raise AdapterProtocolError("stream ended without a terminal chunk")
