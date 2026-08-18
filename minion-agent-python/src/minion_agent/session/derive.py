"""Projecting the log's surface into model history.

Encoding lives here rather than on the message types because it is a property
of *storage*, not of the vocabulary: the LLM layer must not know a log exists.

Images encode by reference when they carry one; inline bytes are base64-encoded
so the log stays JSON-safe. The session service resolves inline images to
content-addressed references before dispatch, so a logged reference is
immutable (design spec section 4).
"""

from __future__ import annotations

import base64
from typing import Any

from ..llm.content import ContentBlock, ImageBlock, TextBlock, ThinkingBlock, ToolCallBlock
from ..llm.messages import (
    AssistantMessage,
    Message,
    StopReason,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .events import EventKind, SessionEvent, is_surface
from .log import SessionLog


def _encode_block(block: ContentBlock) -> dict[str, Any]:
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ThinkingBlock():
            return {"type": "thinking", "thinking": block.thinking}
        case ImageBlock():
            encoded: dict[str, Any] = {"type": "image", "mime_type": block.mime_type}
            if block.reference is not None:
                encoded["reference"] = block.reference
            else:
                # Guaranteed non-None by ImageBlock.__post_init__.
                assert block.data is not None
                encoded["data"] = base64.b64encode(block.data).decode("ascii")
            return encoded
        case ToolCallBlock():
            return {
                "type": "tool_call",
                "id": block.id,
                "name": block.name,
                "arguments": block.arguments,
            }


def _decode_block(raw: dict[str, Any]) -> ContentBlock:
    kind = raw["type"]
    if kind == "text":
        return TextBlock(text=raw["text"])
    if kind == "thinking":
        return ThinkingBlock(thinking=raw["thinking"])
    if kind == "image":
        if "reference" in raw:
            return ImageBlock(mime_type=raw["mime_type"], reference=raw["reference"])
        return ImageBlock(mime_type=raw["mime_type"], data=base64.b64decode(raw["data"]))
    if kind == "tool_call":
        return ToolCallBlock(id=raw["id"], name=raw["name"], arguments=raw["arguments"])
    raise ValueError(f"unknown content block type {kind!r}")


def encode_message(message: Message) -> dict[str, Any]:
    """Render `message` as JSON-safe data the log will accept."""
    content = [_encode_block(block) for block in message.content]
    match message:
        case UserMessage():
            return {"role": "user", "content": content, "timestamp": message.timestamp}
        case AssistantMessage():
            return {
                "role": "assistant",
                "content": content,
                "timestamp": message.timestamp,
                "stop_reason": message.stop_reason.value,
                "model": message.model,
                "provider": message.provider,
                "error_message": message.error_message,
                "usage": {
                    "input": message.usage.input,
                    "output": message.usage.output,
                    "cache_read": message.usage.cache_read,
                    "cache_write": message.usage.cache_write,
                    "reasoning": message.usage.reasoning,
                },
            }
        case ToolResultMessage():
            return {
                "role": "tool_result",
                "content": content,
                "timestamp": message.timestamp,
                "tool_call_id": message.tool_call_id,
                "is_error": message.is_error,
            }


def decode_message(raw: dict[str, Any]) -> Message:
    """Restore a message encoded by `encode_message`."""
    content = tuple(_decode_block(block) for block in raw["content"])
    role = raw["role"]
    if role == "user":
        return UserMessage(content=content, timestamp=raw["timestamp"])
    if role == "assistant":
        usage = raw["usage"]
        return AssistantMessage(
            content=content,
            stop_reason=StopReason(raw["stop_reason"]),
            usage=Usage(
                input=usage["input"],
                output=usage["output"],
                cache_read=usage["cache_read"],
                cache_write=usage["cache_write"],
                reasoning=usage["reasoning"],
            ),
            model=raw["model"],
            provider=raw["provider"],
            timestamp=raw["timestamp"],
            error_message=raw["error_message"],
        )
    if role == "tool_result":
        return ToolResultMessage(
            tool_call_id=raw["tool_call_id"],
            content=content,
            timestamp=raw["timestamp"],
            is_error=raw["is_error"],
        )
    raise ValueError(f"unknown message role {role!r}")


def messages_from(events: tuple[SessionEvent, ...]) -> tuple[Message, ...]:
    """Decode a run of surface events into messages."""
    return tuple(decode_message(event.data["message"]) for event in events)


def _latest_of(events: tuple[SessionEvent, ...], kind: EventKind) -> SessionEvent | None:
    """The most recent event of `kind` among `events`, or None."""
    for event in reversed(events):
        if event.kind is kind:
            return event
    return None


def effective_surface(log: SessionLog) -> tuple[SessionEvent, ...]:
    """The surface entries that still participate, ignoring compaction.

    A reset excludes everything at or before it; the latest one wins.

    Public because `operations.py` needs it to record what a compaction
    supersedes — a private cross-module import would be worse than a named
    seam.
    """
    reset_event = _latest_of(log.events, EventKind.SESSION_RESET)
    floor = reset_event.seq if reset_event is not None else 0
    return tuple(event for event in log.surface() if event.seq > floor)


def _derive(log: SessionLog, limit: int) -> tuple[Message, ...]:
    """Project `log` up to sequence `limit`, applying reset and compaction.

    `limit` exists for forks: a child sees its ancestor only as far as the
    boundary fixed at fork time, so the ancestor's later writes stay private
    to it.

    Sequence numbers restart in a fork, so a compaction's `superseded_through`
    and `retained` refer to that log's *own* events. Inherited history is
    therefore dropped wholesale when a compaction or reset is active, rather
    than filtered by sequence — comparing across logs would match unrelated
    entries that happen to share a number.
    """
    events = tuple(event for event in log.events if event.seq <= limit)

    reset_event = _latest_of(events, EventKind.SESSION_RESET)
    floor = reset_event.seq if reset_event is not None else 0
    own_surface = tuple(event for event in events if is_surface(event) and event.seq > floor)

    compaction = _latest_of(events, EventKind.COMPACTION)
    if compaction is not None and compaction.seq > floor:
        superseded = compaction.data["superseded_through"]
        retained = set(compaction.data["retained"])
        summary = UserMessage(content=(TextBlock(text=compaction.data["summary"]),), timestamp=0)
        kept = tuple(
            event for event in own_surface if event.seq > superseded or event.seq in retained
        )
        return (summary, *messages_from(kept))

    inherited: tuple[Message, ...] = ()
    if reset_event is None and log.ancestor is not None:
        inherited = _derive(log.ancestor, log.boundary)

    return (*inherited, *messages_from(own_surface))


def derive_messages(log: SessionLog) -> tuple[Message, ...]:
    """Project `log` into model history, walking any ancestry it has."""
    return _derive(log, len(log))
