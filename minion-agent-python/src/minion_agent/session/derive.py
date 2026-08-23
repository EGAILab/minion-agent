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
    AssistantMessageDiagnostic,
    Cost,
    DeferredHandle,
    DiagnosticError,
    Message,
    StopReason,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .events import EventKind, EventName, SessionEvent, is_surface
from .log import SessionLog


def _encode_block(block: ContentBlock) -> dict[str, Any]:
    match block:
        case TextBlock():
            encoded: dict[str, Any] = {"type": "text", "text": block.text}
            if block.text_signature is not None:
                encoded["text_signature"] = block.text_signature
            return encoded
        case ThinkingBlock():
            encoded = {"type": "thinking", "thinking": block.thinking}
            if block.thinking_signature is not None:
                encoded["thinking_signature"] = block.thinking_signature
            if block.redacted:
                encoded["redacted"] = True
            return encoded
        case ImageBlock():
            encoded = {"type": "image", "mime_type": block.mime_type}
            if block.reference is not None:
                encoded["reference"] = block.reference
            else:
                # Guaranteed non-None by ImageBlock.__post_init__.
                assert block.data is not None
                encoded["data"] = base64.b64encode(block.data).decode("ascii")
            return encoded
        case ToolCallBlock():
            encoded = {
                "type": "tool_call",
                "id": block.id,
                "name": block.name,
                "arguments": block.arguments,
            }
            if block.thought_signature is not None:
                encoded["thought_signature"] = block.thought_signature
            if block.namespace is not None:
                encoded["namespace"] = block.namespace
            return encoded


def _decode_block(raw: dict[str, Any]) -> ContentBlock:
    kind = raw["type"]
    if kind == "text":
        return TextBlock(text=raw["text"], text_signature=raw.get("text_signature"))
    if kind == "thinking":
        return ThinkingBlock(
            thinking=raw["thinking"],
            thinking_signature=raw.get("thinking_signature"),
            redacted=raw.get("redacted", False),
        )
    if kind == "image":
        if "reference" in raw:
            return ImageBlock(mime_type=raw["mime_type"], reference=raw["reference"])
        return ImageBlock(mime_type=raw["mime_type"], data=base64.b64decode(raw["data"]))
    if kind == "tool_call":
        return ToolCallBlock(
            id=raw["id"],
            name=raw["name"],
            arguments=raw["arguments"],
            thought_signature=raw.get("thought_signature"),
            namespace=raw.get("namespace"),
        )
    raise ValueError(f"unknown content block type {kind!r}")


def _encode_usage(usage: Usage) -> dict[str, Any]:
    return {
        "input": usage.input,
        "output": usage.output,
        "cache_read": usage.cache_read,
        "cache_write": usage.cache_write,
        "cache_write_1h": usage.cache_write_1h,
        "reasoning": usage.reasoning,
        "total_tokens": usage.total_tokens,
        "cost": {
            "input": usage.cost.input,
            "output": usage.cost.output,
            "cache_read": usage.cost.cache_read,
            "cache_write": usage.cost.cache_write,
            "total": usage.cost.total,
        },
    }


def _decode_usage(raw: dict[str, Any]) -> Usage:
    cost = raw.get("cost")
    return Usage(
        input=raw["input"],
        output=raw["output"],
        cache_read=raw["cache_read"],
        cache_write=raw["cache_write"],
        cache_write_1h=raw.get("cache_write_1h"),
        reasoning=raw["reasoning"],
        total_tokens=raw.get("total_tokens", 0),
        cost=Cost(**cost) if cost is not None else Cost(),
    )


def _encode_diagnostic(diagnostic: AssistantMessageDiagnostic) -> dict[str, Any]:
    encoded: dict[str, Any] = {"type": diagnostic.type, "timestamp": diagnostic.timestamp}
    if diagnostic.error is not None:
        error: dict[str, Any] = {"message": diagnostic.error.message}
        if diagnostic.error.name is not None:
            error["name"] = diagnostic.error.name
        if diagnostic.error.stack is not None:
            error["stack"] = diagnostic.error.stack
        if diagnostic.error.code is not None:
            error["code"] = diagnostic.error.code
        encoded["error"] = error
    if diagnostic.details is not None:
        encoded["details"] = diagnostic.details
    return encoded


def _decode_diagnostic(raw: dict[str, Any]) -> AssistantMessageDiagnostic:
    raw_error = raw.get("error")
    error = (
        DiagnosticError(
            message=raw_error["message"],
            name=raw_error.get("name"),
            stack=raw_error.get("stack"),
            code=raw_error.get("code"),
        )
        if raw_error is not None
        else None
    )
    return AssistantMessageDiagnostic(
        type=raw["type"], timestamp=raw["timestamp"], error=error, details=raw.get("details")
    )


def _encode_deferred(handle: DeferredHandle) -> dict[str, Any]:
    encoded: dict[str, Any] = {
        "provider": handle.provider,
        "model_id": handle.model_id,
        "api": handle.api,
        "id": handle.id,
    }
    if handle.expires_at is not None:
        encoded["expires_at"] = handle.expires_at
    if handle.poll_after_ms is not None:
        encoded["poll_after_ms"] = handle.poll_after_ms
    if handle.data is not None:
        encoded["data"] = handle.data
    return encoded


def _decode_deferred(raw: dict[str, Any]) -> DeferredHandle:
    return DeferredHandle(
        provider=raw["provider"],
        model_id=raw["model_id"],
        api=raw["api"],
        id=raw["id"],
        expires_at=raw.get("expires_at"),
        poll_after_ms=raw.get("poll_after_ms"),
        data=raw.get("data"),
    )


def encode_message(message: Message) -> dict[str, Any]:
    """Render `message` as JSON-safe data the log will accept."""
    content = [_encode_block(block) for block in message.content]
    match message:
        case UserMessage():
            return {"role": "user", "content": content, "timestamp": message.timestamp}
        case AssistantMessage():
            encoded_assistant: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "timestamp": message.timestamp,
                "stop_reason": message.stop_reason.value,
                "model": message.model,
                "provider": message.provider,
                "error_message": message.error_message,
                "usage": _encode_usage(message.usage),
                "api": message.api,
            }
            if message.response_model is not None:
                encoded_assistant["response_model"] = message.response_model
            if message.response_id is not None:
                encoded_assistant["response_id"] = message.response_id
            if message.diagnostics is not None:
                encoded_assistant["diagnostics"] = [
                    _encode_diagnostic(diagnostic) for diagnostic in message.diagnostics
                ]
            if message.deferred is not None:
                encoded_assistant["deferred"] = _encode_deferred(message.deferred)
            if message.raw_stop_reason is not None:
                encoded_assistant["raw_stop_reason"] = message.raw_stop_reason
            if message.end_turn is not None:
                encoded_assistant["end_turn"] = message.end_turn
            return encoded_assistant
        case ToolResultMessage():
            encoded: dict[str, Any] = {
                "role": "tool_result",
                "content": content,
                "timestamp": message.timestamp,
                "tool_call_id": message.tool_call_id,
                "is_error": message.is_error,
            }
            if message.tool_name is not None:
                encoded["tool_name"] = message.tool_name
            if message.details is not None:
                encoded["details"] = message.details
            if message.usage is not None:
                encoded["usage"] = _encode_usage(message.usage)
            if message.added_tool_names is not None:
                encoded["added_tool_names"] = list(message.added_tool_names)
            return encoded


def decode_message(raw: dict[str, Any]) -> Message:
    """Restore a message encoded by `encode_message`."""
    content = tuple(_decode_block(block) for block in raw["content"])
    role = raw["role"]
    if role == "user":
        return UserMessage(content=content, timestamp=raw["timestamp"])
    if role == "assistant":
        raw_diagnostics = raw.get("diagnostics")
        raw_deferred = raw.get("deferred")
        return AssistantMessage(
            content=content,
            stop_reason=StopReason(raw["stop_reason"]),
            usage=_decode_usage(raw["usage"]),
            model=raw["model"],
            provider=raw["provider"],
            timestamp=raw["timestamp"],
            error_message=raw["error_message"],
            api=raw.get("api", "mock"),
            response_model=raw.get("response_model"),
            response_id=raw.get("response_id"),
            diagnostics=(
                tuple(_decode_diagnostic(entry) for entry in raw_diagnostics)
                if raw_diagnostics is not None
                else None
            ),
            deferred=_decode_deferred(raw_deferred) if raw_deferred is not None else None,
            raw_stop_reason=raw.get("raw_stop_reason"),
            end_turn=raw.get("end_turn"),
        )
    if role == "tool_result":
        raw_usage = raw.get("usage")
        added_tool_names = raw.get("added_tool_names")
        return ToolResultMessage(
            tool_call_id=raw["tool_call_id"],
            content=content,
            timestamp=raw["timestamp"],
            is_error=raw["is_error"],
            tool_name=raw.get("tool_name"),
            details=raw.get("details"),
            usage=_decode_usage(raw_usage) if raw_usage is not None else None,
            added_tool_names=tuple(added_tool_names) if added_tool_names is not None else None,
        )
    raise ValueError(f"unknown message role {role!r}")


def messages_from(events: tuple[SessionEvent, ...]) -> tuple[Message, ...]:
    """Decode a run of surface events into messages."""
    return tuple(decode_message(event.data["message"]) for event in events)


def _latest_of(events: tuple[SessionEvent, ...], kind: EventName) -> SessionEvent | None:
    """The most recent event named `kind` among `events`, or None.

    Compared by value, not identity. The event name is the language-neutral
    identity, so `"session/reset"` and `EventKind.SESSION_RESET` are the same
    event — an identity check would silently ignore the former, and a second
    implementation comparing strings would disagree with this one.
    """
    for event in reversed(events):
        if event.kind == kind:
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
    # `log.surface_kinds`, not the core set: a plugin-declared surface event
    # must derive here exactly as a core one does (design spec section 5).
    own_surface = tuple(
        event for event in events if is_surface(event, log.surface_kinds) and event.seq > floor
    )

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
