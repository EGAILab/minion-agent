"""Executes target-model transform (XFORM) conformance scenarios directly against the real
`transform_messages()` seam.

Deliberately independent of `agent_runner.py`/`session_runner.py`'s own thin decoders: this DSL's
message shape must, uniquely, accept `content: null` on any role (AI-026's legacy-content input,
which the real typed `Message` dataclasses cannot themselves represent) and `content: <string>` on
`user` (the first-class `UserMessage.content: string | [...]` shape, `spec/llm.md`). Each family's
runner stays thin for its own DSL rather than sharing a decoder tuned for a different one.
"""

from __future__ import annotations

import base64
from typing import Any

from minion_agent.llm.content import (
    ContentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from minion_agent.llm.messages import (
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
from minion_agent.llm.service import ModelId
from minion_agent.llm.transform_messages import NormalizeToolCallId, TargetModel, transform_messages


def _block(raw: dict[str, Any]) -> ContentBlock:
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


def _user_content(raw: Any) -> str | tuple[ContentBlock, ...] | None:
    """`UserMessage.content` is `string | tuple[ContentBlock, ...]` -- both first-class
    (`spec/llm.md`) -- plus the legacy-null input case (`AI-026`). A JSON string stays a string;
    only a JSON array is decoded into blocks."""
    if raw is None or isinstance(raw, str):
        return raw
    return tuple(_block(block) for block in raw)


def _content(raw: list[dict[str, Any]] | None) -> tuple[ContentBlock, ...] | None:
    if raw is None:
        return None
    return tuple(_block(block) for block in raw)


def _usage(raw: dict[str, Any]) -> Usage:
    """Reads a schema-validated `Usage` object directly -- no defaults, no fabricated fields
    (`XFORM-R002`). `AssistantMessage.usage` is required (`spec/llm.md`) and the schema now
    enforces every non-optional member is present too, so a schema-valid scenario always
    supplies a complete object here; `ToolResultMessage.usage` stays genuinely optional at the
    call site (`None` when the key is absent), but when present it is exercised through this
    same function and so is equally complete."""
    cost = raw["cost"]
    return Usage(
        input=raw["input"],
        output=raw["output"],
        cache_read=raw["cache_read"],
        cache_write=raw["cache_write"],
        cache_write_1h=raw.get("cache_write_1h"),
        reasoning=raw.get("reasoning"),
        total_tokens=raw["total_tokens"],
        cost=Cost(
            input=cost["input"],
            output=cost["output"],
            cache_read=cost["cache_read"],
            cache_write=cost["cache_write"],
            total=cost["total"],
        ),
    )


def _diagnostic(raw: dict[str, Any]) -> AssistantMessageDiagnostic:
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


def _deferred(raw: dict[str, Any] | None) -> DeferredHandle | None:
    if raw is None:
        return None
    return DeferredHandle(
        provider=raw["provider"],
        model_id=raw["model_id"],
        api=raw["api"],
        id=raw["id"],
        expires_at=raw.get("expires_at"),
        poll_after_ms=raw.get("poll_after_ms"),
        data=raw.get("data"),
    )


def _message(raw: dict[str, Any]) -> Message:
    role = raw["role"]
    if role == "user":
        return UserMessage(content=_user_content(raw.get("content")), timestamp=raw["timestamp"])
    if role == "assistant":
        raw_diagnostics = raw.get("diagnostics")
        return AssistantMessage(
            content=_content(raw.get("content")),  # type: ignore[arg-type]
            stop_reason=StopReason(raw["stop_reason"]),
            usage=_usage(raw["usage"]),
            model=raw["model"],
            provider=raw["provider"],
            api=raw["api"],
            timestamp=raw["timestamp"],
            response_model=raw.get("response_model"),
            response_id=raw.get("response_id"),
            diagnostics=(
                tuple(_diagnostic(d) for d in raw_diagnostics)
                if raw_diagnostics is not None
                else None
            ),
            deferred=_deferred(raw.get("deferred")),
            error_message=raw.get("error_message"),
            raw_stop_reason=raw.get("raw_stop_reason"),
            end_turn=raw.get("end_turn"),
        )
    if role == "tool_result":
        raw_usage = raw.get("usage")
        added_tool_names = raw.get("added_tool_names")
        return ToolResultMessage(
            tool_call_id=raw["tool_call_id"],
            content=_content(raw.get("content")),  # type: ignore[arg-type]
            timestamp=raw.get("timestamp", 0),
            tool_name=raw["tool_name"],
            is_error=raw["is_error"],
            details=raw.get("details"),
            usage=_usage(raw_usage) if raw_usage is not None else None,
            added_tool_names=tuple(added_tool_names) if added_tool_names is not None else None,
        )
    raise ValueError(f"unknown message role {role!r}")


def _normalize_block(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text, "text_signature": block.text_signature}
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "thinking_signature": block.thinking_signature,
            "redacted": block.redacted,
        }
    if isinstance(block, ImageBlock):
        normalized: dict[str, Any] = {"type": "image", "mime_type": block.mime_type}
        if block.reference is not None:
            normalized["reference"] = block.reference
        else:
            assert block.data is not None
            normalized["data"] = base64.b64encode(block.data).decode("ascii")
        return normalized
    return {
        "type": "tool_call",
        "id": block.id,
        "name": block.name,
        "arguments": block.arguments,
        "thought_signature": block.thought_signature,
        "namespace": block.namespace,
    }


def _normalize_user_content(content: str | tuple[ContentBlock, ...]) -> Any:
    if isinstance(content, str):
        return content
    return [_normalize_block(block) for block in content]


def _normalize_usage(usage: Usage) -> dict[str, Any]:
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


def _normalize_diagnostic(diagnostic: AssistantMessageDiagnostic) -> dict[str, Any]:
    error = diagnostic.error
    return {
        "type": diagnostic.type,
        "timestamp": diagnostic.timestamp,
        "error": (
            None
            if error is None
            else {
                "message": error.message,
                "name": error.name,
                "stack": error.stack,
                "code": error.code,
            }
        ),
        "details": diagnostic.details,
    }


def _normalize_deferred(handle: DeferredHandle | None) -> dict[str, Any] | None:
    if handle is None:
        return None
    return {
        "provider": handle.provider,
        "model_id": handle.model_id,
        "api": handle.api,
        "id": handle.id,
        "expires_at": handle.expires_at,
        "poll_after_ms": handle.poll_after_ms,
        "data": handle.data,
    }


def _normalize_message(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": _normalize_user_content(message.content),
            "timestamp": message.timestamp,
        }
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [_normalize_block(block) for block in message.content],
            "provider": message.provider,
            "api": message.api,
            "model": message.model,
            "stop_reason": message.stop_reason.value,
            "timestamp": message.timestamp,
            "usage": _normalize_usage(message.usage),
            "response_model": message.response_model,
            "response_id": message.response_id,
            "diagnostics": (
                None
                if message.diagnostics is None
                else [_normalize_diagnostic(d) for d in message.diagnostics]
            ),
            "deferred": _normalize_deferred(message.deferred),
            "error_message": message.error_message,
            "raw_stop_reason": message.raw_stop_reason,
            "end_turn": message.end_turn,
        }
    return {
        "role": "tool_result",
        "content": [_normalize_block(block) for block in message.content],
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "is_error": message.is_error,
        # timestamp is real, observable state -- always present here; a scenario's own
        # expect.messages omits it for a synthesized result (wall-clock, not a contract value)
        # and the test comparison drops it from this dict before comparing in that case --
        # see test_transform_conformance.py and spec/target-model-transformation.md.
        "timestamp": message.timestamp,
        "details": message.details,
        "usage": None if message.usage is None else _normalize_usage(message.usage),
        "added_tool_names": (
            None if message.added_tool_names is None else list(message.added_tool_names)
        ),
    }


def _normalizer(mapping: dict[str, str]) -> NormalizeToolCallId:
    def normalize(call_id: str, target: TargetModel, source: AssistantMessage) -> str:
        return mapping.get(call_id, call_id)

    return normalize


def run_transform_scenario(document: dict[str, Any]) -> dict[str, Any]:
    """Run the scenario through the real `transform_messages()` seam and return its observable
    output. Thin: parses typed values, calls the real function, normalizes the real result."""
    spec = document["transform"]
    messages = [_message(raw) for raw in spec["messages"]]
    target_spec = spec["target"]
    target = TargetModel(
        identity=ModelId(
            provider=target_spec["provider"], model=target_spec["model_id"], api=target_spec["api"]
        ),
        supports_images=target_spec["supports_images"],
    )
    normalizer = (
        _normalizer(spec["normalize_tool_call_ids"])
        if "normalize_tool_call_ids" in spec
        else None
    )

    try:
        result = transform_messages(messages, target, normalizer)
    except Exception as error:  # surfaced, not raised -- see test_transform_conformance.py
        return {"messages": None, "error": {"type": type(error).__name__, "message": str(error)}}

    return {"messages": [_normalize_message(message) for message in result], "error": None}
