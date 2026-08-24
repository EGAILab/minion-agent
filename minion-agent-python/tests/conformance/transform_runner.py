"""Executes target-model transform (XFORM) conformance scenarios directly against the real
`transform_messages()` seam.

Deliberately independent of `agent_runner.py`/`session_runner.py`'s own thin decoders: this DSL's
message shape is intentionally narrower (no usage/diagnostics/deferred -- XFORM's own rules never
touch them) and, uniquely, must accept `content: null` on input to script AI-026's legacy-content
case, which the real typed `Message` dataclasses cannot themselves represent. Each family's runner
stays thin for its own DSL rather than sharing a decoder tuned for a different one.
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


def _content(raw: list[dict[str, Any]] | None) -> tuple[ContentBlock, ...] | None:
    if raw is None:
        return None
    return tuple(_block(block) for block in raw)


def _message(raw: dict[str, Any]) -> Message:
    role = raw["role"]
    content = _content(raw.get("content"))
    if role == "user":
        return UserMessage(content=content, timestamp=raw["timestamp"])  # type: ignore[arg-type]
    if role == "assistant":
        return AssistantMessage(
            content=content,  # type: ignore[arg-type]
            stop_reason=StopReason(raw["stop_reason"]),
            usage=Usage(),
            model=raw["model"],
            provider=raw["provider"],
            api=raw["api"],
            timestamp=raw["timestamp"],
        )
    if role == "tool_result":
        return ToolResultMessage(
            tool_call_id=raw["tool_call_id"],
            content=content,  # type: ignore[arg-type]
            timestamp=raw.get("timestamp", 0),
            tool_name=raw["tool_name"],
            is_error=raw["is_error"],
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


def _normalize_message(message: Message) -> dict[str, Any]:
    content = [_normalize_block(block) for block in message.content]
    if isinstance(message, UserMessage):
        return {"role": "user", "content": content, "timestamp": message.timestamp}
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": content,
            "provider": message.provider,
            "api": message.api,
            "model": message.model,
            "stop_reason": message.stop_reason.value,
            "timestamp": message.timestamp,
        }
    return {
        "role": "tool_result",
        "content": content,
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "is_error": message.is_error,
        # timestamp deliberately excluded: a synthesized orphan result's timestamp is real
        # wall-clock time (matching Pi's Date.now()), not an observable XFORM contract value --
        # see spec/target-model-transformation.md.
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
