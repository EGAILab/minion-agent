"""Executes session conformance scenarios.

Log operations in, derived messages out, with no model in play — which is what
makes derivation after fork, reset, and repeated compaction assertable without
a provider. `record_header` steps drive the real content-addressed store the
same way, with no model in play either.
"""

from __future__ import annotations

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
    text_of,
)
from minion_agent.llm.tools import ToolSchema
from minion_agent.session.artifacts import ArtifactStore
from minion_agent.session.derive import derive_messages, encode_message
from minion_agent.session.events import CORE_SURFACE_KINDS, EventKind
from minion_agent.session.log import SessionLog
from minion_agent.session.operations import compact, fork, reset
from minion_agent.session.request_header import reconstruct_header, reconstruct_tools, record_header

_KIND = {
    "user": EventKind.USER_MESSAGE,
    "assistant": EventKind.ASSISTANT_MESSAGE,
    "tool_result": EventKind.TOOL_RESULT,
}


def _block(spec: dict[str, Any]) -> ContentBlock:
    """Build a real content block from scenario data. Deserializes only --
    no derived/computed values, matching the agent-family DSL's own rule."""
    kind = spec["type"]
    if kind == "text":
        return TextBlock(text=spec["text"], text_signature=spec.get("text_signature"))
    if kind == "thinking":
        return ThinkingBlock(
            thinking=spec["thinking"],
            thinking_signature=spec.get("thinking_signature"),
            redacted=spec.get("redacted", False),
        )
    if kind == "image":
        if "reference" in spec:
            return ImageBlock(mime_type=spec["mime_type"], reference=spec["reference"])
        return ImageBlock(mime_type=spec["mime_type"], data=spec["data"].encode("utf-8"))
    if kind == "tool_call":
        return ToolCallBlock(
            id=spec["id"],
            name=spec["name"],
            arguments=spec.get("arguments", {}),
            thought_signature=spec.get("thought_signature"),
            namespace=spec.get("namespace"),
        )
    raise ValueError(f"unknown content block type {kind!r}")


def _usage(spec: dict[str, Any] | None) -> Usage:
    if spec is None:
        return Usage()
    cost = spec.get("cost")
    return Usage(
        input=spec.get("input", 0),
        output=spec.get("output", 0),
        cache_read=spec.get("cache_read", 0),
        cache_write=spec.get("cache_write", 0),
        cache_write_1h=spec.get("cache_write_1h"),
        reasoning=spec.get("reasoning"),
        total_tokens=spec.get("total_tokens", 0),
        cost=Cost(**cost) if cost is not None else Cost(),
    )


def _diagnostic(spec: dict[str, Any]) -> AssistantMessageDiagnostic:
    raw_error = spec.get("error")
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
        type=spec["type"], timestamp=spec["timestamp"], error=error, details=spec.get("details")
    )


def _deferred(spec: dict[str, Any] | None) -> DeferredHandle | None:
    if spec is None:
        return None
    return DeferredHandle(
        provider=spec["provider"],
        model_id=spec["model_id"],
        api=spec["api"],
        id=spec["id"],
        expires_at=spec.get("expires_at"),
        poll_after_ms=spec.get("poll_after_ms"),
        data=spec.get("data"),
    )


def _content(spec: dict[str, Any]) -> tuple[ContentBlock, ...]:
    if "content" in spec:
        return tuple(_block(item) for item in spec["content"])
    if "text" in spec:
        return (TextBlock(text=spec["text"]),)
    return ()


def _message(role: str, spec: dict[str, Any]) -> Message:
    """Build the message a role carries, threading every scriptable field.

    A plugin-declared kind builds as a user message: the payload shape is the
    core vocabulary, and only the event *name* is new.
    """
    content = _content(spec)
    if role not in _KIND or role == "user":
        return UserMessage(content=content, timestamp=1)
    if role == "assistant":
        raw_diagnostics = spec.get("diagnostics")
        return AssistantMessage(
            content=content,
            stop_reason=StopReason(spec.get("stop_reason", "stop")),
            usage=_usage(spec.get("usage")),
            model="mock-1",
            provider="mock",
            timestamp=1,
            api=spec.get("api", "mock"),
            response_model=spec.get("response_model"),
            response_id=spec.get("response_id"),
            diagnostics=(
                tuple(_diagnostic(d) for d in raw_diagnostics)
                if raw_diagnostics is not None
                else None
            ),
            deferred=_deferred(spec.get("deferred")),
            raw_stop_reason=spec.get("raw_stop_reason"),
            end_turn=spec.get("end_turn"),
        )
    return ToolResultMessage(
        tool_call_id="t1",
        content=content,
        timestamp=1,
        tool_name=spec.get("tool_name"),
        details=spec.get("details"),
        usage=_usage(spec["usage"]) if "usage" in spec else None,
        added_tool_names=(
            tuple(spec["added_tool_names"]) if "added_tool_names" in spec else None
        ),
    )


def _role_of(message: Message) -> str:
    return {
        UserMessage: "user",
        AssistantMessage: "assistant",
        ToolResultMessage: "tool_result",
    }[type(message)]


def _normalize_block(block: ContentBlock) -> dict[str, Any]:
    """Read a real content block's attributes. No synthesis."""
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
        encoded: dict[str, Any] = {"type": "image", "mime_type": block.mime_type}
        if block.reference is not None:
            encoded["reference"] = block.reference
        else:
            assert block.data is not None
            encoded["data"] = block.data.decode("utf-8")
        return encoded
    assert isinstance(block, ToolCallBlock)
    return {
        "type": "tool_call",
        "id": block.id,
        "name": block.name,
        "arguments": block.arguments,
        "thought_signature": block.thought_signature,
        "namespace": block.namespace,
    }


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


def _assistant_detail(message: AssistantMessage) -> dict[str, Any]:
    return {
        "content": [_normalize_block(block) for block in message.content],
        "stop_reason": message.stop_reason.value,
        "usage": _normalize_usage(message.usage),
        "api": message.api,
        "response_model": message.response_model,
        "response_id": message.response_id,
        "diagnostics": (
            None
            if message.diagnostics is None
            else [_normalize_diagnostic(d) for d in message.diagnostics]
        ),
        "deferred": _normalize_deferred(message.deferred),
        "raw_stop_reason": message.raw_stop_reason,
        "end_turn": message.end_turn,
    }


def _tool_result_detail(message: ToolResultMessage) -> dict[str, Any]:
    return {
        "content": [_normalize_block(block) for block in message.content],
        "is_error": message.is_error,
        "tool_name": message.tool_name,
        "details": message.details,
        "usage": None if message.usage is None else _normalize_usage(message.usage),
        "added_tool_names": (
            None if message.added_tool_names is None else list(message.added_tool_names)
        ),
    }


def run_session_scenario(document: dict[str, Any]) -> dict[str, Any]:
    """Apply the scenario's steps and return every observable this DSL exposes."""
    surface = CORE_SURFACE_KINDS | frozenset(document.get("surface_kinds", ()))
    log = SessionLog("scenario", surface_kinds=surface)
    store = ArtifactStore()
    forks = 0
    last_header: tuple[Any, ArtifactStore] | None = None

    for step in document["steps"]:
        if "append" in step:
            spec = step["append"]
            role = spec["role"]
            message = _message(role, spec)
            # A core role encodes through the real vocabulary codec, exactly
            # as production code does; a plugin-declared kind carries the
            # same payload shape under its own event name.
            log.append(_KIND.get(role, role), {"message": encode_message(message)})
        elif "record_header" in step:
            spec = step["record_header"]
            tools = tuple(
                ToolSchema(name=t["name"], description=t["description"], parameters=t["parameters"])
                for t in spec.get("tools", ())
            )
            event = record_header(log, store, spec["components"], model=spec["model"], tools=tools)
            last_header = (event, store)
        elif "fork" in step:
            forks += 1
            log = fork(log, f"fork-{forks}", at=step["fork"].get("at"))
        elif "reset" in step:
            reset(log)
        elif "compact" in step:
            spec = step["compact"]
            compact(log, summary=spec["summary"], keep=spec.get("keep", 0))
        # "derive" is a no-op marker: derivation happens once, at the end.

    messages = derive_messages(log)
    result: dict[str, Any] = {
        "messages": [{"role": _role_of(m), "text": text_of(m)} for m in messages],
        "assistant_details": [
            _assistant_detail(m) for m in messages if isinstance(m, AssistantMessage)
        ],
        "tool_result_details": [
            _tool_result_detail(m) for m in messages if isinstance(m, ToolResultMessage)
        ],
        "artifact_count": len(store),
    }
    if last_header is not None:
        event, header_store = last_header
        result["reconstructed_header"] = {
            "components": reconstruct_header(event, header_store),
            "tools": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in reconstruct_tools(event, header_store)
            ],
        }
    return result
