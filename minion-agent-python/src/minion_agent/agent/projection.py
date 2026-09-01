"""Pi's observable event stream, rebuilt from the log.

The log is the source of truth (design spec section 5); Pi's `AgentEvent`
union is a *derived view* of it. Conformance asserts this projection, which is
what keeps Pi's observable semantics pinned while the internals follow DSH.

Event kinds are compared by value throughout. The name string is the
language-neutral identity (section 5), so a log written with raw strings must
project exactly as one written with the core constants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import (
    AssistantMessage,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
)
from ..llm.stream import (
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from ..session import EventKind, SessionEvent, SessionLog, decode_message
from ..tools.result import ToolResult


@dataclass(frozen=True, slots=True)
class AgentStart:
    """One pi-equivalent run opens (`AGENT_START` in the log). `causes` is a
    disclosed Minion enrichment -- pinned Pi's own `agent_start` carries no
    fields at all -- naming whatever queued input triggered this run."""

    causes: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class AgentEnd:
    """One pi-equivalent run closes (`AGENT_END` in the log).

    `messages` is pi's own `agent_end.messages`: invocation-local, not the
    whole transcript -- every message this run itself produced or consumed,
    in order (the initial claimed batch, then everything appended by each
    turn). `reason`/`causes` are disclosed Minion enrichments beyond pi's own
    bare `{messages}` shape.
    """

    reason: str
    causes: tuple[dict[str, Any], ...] = ()
    messages: tuple[Message, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnStart:
    """One turn opens (`TURN_START` in the log). Pinned Pi's own `turn_start`
    carries no fields at all, and this projection carries none either --
    `causes` belongs to the run that triggered it (`AgentStart`), not to each
    individual turn within that run."""


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """One turn closes (`TURN_END` in the log): pinned Pi's own
    `turn_end{message, toolResults}`, reconstructed from the `MessageStart`/
    `MessageEnd` pairs this same turn already emitted -- never a second,
    independently-logged copy of the same content."""

    message: Message | None = None
    tool_results: tuple[Message, ...] = ()


type AssistantMessageEvent = (
    TextStart
    | TextDelta
    | TextEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
)
"""Pinned Pi's own raw `assistantMessageEvent` shape, verbatim -- the certified Layer-02/04
`StreamChunk` variant that produced this update, carrying its own type-specific fields (a delta
string, the finalized text/thinking/tool-call block, etc.), not a normalized label."""


@dataclass(frozen=True, slots=True)
class MessageUpdate:
    """One streaming update. Pi's tenth event, `{assistantMessageEvent, message}` (Layer 08, PASS 6,
    `L08-R002`): `event` is the COMPLETE, type-specific stream chunk pinned Pi's own
    `assistantMessageEvent` carries -- an earlier revision normalized it into a bare `kind` string
    and `content_index`, dropping the delta/text/thinking/tool-call payload those events actually
    carry. `message` is the FULL partial assistant message as accumulated so far (`L08-R003`)."""

    event: AssistantMessageEvent
    message: Message


@dataclass(frozen=True, slots=True)
class MessageStart:
    message: Message


@dataclass(frozen=True, slots=True)
class MessageEnd:
    message: Message


@dataclass(frozen=True, slots=True)
class ToolExecutionStart:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolExecutionEnd:
    """Pinned Pi's own `tool_execution_end` shape (Layer 08, PASS 6, `L08-R002`): `tool_name` and
    the finalized `result` itself, not merely `is_error` derived from it -- an earlier revision
    exposed only `tool_call_id`/`is_error`, a real payload reduction pinned Pi does not have."""

    tool_call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool


@dataclass(frozen=True, slots=True)
class ToolExecutionUpdate:
    """Pinned Pi's own `tool_execution_update` shape (Layer 08, PASS 6, `L08-R002`; new event,
    previously missing from the union entirely): a tool's own live partial-output report, reaching
    the unified Agent-event seam the same way pinned Pi's does. `arguments` is the ORIGINAL,
    pre-`prepare_arguments`/validation call arguments, matching pinned Pi's own
    `PreparedToolCall.toolCall.arguments` exactly (already-certified Layer-06 rule,
    `IR-L06-005`). `partial_result` is a `ToolResult` (Layer 08, `L08-R011`) -- pinned Pi's own
    `AgentToolUpdateCallback<T>` carries `partialResult: AgentToolResult<T>`, the SAME structured
    shape a tool's own final result is; an earlier revision narrowed this field to `str`, a real
    payload reduction pinned Pi does not have."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    partial_result: ToolResult


type AgentEvent = (
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageUpdate
    | MessageStart
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionUpdate
    | ToolExecutionEnd
)


def _tool_result_from_message(message: ToolResultMessage, *, terminate: bool) -> ToolResult:
    """Rebuild the Layer-06 `ToolResult` a `TOOL_RESULT` log entry's own `ToolResultMessage` was
    projected from (Layer 08, PASS 6, `L08-R002`): `ToolResult.to_message()` (`tools/result.py`)
    copies every field onto `ToolResultMessage` verbatim except `terminate`, which never reaches
    the model by design -- the one extra field `_run_step` now logs alongside the message for
    exactly this reconstruction. `details or {}`/`added_tool_names or ()` restore each field's
    own `ToolResult` default; every other field round-trips exactly."""
    return ToolResult(
        tool_call_id=message.tool_call_id,
        content=message.content,
        tool_name=message.tool_name,
        is_error=message.is_error,
        details=message.details or {},
        terminate=terminate,
        added_tool_names=tuple(message.added_tool_names) if message.added_tool_names else (),
        usage=message.usage,
    )


def _assistant_message_event(
    kind: str, content_index: int, message: AssistantMessage, delta: str | None
) -> AssistantMessageEvent:
    """Rebuild pinned Pi's own raw `assistantMessageEvent` from an `ASSISTANT_CHUNK` log entry
    (Layer 08, PASS 6, `L08-R002`). A "start" chunk needs nothing beyond `content_index`/
    `partial`; a "delta" chunk's own incremental string is not recoverable from `partial` alone
    (only the accumulated total is), so `_run_step` logs it explicitly; an "end" chunk's own
    finalized text/thinking/tool-call is already present in `partial.content[content_index]`,
    the same completed block the certified `StreamChunk` variant itself carried."""
    block = message.content[content_index]
    if kind == "text_start":
        return TextStart(content_index=content_index, partial=message)
    if kind == "text_delta":
        assert delta is not None
        return TextDelta(content_index=content_index, delta=delta, partial=message)
    if kind == "text_end":
        assert isinstance(block, TextBlock)
        return TextEnd(content_index=content_index, text=block.text, partial=message)
    if kind == "thinking_start":
        return ThinkingStart(content_index=content_index, partial=message)
    if kind == "thinking_delta":
        assert delta is not None
        return ThinkingDelta(content_index=content_index, delta=delta, partial=message)
    if kind == "thinking_end":
        assert isinstance(block, ThinkingBlock)
        return ThinkingEnd(content_index=content_index, thinking=block.thinking, partial=message)
    if kind == "toolcall_start":
        return ToolCallStart(content_index=content_index, partial=message)
    if kind == "toolcall_delta":
        assert delta is not None
        return ToolCallDelta(content_index=content_index, delta=delta, partial=message)
    assert kind == "toolcall_end"
    assert isinstance(block, ToolCallBlock)
    return ToolCallEnd(content_index=content_index, tool_call=block, partial=message)


_MESSAGE_KINDS = frozenset({EventKind.USER_MESSAGE, EventKind.TOOL_RESULT})
"""`ASSISTANT_MESSAGE` is handled in its own branch (Layer 08, PASS 3): whether it gets its own
`MessageStart` depends on whether an `ASSISTANT_STREAM_START` already opened it, which this
generic per-entry handling cannot express."""


def project(log: SessionLog) -> tuple[AgentEvent, ...]:
    """Rebuild the Pi event stream from `log`.

    `AgentStart`/`AgentEnd` and `TurnStart`/`TurnEnd` are driven entirely by
    real `AGENT_START`/`AGENT_END`/`TURN_START`/`TURN_END` log entries, never
    synthesized around the whole log -- a log with no run in it projects to
    nothing at all, and a log holding several runs projects several
    `AgentStart`/`AgentEnd` brackets, each scoped to its own run.

    Two emission orders coexist (design spec section 6): `tool_execution_end`
    in completion order, message events in source order. `tool/result` log
    entries are always appended in source order, so a contiguous run of them
    is buffered here and re-sorted by `completion_index` before its
    `ToolExecutionEnd` events are emitted; the `MessageStart`/`MessageEnd`
    pairs that follow still emit in the original, source, order.

    `TURN_END`/`AGENT_END` entries may carry explicit `"message"`/`"messages"`
    overrides (Layer 08, PASS 3, `L08-R002`): pinned Pi's `handleRunFailure`
    fallback emits `turn_end(failure, [])` / `agent_end(messages=[failure])`
    -- the failure message ONLY, not whatever this run's own turn/run
    accumulators happen to hold -- and does so without a preceding
    `turn_start` at all. Fixing the projection to honor an explicit override
    (rather than inventing a synthetic `turn_start` pinned Pi never emits)
    is the sanctioned fix; the accumulators' own normal behavior is
    unaffected for every other `TURN_END`/`AGENT_END`.
    """
    events: list[AgentEvent] = []
    # Buffered `tool/result` entries of the run currently being accumulated.
    pending_results: list[SessionEvent] = []
    # Every message this run has produced/consumed so far, in order -- pi's
    # own `agent_end.messages` (invocation-local, reset at each AGENT_START).
    run_messages: list[Message] = []
    # This turn's own finalized assistant reply and tool results -- pi's own
    # `turn_end{message, toolResults}` (reset at each TURN_START).
    turn_message: Message | None = None
    turn_results: list[Message] = []
    # Whether ASSISTANT_STREAM_START already opened a MessageStart for the
    # assistant reply currently streaming -- ASSISTANT_MESSAGE must not open
    # a second one when it did (Layer 08, PASS 3, L08-R003).
    stream_open = False

    def emit_message(message: Message, *, suppress_start: bool = False) -> None:
        nonlocal turn_message
        if not suppress_start:
            events.append(MessageStart(message=message))
        events.append(MessageEnd(message=message))
        run_messages.append(message)
        if isinstance(message, AssistantMessage):
            turn_message = message

    def flush_results() -> None:
        for entry in sorted(pending_results, key=lambda e: e.data.get("completion_index", 0)):
            message = decode_message(entry.data["message"])
            assert isinstance(message, ToolResultMessage)
            events.append(
                ToolExecutionEnd(
                    tool_call_id=message.tool_call_id,
                    tool_name=message.tool_name,
                    result=_tool_result_from_message(
                        message, terminate=entry.data.get("terminate", False)
                    ),
                    is_error=message.is_error,
                )
            )
        for entry in pending_results:
            message = decode_message(entry.data["message"])
            emit_message(message)
            turn_results.append(message)
        pending_results.clear()

    for entry in log.events:
        if entry.kind != EventKind.TOOL_RESULT:
            flush_results()

        if entry.kind == EventKind.AGENT_START:
            run_messages = []
            events.append(AgentStart(causes=tuple(entry.data.get("causes", ()))))

        elif entry.kind == EventKind.AGENT_END:
            if "messages" in entry.data:
                # Pi's handleRunFailure: agent_end.messages is [failureMessage] only.
                messages = tuple(decode_message(m) for m in entry.data["messages"])
            else:
                messages = tuple(run_messages)
            events.append(
                AgentEnd(
                    reason=entry.data.get("reason", "completed"),
                    causes=tuple(entry.data.get("causes", ())),
                    messages=messages,
                )
            )

        elif entry.kind == EventKind.TURN_START:
            turn_message, turn_results = None, []
            events.append(TurnStart())

        elif entry.kind == EventKind.TURN_END:
            if "message" in entry.data:
                # Pi's handleRunFailure: turn_end(failure, []) -- no preceding
                # turn_start at all; the failure message, not this turn's
                # own (possibly nonexistent) accumulator.
                override = entry.data["message"]
                message = decode_message(override) if override is not None else None
                events.append(TurnEnd(message=message, tool_results=()))
            else:
                events.append(TurnEnd(message=turn_message, tool_results=tuple(turn_results)))

        elif entry.kind == EventKind.ASSISTANT_STREAM_START:
            message = decode_message(entry.data["partial"])
            events.append(MessageStart(message=message))
            stream_open = True

        elif entry.kind == EventKind.ASSISTANT_CHUNK:
            partial_message = decode_message(entry.data["partial"])
            assert isinstance(partial_message, AssistantMessage)
            events.append(
                MessageUpdate(
                    event=_assistant_message_event(
                        entry.data["kind"],
                        entry.data["content_index"],
                        partial_message,
                        entry.data.get("delta"),
                    ),
                    message=partial_message,
                )
            )

        elif entry.kind == EventKind.ASSISTANT_MESSAGE:
            emit_message(decode_message(entry.data["message"]), suppress_start=stream_open)
            stream_open = False

        elif entry.kind == EventKind.TOOL_CALL:
            events.append(
                ToolExecutionStart(
                    tool_call_id=entry.data["id"],
                    tool_name=entry.data["name"],
                    arguments=entry.data.get("arguments", {}),
                )
            )

        elif entry.kind == EventKind.TOOL_RESULT:
            # Buffered rather than emitted inline: this run's
            # ToolExecutionEnd events must sort by completion_index, which
            # is only knowable once the whole run has been collected.
            pending_results.append(entry)

        elif entry.kind in _MESSAGE_KINDS:
            emit_message(decode_message(entry.data["message"]))

    flush_results()
    return tuple(events)


def event_names(events: tuple[AgentEvent, ...]) -> list[str]:
    """Snake-case names, the form conformance scenarios assert against."""
    return [re.sub(r"(?<!^)(?=[A-Z])", "_", type(event).__name__).lower() for event in events]
