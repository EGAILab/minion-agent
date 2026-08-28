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

from ..llm import AssistantMessage, Message
from ..session import EventKind, SessionEvent, SessionLog, decode_message


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


@dataclass(frozen=True, slots=True)
class MessageUpdate:
    """One streaming update. Pi's tenth event, `{assistantMessageEvent, message}`:
    `message` is the FULL partial assistant message as accumulated so far (Layer 08, PASS 3,
    `L08-R003` -- an earlier revision reconstructed only accumulated text and dropped thinking/
    tool-call partial content entirely, when the certified Layer-02/04 `StreamChunk` already
    carries the complete partial on every variant). `kind`/`content_index` are Minion's own
    decomposition of pi's raw `assistantMessageEvent`, kept for existing diagnostic value."""

    kind: str
    content_index: int
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
    tool_call_id: str
    is_error: bool


type AgentEvent = (
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageUpdate
    | MessageStart
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionEnd
)

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
            events.append(
                ToolExecutionEnd(
                    tool_call_id=entry.data["message"]["tool_call_id"],
                    is_error=entry.data["message"]["is_error"],
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
            events.append(
                MessageUpdate(
                    kind=entry.data["kind"],
                    content_index=entry.data["content_index"],
                    message=decode_message(entry.data["partial"]),
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
