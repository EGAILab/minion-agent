"""Pi's event stream, rebuilt from the log."""

from minion_agent.agent.projection import (
    AgentEnd,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
    project,
)
from minion_agent.llm import TextBlock, UserMessage
from minion_agent.session import EventKind, SessionLog, encode_message


def _log_turn(*, with_tool: bool = False) -> SessionLog:
    log = SessionLog("s1")
    log.append(EventKind.TURN_START, {"causes": []})
    log.append(
        EventKind.USER_MESSAGE,
        {"message": encode_message(UserMessage((TextBlock(text="hi"),), 1))},
    )
    log.append(EventKind.STEP_START, {"reason": "initial"})
    if with_tool:
        log.append(EventKind.TOOL_CALL, {"id": "t1", "name": "echo", "arguments": {}})
        log.append(
            EventKind.TOOL_RESULT,
            {
                "message": {
                    "role": "tool_result",
                    "content": [{"type": "text", "text": "ok"}],
                    "timestamp": 0,
                    "tool_call_id": "t1",
                    "is_error": False,
                }
            },
        )
    log.append(EventKind.STEP_END, {})
    log.append(EventKind.TURN_END, {"reason": "completed", "causes": []})
    return log


def test_a_projection_is_bracketed_by_agent_start_and_end() -> None:
    events = project(_log_turn())

    assert isinstance(events[0], AgentStart)
    assert isinstance(events[-1], AgentEnd)


def test_turns_are_bracketed() -> None:
    events = project(_log_turn())
    kinds = [type(event) for event in events]

    assert kinds.index(TurnStart) < kinds.index(TurnEnd)


def test_messages_are_bracketed_by_start_and_end() -> None:
    events = project(_log_turn())

    assert any(isinstance(event, MessageStart) for event in events)
    assert any(isinstance(event, MessageEnd) for event in events)


def test_tool_execution_is_bracketed() -> None:
    events = project(_log_turn(with_tool=True))
    kinds = [type(event) for event in events]

    assert kinds.index(ToolExecutionStart) < kinds.index(ToolExecutionEnd)


def test_a_tool_execution_event_carries_its_call_id() -> None:
    events = project(_log_turn(with_tool=True))

    start = next(e for e in events if isinstance(e, ToolExecutionStart))
    assert (start.tool_call_id, start.tool_name) == ("t1", "echo")


def test_an_empty_log_projects_to_a_bare_bracket() -> None:
    assert [type(e) for e in project(SessionLog("s1"))] == [AgentStart, AgentEnd]


def test_turn_end_carries_its_causes() -> None:
    log = SessionLog("s1")
    log.append(EventKind.TURN_START, {"causes": [{"id": "e1", "origin": "matrix"}]})
    log.append(
        EventKind.TURN_END,
        {"reason": "completed", "causes": [{"id": "e1", "origin": "matrix"}]},
    )

    end = next(e for e in project(log) if isinstance(e, TurnEnd))

    assert end.causes[0]["origin"] == "matrix"


def test_a_chunk_projects_to_a_message_update() -> None:
    """Pi's tenth event, and the reason chunks are logged at all."""
    log = SessionLog("s1")
    log.append(EventKind.ASSISTANT_CHUNK, {"kind": "text", "content_index": 0, "delta": "hi"})

    update = next(e for e in project(log) if isinstance(e, MessageUpdate))

    assert (update.kind, update.delta) == ("text", "hi")


def test_updates_precede_the_message_they_assemble() -> None:
    log = SessionLog("s1")
    log.append(EventKind.ASSISTANT_CHUNK, {"kind": "text", "content_index": 0, "delta": "hi"})
    log.append(
        EventKind.ASSISTANT_MESSAGE,
        {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "timestamp": 0,
                "stop_reason": "stop",
                "model": "m",
                "provider": "p",
                "error_message": None,
                "usage": {
                    "input": 0,
                    "output": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "reasoning": 0,
                },
            }
        },
    )

    kinds = [type(e) for e in project(log)]

    assert kinds.index(MessageUpdate) < kinds.index(MessageStart)


def _tool_result_entry(*, call_id: str, text: str, completion_index: int) -> dict:
    return {
        "message": {
            "role": "tool_result",
            "content": [{"type": "text", "text": text}],
            "timestamp": 0,
            "tool_call_id": call_id,
            "is_error": False,
        },
        "completion_index": completion_index,
    }


def test_tool_execution_end_follows_completion_order_not_source_order() -> None:
    """t1 is requested first but finishes second.

    ToolExecutionEnd must still report [t2, t1] -- completion order -- while
    the result messages stay in [t1, t2] source order. Can fail: an
    implementation that emits ToolExecutionEnd inline while walking the log
    (source order) produces [t1, t2] here and this assertion fails.
    """
    log = SessionLog("s1")
    log.append(EventKind.TOOL_CALL, {"id": "t1", "name": "slow", "arguments": {}})
    log.append(EventKind.TOOL_CALL, {"id": "t2", "name": "quick", "arguments": {}})
    log.append(
        EventKind.TOOL_RESULT,
        _tool_result_entry(call_id="t1", text="slow", completion_index=1),
    )
    log.append(
        EventKind.TOOL_RESULT,
        _tool_result_entry(call_id="t2", text="quick", completion_index=0),
    )

    events = project(log)

    ends = [e for e in events if isinstance(e, ToolExecutionEnd)]
    assert [e.tool_call_id for e in ends] == ["t2", "t1"]

    starts = [e for e in events if isinstance(e, MessageStart)]
    assert [e.message.tool_call_id for e in starts] == ["t1", "t2"]  # type: ignore[attr-defined]


def test_a_result_run_ending_the_log_is_not_dropped() -> None:
    """The buffer must flush even when nothing follows the last tool/result.

    Can fail: a fix that only flushes on encountering a later non-result
    entry (and not after the loop) would drop this run's events entirely.
    """
    log = SessionLog("s1")
    log.append(EventKind.TOOL_CALL, {"id": "t1", "name": "echo", "arguments": {}})
    log.append(
        EventKind.TOOL_RESULT,
        _tool_result_entry(call_id="t1", text="ok", completion_index=0),
    )

    events = project(log)

    assert any(isinstance(e, ToolExecutionEnd) for e in events)
    assert isinstance(events[-1], AgentEnd)


def test_a_missing_completion_index_defaults_rather_than_raising() -> None:
    """A log entry written before `completion_index` existed still projects.

    Can fail: reading `entry.data["completion_index"]` (no default) raises
    KeyError on this log instead of projecting deterministically.
    """
    log = SessionLog("s1")
    log.append(EventKind.TOOL_CALL, {"id": "t1", "name": "echo", "arguments": {}})
    log.append(
        EventKind.TOOL_RESULT,
        {
            "message": {
                "role": "tool_result",
                "content": [{"type": "text", "text": "ok"}],
                "timestamp": 0,
                "tool_call_id": "t1",
                "is_error": False,
            }
        },
    )

    events = project(log)

    ends = [e for e in events if isinstance(e, ToolExecutionEnd)]
    assert [e.tool_call_id for e in ends] == ["t1"]


def test_raw_string_kinds_project_identically_to_the_constants() -> None:
    """The name is the identity: an identity check would silently drop these."""
    log = SessionLog("s1")
    log.append("turn/start", {"causes": []})
    log.append("tool/call", {"id": "t1", "name": "echo", "arguments": {}})
    log.append("turn/end", {"reason": "completed", "causes": []})

    assert [type(e) for e in project(log)] == [
        AgentStart,
        TurnStart,
        ToolExecutionStart,
        TurnEnd,
        AgentEnd,
    ]
