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
from minion_agent.llm import AssistantMessage, StopReason, TextBlock, Usage, UserMessage
from minion_agent.llm.stream import TextDelta
from minion_agent.session import EventKind, SessionLog, encode_message


def _log_run(*, with_tool: bool = False) -> SessionLog:
    log = SessionLog("s1")
    log.append(EventKind.AGENT_START, {"causes": []})
    log.append(EventKind.TURN_START, {"reason": "initial"})
    log.append(
        EventKind.USER_MESSAGE,
        {"message": encode_message(UserMessage((TextBlock(text="hi"),), 1))},
    )
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
                    "tool_name": "echo",
                    "is_error": False,
                }
            },
        )
    log.append(EventKind.TURN_END, {})
    log.append(EventKind.AGENT_END, {"reason": "completed", "causes": []})
    return log


def test_a_projection_is_bracketed_by_agent_start_and_end() -> None:
    events = project(_log_run())

    assert isinstance(events[0], AgentStart)
    assert isinstance(events[-1], AgentEnd)


def test_turns_are_bracketed() -> None:
    events = project(_log_run())
    kinds = [type(event) for event in events]

    assert kinds.index(TurnStart) < kinds.index(TurnEnd)


def test_messages_are_bracketed_by_start_and_end() -> None:
    events = project(_log_run())

    assert any(isinstance(event, MessageStart) for event in events)
    assert any(isinstance(event, MessageEnd) for event in events)


def test_tool_execution_is_bracketed() -> None:
    events = project(_log_run(with_tool=True))
    kinds = [type(event) for event in events]

    assert kinds.index(ToolExecutionStart) < kinds.index(ToolExecutionEnd)


def test_a_tool_execution_event_carries_its_call_id() -> None:
    events = project(_log_run(with_tool=True))

    start = next(e for e in events if isinstance(e, ToolExecutionStart))
    assert (start.tool_call_id, start.tool_name) == ("t1", "echo")


def test_an_empty_log_projects_to_nothing() -> None:
    """No `AGENT_START`/`AGENT_END` in the log means no run happened at all --
    the projection is not synthesized around whatever the log happens to
    contain, only driven by real run-boundary events."""
    assert project(SessionLog("s1")) == ()


def test_a_log_with_no_run_boundary_projects_to_nothing_even_with_content() -> None:
    """Turn/message events with no enclosing `AGENT_START`/`AGENT_END` still
    project their own content -- there is simply no `AgentStart`/`AgentEnd`
    bracket, since none was logged."""
    log = SessionLog("s1")
    log.append(EventKind.TURN_START, {"reason": "initial"})
    log.append(EventKind.TURN_END, {})

    events = project(log)

    assert not any(isinstance(e, AgentStart | AgentEnd) for e in events)
    assert [type(e) for e in events] == [TurnStart, TurnEnd]


def test_agent_start_carries_its_causes() -> None:
    log = SessionLog("s1")
    log.append(EventKind.AGENT_START, {"causes": [{"id": "e1", "origin": "matrix"}]})
    log.append(EventKind.AGENT_END, {"reason": "completed", "causes": []})

    start = next(e for e in project(log) if isinstance(e, AgentStart))

    assert start.causes[0]["origin"] == "matrix"


def test_agent_end_carries_its_reason_and_causes() -> None:
    log = SessionLog("s1")
    log.append(EventKind.AGENT_START, {"causes": [{"id": "e1", "origin": "matrix"}]})
    log.append(
        EventKind.AGENT_END,
        {"reason": "completed", "causes": [{"id": "e1", "origin": "matrix"}]},
    )

    end = next(e for e in project(log) if isinstance(e, AgentEnd))

    assert end.reason == "completed"
    assert end.causes[0]["origin"] == "matrix"


def test_agent_end_messages_are_invocation_local() -> None:
    """Pi's `agent_end.messages`: everything this run itself produced or
    consumed, not the whole log -- confirmed by scoping to only this run's
    own `AGENT_START`/`AGENT_END` bracket, with nothing outside it leaking in."""
    log = SessionLog("s1")
    log.append(EventKind.AGENT_START, {"causes": []})
    log.append(EventKind.TURN_START, {"reason": "initial"})
    log.append(
        EventKind.USER_MESSAGE,
        {"message": encode_message(UserMessage((TextBlock(text="hi"),), 1))},
    )
    log.append(
        EventKind.ASSISTANT_MESSAGE,
        {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
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
    log.append(EventKind.TURN_END, {})
    log.append(EventKind.AGENT_END, {"reason": "completed", "causes": []})

    end = next(e for e in project(log) if isinstance(e, AgentEnd))

    assert [m.__class__.__name__ for m in end.messages] == ["UserMessage", "AssistantMessage"]


def test_a_second_run_does_not_see_the_first_runs_messages() -> None:
    log = SessionLog("s1")
    log.append(EventKind.AGENT_START, {"causes": []})
    log.append(
        EventKind.USER_MESSAGE,
        {"message": encode_message(UserMessage((TextBlock(text="first"),), 1))},
    )
    log.append(EventKind.AGENT_END, {"reason": "completed", "causes": []})
    log.append(EventKind.AGENT_START, {"causes": []})
    log.append(
        EventKind.USER_MESSAGE,
        {"message": encode_message(UserMessage((TextBlock(text="second"),), 1))},
    )
    log.append(EventKind.AGENT_END, {"reason": "completed", "causes": []})

    ends = [e for e in project(log) if isinstance(e, AgentEnd)]

    assert len(ends[0].messages) == 1
    assert ends[0].messages[0].content[0].text == "first"  # type: ignore[union-attr]
    assert len(ends[1].messages) == 1
    assert ends[1].messages[0].content[0].text == "second"  # type: ignore[union-attr]


def test_turn_start_carries_no_causes() -> None:
    """`causes` belongs to `AgentStart` (the run), not `TurnStart` (each
    turn within it) -- confirmed by pinned pi's own bare `turn_start`."""
    log = SessionLog("s1")
    log.append(EventKind.TURN_START, {"reason": "initial"})

    start = next(e for e in project(log) if isinstance(e, TurnStart))

    assert start == TurnStart()


def test_turn_end_carries_the_turns_own_assistant_message() -> None:
    log = SessionLog("s1")
    log.append(EventKind.AGENT_START, {"causes": []})
    log.append(EventKind.TURN_START, {"reason": "initial"})
    log.append(
        EventKind.ASSISTANT_MESSAGE,
        {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
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
    log.append(EventKind.TURN_END, {})
    log.append(EventKind.AGENT_END, {"reason": "completed", "causes": []})

    end = next(e for e in project(log) if isinstance(e, TurnEnd))

    assert end.message is not None
    assert end.message.content[0].text == "hello"  # type: ignore[union-attr]


def test_turn_end_carries_its_own_tool_results() -> None:
    events = project(_log_run(with_tool=True))

    end = next(e for e in events if isinstance(e, TurnEnd))

    assert [m.tool_call_id for m in end.tool_results] == ["t1"]  # type: ignore[union-attr]


def test_a_second_turns_end_does_not_see_the_first_turns_tool_results() -> None:
    log = SessionLog("s1")
    log.append(EventKind.AGENT_START, {"causes": []})
    log.append(EventKind.TURN_START, {"reason": "initial"})
    log.append(EventKind.TOOL_CALL, {"id": "t1", "name": "echo", "arguments": {}})
    log.append(
        EventKind.TOOL_RESULT,
        {
            "message": {
                "role": "tool_result",
                "content": [{"type": "text", "text": "ok"}],
                "timestamp": 0,
                "tool_call_id": "t1",
                "tool_name": "echo",
                "is_error": False,
            }
        },
    )
    log.append(EventKind.TURN_END, {})
    log.append(EventKind.TURN_START, {"reason": "tool_results"})
    log.append(EventKind.TURN_END, {})
    log.append(EventKind.AGENT_END, {"reason": "completed", "causes": []})

    ends = [e for e in project(log) if isinstance(e, TurnEnd)]

    assert len(ends[0].tool_results) == 1
    assert ends[1].tool_results == ()
    assert ends[1].message is None


def _partial(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text=text),),
        stop_reason=StopReason.STOP,
        usage=Usage(),
        model="m",
        provider="p",
        timestamp=0,
    )


def test_a_chunk_projects_to_a_message_update() -> None:
    """Pi's tenth event, `{assistantMessageEvent, message}` (`L08-R002`, PASS 6): `event` is the
    raw, type-specific stream chunk -- here a `TextDelta` carrying its own `delta` -- and `message`
    is the full partial assistant message as accumulated so far (`L08-R003`), not a raw string."""
    log = SessionLog("s1")
    partial = _partial("hi")
    log.append(
        EventKind.ASSISTANT_CHUNK,
        {
            "kind": "text_delta",
            "content_index": 0,
            "partial": encode_message(partial),
            "delta": "hi",
        },
    )

    update = next(e for e in project(log) if isinstance(e, MessageUpdate))

    assert update.message == partial
    assert isinstance(update.event, TextDelta)
    assert (update.event.content_index, update.event.delta) == (0, "hi")


def test_updates_precede_the_message_they_assemble() -> None:
    """Pinned pi's exact order, `message_start -> message_update* ->
    message_end` (`L08-R003`): the stream's own `"start"` event opens the
    reply's `MessageStart` before any `MessageUpdate`."""
    log = SessionLog("s1")
    log.append(EventKind.ASSISTANT_STREAM_START, {"partial": encode_message(_partial(""))})
    log.append(
        EventKind.ASSISTANT_CHUNK,
        {
            "kind": "text_delta",
            "content_index": 0,
            "partial": encode_message(_partial("hi")),
            "delta": "hi",
        },
    )
    log.append(EventKind.ASSISTANT_MESSAGE, {"message": encode_message(_partial("hi"))})

    kinds = [type(e) for e in project(log)]

    assert kinds.index(MessageStart) < kinds.index(MessageUpdate) < kinds.index(MessageEnd)


def _tool_result_entry(
    *, call_id: str, text: str, completion_index: int, tool_name: str = "echo"
) -> dict:
    return {
        "message": {
            "role": "tool_result",
            "content": [{"type": "text", "text": text}],
            "timestamp": 0,
            "tool_call_id": call_id,
            "tool_name": tool_name,
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
        _tool_result_entry(call_id="t1", text="slow", completion_index=1, tool_name="slow"),
    )
    log.append(
        EventKind.TOOL_RESULT,
        _tool_result_entry(call_id="t2", text="quick", completion_index=0, tool_name="quick"),
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
    log.append(EventKind.AGENT_START, {"causes": []})
    log.append(EventKind.TOOL_CALL, {"id": "t1", "name": "echo", "arguments": {}})
    log.append(
        EventKind.TOOL_RESULT,
        _tool_result_entry(call_id="t1", text="ok", completion_index=0),
    )
    log.append(EventKind.AGENT_END, {"reason": "completed", "causes": []})

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
                "tool_name": "echo",
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
    log.append("agent/start", {"causes": []})
    log.append("turn/start", {"reason": "initial"})
    log.append("tool/call", {"id": "t1", "name": "echo", "arguments": {}})
    log.append("turn/end", {})
    log.append("agent/end", {"reason": "completed", "causes": []})

    assert [type(e) for e in project(log)] == [
        AgentStart,
        TurnStart,
        ToolExecutionStart,
        TurnEnd,
        AgentEnd,
    ]
