"""Pi's event stream, rebuilt from the log."""

from minion_agent.agent.projection import (
    AgentEnd,
    AgentStart,
    MessageEnd,
    MessageStart,
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
