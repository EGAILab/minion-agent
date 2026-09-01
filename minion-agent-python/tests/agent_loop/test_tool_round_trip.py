"""The Phase 3 milestone: a tool call closes and the model is asked again."""

from minion_agent.llm import TextBlock, ToolCallBlock, UserMessage, text_of
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind, derive_messages

from .test_single_turn import _loop, _register


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _tool_call(name: str = "echo", **arguments: object) -> ScriptedResponse:
    return ScriptedResponse(
        (ToolCallBlock(id="t1", name=name, arguments=dict(arguments)),),
        StopReason.TOOL_USE,
    )


async def test_a_tool_call_is_executed_and_answered() -> None:
    loop = _loop(
        _tool_call(value="pong"),
        ScriptedResponse((TextBlock(text="all done"),), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: str(args["value"]))
    loop.instance.inbox.followup(_say("ping"))

    await loop.run_until_idle()

    assert [text_of(m) for m in derive_messages(loop.instance.log)] == [
        "ping",
        "",
        "pong",
        "all done",
    ]


async def test_the_model_is_asked_again_after_the_result() -> None:
    loop = _loop(
        _tool_call(value="pong"),
        ScriptedResponse((TextBlock(text="done"),), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: str(args["value"]))
    loop.instance.inbox.followup(_say("ping"))

    await loop.run_until_idle()

    steps = [e for e in loop.instance.log.events if e.kind == EventKind.TURN_START]
    assert [step.data["reason"] for step in steps] == ["initial", "tool_results"]


async def test_the_call_and_its_result_are_both_logged() -> None:
    loop = _loop(_tool_call(value="pong"), ScriptedResponse((), StopReason.STOP))
    _register(loop, "echo", lambda tool_call_id, args: str(args["value"]))
    loop.instance.inbox.followup(_say("ping"))

    await loop.run_until_idle()

    kinds = [e.kind for e in loop.instance.log.events]
    assert EventKind.TOOL_CALL in kinds
    assert EventKind.TOOL_RESULT in kinds
    assert kinds.index(EventKind.TOOL_CALL) < kinds.index(EventKind.TOOL_RESULT)


async def test_an_unknown_tool_still_closes_the_loop() -> None:
    """Every call gets a result, so the transcript stays coherent."""
    loop = _loop(_tool_call("missing"), ScriptedResponse((TextBlock(text="ok"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("ping"))

    await loop.run_until_idle()

    derived = derive_messages(loop.instance.log)
    assert text_of(derived[2]) == "Tool missing not found"
    assert text_of(derived[-1]) == "ok"


async def test_several_calls_in_one_message_each_get_a_result() -> None:
    loop = _loop(
        ScriptedResponse(
            (
                ToolCallBlock(id="t1", name="echo", arguments={"value": "one"}),
                ToolCallBlock(id="t2", name="echo", arguments={"value": "two"}),
            ),
            StopReason.TOOL_USE,
        ),
        ScriptedResponse((), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: str(args["value"]))
    loop.instance.inbox.followup(_say("ping"))

    await loop.run_until_idle()

    results = [e for e in loop.instance.log.events if e.kind == EventKind.TOOL_RESULT]
    assert len(results) == 2


async def test_a_long_tool_loop_is_not_bounded_by_any_turn_count() -> None:
    """`L08-R005`: pinned Pi has no `max_steps`-equivalent stop rule, and the
    Minion-only turn-counter cap that used to bound this exact shape (a model
    that only ever calls tools) is now fully removed from the Pi-equivalent
    `prompt()`/`continue()` seam -- not repositioned, not parity-neutral.
    Regression: the old default cap (formerly `AgentDefinition.max_steps`,
    default 16) would have stopped this run at its 16th turn; it does not."""
    loop = _loop(
        *[_tool_call(value="again") for _ in range(20)], ScriptedResponse((), StopReason.STOP)
    )
    _register(loop, "echo", lambda tool_call_id, args: "again")
    loop.instance.inbox.followup(_say("ping"))

    await loop.run_until_idle()

    steps = [e for e in loop.instance.log.events if e.kind == EventKind.TURN_START]
    assert len(steps) == 21  # 20 tool-calling turns + the final stopping turn

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "completed"
