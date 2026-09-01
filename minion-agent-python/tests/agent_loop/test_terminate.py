"""The terminate fold: unanimous, and not overridable."""

from typing import Any

from minion_agent.agent.decisions import TurnStopping
from minion_agent.agent.events import AGENT_TURN_STOPPING
from minion_agent.llm import TextBlock, ToolCallBlock, UserMessage
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.result import ToolResult

from .test_single_turn import _loop


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _calls(*names: str) -> ScriptedResponse:
    return ScriptedResponse(
        tuple(
            ToolCallBlock(id=f"t{index}", name=name, arguments={})
            for index, name in enumerate(names, start=1)
        ),
        StopReason.TOOL_USE,
    )


def _tool(name: str, *, terminate: bool) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=lambda tool_call_id, args: ToolResult(
            tool_call_id="", content=(), tool_name=name, terminate=terminate
        ),
        label=name,
    )


def _steps(loop: Any) -> int:
    return len([e for e in loop.instance.log.events if e.kind == EventKind.TURN_START])


async def test_a_unanimous_batch_ends_the_turn() -> None:
    loop = _loop(_calls("stop", "stop"), ScriptedResponse((), StopReason.STOP))
    loop.tools.register(_tool("stop", terminate=True))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert _steps(loop) == 1
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "terminated"


async def test_one_dissenting_result_continues_the_turn() -> None:
    """Pi's rule is unanimity. Any-wins would let a single tool end a turn the
    other tools in its own batch were still working on."""
    loop = _loop(_calls("stop", "go"), ScriptedResponse((), StopReason.STOP))
    loop.tools.register(_tool("stop", terminate=True))
    loop.tools.register(_tool("go", terminate=False))
    loop.instance.inbox.followup(_say("start"))

    await loop.run_until_idle()

    assert _steps(loop) == 2


async def test_terminate_is_not_overridable_even_though_the_decision_still_fires() -> None:
    """Layer 08, PASS 2: hard termination is not overridable, but it is NOT
    reached by skipping the stop decision -- pinned pi's own `shouldStopAfterTurn`
    still runs after every turn regardless of `hasMoreToolCalls`/`terminate`
    (confirmed directly against `runLoop`); `terminate` only ever affects
    whether the tool-driven inner loop has more work. A listener answering
    CONTINUE here is asked, and still cannot manufacture a second request:
    with no more tool calls and nothing steered, the inner loop exhausts on
    its own regardless of the listener's answer. An earlier revision of this
    test (and the `agent/turn-stopping` docstring it was pinning) claimed the
    decision was skipped entirely -- that was never actually true of pinned
    Pi."""
    loop = _loop(_calls("stop"), ScriptedResponse((), StopReason.STOP))
    loop.tools.register(_tool("stop", terminate=True))
    asked: list[str] = []

    def record(*args: Any) -> TurnStopping:
        asked.append("asked")
        return TurnStopping.CONTINUE

    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, record)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert asked == ["asked"]
    assert _steps(loop) == 1
    # Strengthened beyond the brief: a turn can also end at one step for an
    # unrelated reason (e.g. max_steps == 1), which would likewise leave
    # `asked` empty without proving termination fired. Pinning the reason
    # rules that confound out -- default max_steps is 16, far above 1.
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "terminated"


async def test_a_terminating_batch_still_logs_its_results() -> None:
    """Ending the turn does not discard work already done.

    Strengthened beyond the brief: the second scripted response is itself a
    tool call rather than a bare stop, so a loop that failed to terminate and
    ran a second step would log a *second* TOOL_RESULT, making the count
    assertion alone able to catch that regression. The step count and end
    reason are pinned too, so "continued normally but still produced exactly
    one result" cannot slip through.
    """
    loop = _loop(_calls("stop"), _calls("go"))
    loop.tools.register(_tool("stop", terminate=True))
    loop.tools.register(_tool("go", terminate=False))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    results = [e for e in loop.instance.log.events if e.kind == EventKind.TOOL_RESULT]
    assert len(results) == 1
    assert _steps(loop) == 1
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "terminated"


async def test_a_step_with_no_tools_never_terminates() -> None:
    """An empty batch has no result asking to stop; vacuous agreement must not
    end a turn."""
    loop = _loop(ScriptedResponse((TextBlock(text="done"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "completed"
