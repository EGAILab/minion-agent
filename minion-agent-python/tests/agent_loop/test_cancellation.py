"""Cancellation, and the guarantee that one agent cannot stall another."""

import asyncio
from typing import Any

from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.registry import AgentRegistry
from minion_agent.agent.tools import ToolService
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import LlmService, ModelId, TextBlock, ToolCallBlock, UserMessage
from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.runtime import Context
from minion_agent.session import EventKind, SessionService

from .test_single_turn import _loop


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _tool_call() -> ScriptedResponse:
    return ScriptedResponse(
        (ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE
    )


def _cancelling_tool(loop: AgentLoop, output: str = "done") -> Any:
    def run(args: dict[str, Any]) -> str:
        loop.cancel()
        return output

    return run


async def test_cancelling_ends_the_turn_at_the_next_boundary() -> None:
    loop = _loop(*[_tool_call() for _ in range(5)])
    loop.tools.register("echo", _cancelling_tool(loop))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    steps = [e for e in loop.instance.log.events if e.kind == EventKind.STEP_START]
    assert len(steps) == 1

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.TURN_END)
    assert end.data["reason"] == "cancelled"


async def test_a_cancelled_turn_still_records_its_tool_result() -> None:
    """Cancellation stops the next request, not the work already in flight."""
    loop = _loop(_tool_call())
    loop.tools.register("echo", _cancelling_tool(loop, "finished"))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    results = [e for e in loop.instance.log.events if e.kind == EventKind.TOOL_RESULT]
    assert len(results) == 1


async def test_cancelling_clears_so_the_next_turn_runs() -> None:
    loop = _loop(
        _tool_call(),
        ScriptedResponse((TextBlock(text="second turn"),), StopReason.STOP),
    )
    loop.tools.register("echo", _cancelling_tool(loop))
    loop.instance.inbox.followup(_say("first"))
    await loop.run_until_idle()

    loop.instance.inbox.followup(_say("second"))
    await loop.run_until_idle()

    ends = [e for e in loop.instance.log.events if e.kind == EventKind.TURN_END]
    assert [end.data["reason"] for end in ends] == ["cancelled", "completed"]


def _service(*responses: ScriptedResponse) -> LlmService:
    """One adapter per agent: a shared script would couple their progress,
    which is the very thing this test is trying to observe."""
    llm = LlmService()
    llm.register(MockAdapter(list(responses)))
    return llm


async def test_a_blocked_agent_does_not_stall_another() -> None:
    """The normative progress guarantee (section 6). An await inside one
    instance must never occupy a runtime-global critical section."""
    ctx = Context()
    sessions = SessionService()
    registry = AgentRegistry(ctx=ctx, sessions=sessions)
    definition = AgentDefinition(name="ada", model=ModelId("mock", "mock-1"))

    released = asyncio.Event()
    blocked_tools = ToolService()

    async def wait_for_release(args: dict[str, Any]) -> str:
        await released.wait()
        return "released"

    blocked_tools.register("echo", wait_for_release)

    blocked = AgentLoop(
        instance=registry.create("blocked", definition).instance,
        llm=_service(_tool_call(), ScriptedResponse((), StopReason.STOP)),
        tools=blocked_tools,
        artifacts=sessions.artifacts,
    )
    free = AgentLoop(
        instance=registry.create("free", definition).instance,
        llm=_service(ScriptedResponse((TextBlock(text="free"),), StopReason.STOP)),
        tools=ToolService(),
        artifacts=sessions.artifacts,
    )

    blocked.instance.inbox.followup(_say("blocked"))
    free.instance.inbox.followup(_say("free"))

    blocked_task = asyncio.create_task(blocked.run_until_idle())
    await free.run_until_idle()  # completes while the other is stuck

    assert len(free.instance.log) > 0
    assert not blocked_task.done()

    released.set()
    await blocked_task
