"""The Minion-only host boundary-stop latch (`L08-R009`: NOT pinned Pi's own
`Agent.abort()`, deferred to Layer 09 -- see `AG-022`), and the guarantee
that one agent cannot stall another."""

import asyncio
from typing import Any

from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.registry import AgentRegistry
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import LlmService, ModelId, TextBlock, ToolCallBlock, UserMessage
from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.runtime import Context
from minion_agent.session import EventKind, SessionService
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.registry import ToolRegistry

from .test_single_turn import _loop, _register


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _tool_call() -> ScriptedResponse:
    return ScriptedResponse(
        (ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE
    )


def _boundary_stopping_tool(loop: AgentLoop, output: str = "done") -> Any:
    def run(tool_call_id: str, args: dict[str, Any]) -> str:
        loop.request_boundary_stop()
        return output

    return run


async def test_boundary_stop_ends_the_turn_at_the_next_boundary() -> None:
    loop = _loop(*[_tool_call() for _ in range(5)])
    _register(loop, "echo", _boundary_stopping_tool(loop))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    steps = [e for e in loop.instance.log.events if e.kind == EventKind.TURN_START]
    assert len(steps) == 1

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "boundary_stop"


async def test_a_boundary_stopped_turn_still_records_its_tool_result() -> None:
    """The boundary stop takes effect at the next request, not the work
    already in flight."""
    loop = _loop(_tool_call())
    _register(loop, "echo", _boundary_stopping_tool(loop, "finished"))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    results = [e for e in loop.instance.log.events if e.kind == EventKind.TOOL_RESULT]
    assert len(results) == 1


async def test_boundary_stop_clears_so_the_next_turn_runs() -> None:
    loop = _loop(
        _tool_call(),
        ScriptedResponse((TextBlock(text="second turn"),), StopReason.STOP),
    )
    _register(loop, "echo", _boundary_stopping_tool(loop))
    loop.instance.inbox.followup(_say("first"))
    await loop.run_until_idle()

    loop.instance.inbox.followup(_say("second"))
    await loop.run_until_idle()

    ends = [e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END]
    assert [end.data["reason"] for end in ends] == ["boundary_stop", "completed"]


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
    # The driver dispatches `tools/*` events through the real pipeline; a bare
    # Context never mounts `tools_plugin`, so declaration has to happen here.
    declare_tools_events(ctx.events)
    sessions = SessionService()
    registry = AgentRegistry(ctx=ctx, sessions=sessions)
    definition = AgentDefinition(name="ada", model=ModelId("mock", "mock-1"))

    released = asyncio.Event()
    blocked_tools = ToolRegistry()

    async def wait_for_release(tool_call_id: str, args: dict[str, Any]) -> str:
        await released.wait()
        return "released"

    blocked_tools.register(
        ToolDefinition(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {}},
            execute=wait_for_release,
            label="Echo",
        )
    )

    blocked = AgentLoop(
        instance=registry.create("blocked", definition).instance,
        llm=_service(_tool_call(), ScriptedResponse((), StopReason.STOP)),
        tools=blocked_tools,
        artifacts=sessions.artifacts,
    )
    free = AgentLoop(
        instance=registry.create("free", definition).instance,
        llm=_service(ScriptedResponse((TextBlock(text="free"),), StopReason.STOP)),
        tools=ToolRegistry(),
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
