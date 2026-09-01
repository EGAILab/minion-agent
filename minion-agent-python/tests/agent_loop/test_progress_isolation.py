"""The guarantee that one agent cannot stall another.

Formerly also covered the local `cancel()`/`request_boundary_stop()`
boundary-stop latch (`test_boundary_stop.py`), removed entirely at Layer 08
PASS 5 (`L08-R009`/`L08-R010`): a public method that could alter a
Pi-equivalent run's own observable outcome had no owner governance approval
for that divergence, and no demonstrated product need justified keeping it
-- the same default this project already applied to `max_steps`
(`L08-R005`). See `assurance/layers/08-agent-loop-python.md`, PASS 5.
"""

import asyncio
from typing import Any

from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.registry import AgentRegistry
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import LlmService, ModelId, TextBlock, ToolCallBlock, UserMessage
from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.runtime import Context
from minion_agent.session import SessionService
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.registry import ToolRegistry


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _tool_call() -> ScriptedResponse:
    return ScriptedResponse(
        (ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE
    )


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
