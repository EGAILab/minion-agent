"""One prompt in, one model request, one logged turn."""

from typing import Any

from minion_agent.agent.identity import AgentDefinition, AgentStatus
from minion_agent.agent.registry import AgentRegistry
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import LlmService, ModelId, TextBlock, UserMessage, text_of
from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.runtime import Context
from minion_agent.session import ArtifactStore, EventKind, SessionService, derive_messages
from minion_agent.tools.definition import ExecutionMode, ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.registry import ToolRegistry


def _register(
    loop: AgentLoop, name: str, fn: Any, mode: ExecutionMode = ExecutionMode.PARALLEL
) -> None:
    """Register a bare callable as a tool, for tests that only care that one
    ran. Production registration goes through `register_tool`, which makes it
    a reversible effect."""
    loop.tools.register(
        ToolDefinition(name=name, description=name, parameters=None, execute=fn, mode=mode)
    )


def _loop_with_adapter(*responses: ScriptedResponse) -> tuple[AgentLoop, MockAdapter]:
    """A loop plus the adapter behind it, for tests that inspect requests."""
    ctx = Context()
    # The driver executes tool calls through the real pipeline, which
    # dispatches `tools/*` events; a bare Context never mounts `tools_plugin`,
    # so the declaration has to happen here for that dispatch to be legal.
    declare_tools_events(ctx.events)
    sessions = SessionService()
    llm = LlmService()
    adapter = MockAdapter(list(responses))
    llm.register(adapter)
    registry = AgentRegistry(ctx=ctx, sessions=sessions)
    handle = registry.create(
        "room-a",
        AgentDefinition(name="ada", model=ModelId("mock", "mock-1"), system="be helpful"),
    )
    loop = AgentLoop(
        instance=handle.instance,
        llm=llm,
        tools=ToolRegistry(),
        artifacts=sessions.artifacts,
    )
    return loop, adapter


def _loop(*responses: ScriptedResponse) -> AgentLoop:
    return _loop_with_adapter(*responses)[0]


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


async def test_a_prompt_produces_one_assistant_message() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="hi there"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    derived = derive_messages(loop.instance.log)
    assert [text_of(message) for message in derived] == ["hello", "hi there"]


async def test_the_turn_is_logged_in_order() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    kinds = [event.kind for event in loop.instance.log.events]
    assert kinds == [
        EventKind.TURN_START,
        EventKind.USER_MESSAGE,
        EventKind.STEP_START,
        EventKind.REQUEST_HEADER,
        EventKind.ASSISTANT_MESSAGE,
        EventKind.STEP_END,
        EventKind.TURN_END,
    ]


async def test_the_instance_returns_to_idle() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert loop.instance.status is AgentStatus.IDLE


async def test_status_transitions_are_announced_once_each_way() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    seen: list[AgentStatus] = []
    loop.instance.on_status_change = seen.append
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert seen == [AgentStatus.RUNNING, AgentStatus.IDLE]


async def test_an_empty_inbox_runs_nothing() -> None:
    loop = _loop()

    await loop.run_until_idle()

    assert len(loop.instance.log) == 0
    assert loop.instance.status is AgentStatus.IDLE


async def test_the_request_carries_the_definition_system_prompt() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    header = next(
        event for event in loop.instance.log.events if event.kind is EventKind.REQUEST_HEADER
    )
    assert header.data["components"]["system_base"].startswith("sha256:")


async def test_the_logged_header_reconstructs_what_was_dispatched() -> None:
    """The invariant section 5 exists for: the model saw what the log says."""
    from minion_agent.session import assemble_system, reconstruct_header

    store = ArtifactStore()
    ctx = Context()
    sessions = SessionService()
    llm = LlmService()
    adapter = MockAdapter([ScriptedResponse((), StopReason.STOP)])
    llm.register(adapter)
    registry = AgentRegistry(ctx=ctx, sessions=sessions)
    handle = registry.create(
        "room-a",
        AgentDefinition(name="ada", model=ModelId("mock", "mock-1"), system="be helpful"),
    )
    loop = AgentLoop(instance=handle.instance, llm=llm, tools=ToolRegistry(), artifacts=store)
    handle.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    header = next(
        event for event in handle.instance.log.events if event.kind is EventKind.REQUEST_HEADER
    )
    assert assemble_system(reconstruct_header(header, store)) == adapter.requests[0].system
