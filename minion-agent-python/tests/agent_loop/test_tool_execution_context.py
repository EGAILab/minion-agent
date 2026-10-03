"""`L0506-D004` (`TOOL-042`, Owner `WP133-F3` = A): a tool that opts in receives, per call, an
immutable snapshot of the EXECUTING agent's authoritative state -- session id, model identity,
thinking level verbatim (`off` included), `session_file` absent today. Driven through real
`AgentLoop` runs; contract `spec/tools.md`, "Per-call tool execution context"."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from minion_agent.agent.identity import AgentDefinition, ThinkingLevel
from minion_agent.agent.registry import AgentRegistry
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import LlmService, ModelId, TextBlock, ToolCallBlock, UserMessage
from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.runtime import Context
from minion_agent.session import SessionService
from minion_agent.tools import ToolExecutionContext
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.registry import ToolRegistry


def _call(call_id: str = "t1") -> ScriptedResponse:
    return ScriptedResponse(
        (ToolCallBlock(id=call_id, name="probe", arguments={}),), StopReason.TOOL_USE
    )


def _stop() -> ScriptedResponse:
    return ScriptedResponse((TextBlock(text="done"),), StopReason.STOP)


class _Adapter(MockAdapter):
    models = frozenset({"m-1", "m-2", "ma", "mb"})


def _llm(*responses: ScriptedResponse) -> LlmService:
    llm = LlmService()
    llm.register(_Adapter(list(responses)))
    return llm


def _probe(seen: list[ToolExecutionContext | None], on_call: Any = None) -> ToolRegistry:
    registry = ToolRegistry()

    async def execute(
        tool_call_id: str, args: dict[str, Any], *, context: ToolExecutionContext | None
    ) -> str:
        seen.append(context)
        if on_call is not None:
            on_call()
        return "ok"

    registry.register(
        ToolDefinition(
            name="probe",
            label="probe",
            description="probe",
            parameters={"type": "object", "properties": {}},
            execute=execute,
            wants_context=True,
        )
    )
    return registry


def _world() -> tuple[AgentRegistry, SessionService]:
    ctx = Context()
    declare_tools_events(ctx.events)
    sessions = SessionService()
    return AgentRegistry(ctx=ctx, sessions=sessions), sessions


async def _run(loop: AgentLoop) -> None:
    loop.instance.inbox.followup(UserMessage(content=(TextBlock(text="go"),), timestamp=1))
    await loop.run_until_idle()


async def test_an_agent_run_call_receives_its_own_authoritative_state() -> None:
    agents, sessions = _world()
    instance = agents.create(
        "ada", AgentDefinition(name="ada", model=ModelId("mock", "m-1"))
    ).instance
    instance.thinking_level = ThinkingLevel.HIGH
    seen: list[ToolExecutionContext | None] = []
    loop = AgentLoop(
        instance=instance,
        llm=_llm(_call(), _stop()),
        tools=_probe(seen),
        artifacts=sessions.artifacts,
    )
    await _run(loop)
    assert seen == [
        ToolExecutionContext(
            session_id=instance.log.session_id,
            session_file=None,
            provider="mock",
            model="m-1",
            reasoning_level="high",
        )
    ]


async def test_off_is_a_present_reasoning_level_not_an_absent_one() -> None:
    agents, sessions = _world()
    instance = agents.create(
        "ada", AgentDefinition(name="ada", model=ModelId("mock", "m-1"))
    ).instance
    seen: list[ToolExecutionContext | None] = []
    loop = AgentLoop(
        instance=instance,
        llm=_llm(_call(), _stop()),
        tools=_probe(seen),
        artifacts=sessions.artifacts,
    )
    await _run(loop)
    assert seen[0] is not None and seen[0].reasoning_level == "off"


async def test_a_later_call_sees_the_changed_state_and_an_earlier_snapshot_is_unchanged() -> None:
    agents, sessions = _world()
    instance = agents.create(
        "ada", AgentDefinition(name="ada", model=ModelId("mock", "m-1"))
    ).instance
    seen: list[ToolExecutionContext | None] = []

    def change() -> None:
        instance.model = ModelId("mock", "m-2")
        instance.thinking_level = ThinkingLevel.LOW

    loop = AgentLoop(
        instance=instance,
        llm=_llm(_call("t1"), _call("t2"), _stop()),
        tools=_probe(seen, on_call=change),
        artifacts=sessions.artifacts,
    )
    await _run(loop)
    first, second = seen
    assert first is not None and second is not None
    assert (first.provider, first.model, first.reasoning_level) == ("mock", "m-1", "off")
    assert (second.provider, second.model, second.reasoning_level) == ("mock", "m-2", "low")
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.model = "changed"  # type: ignore[misc]


async def test_two_agents_sharing_one_registration_each_see_their_own_context() -> None:
    """Owner F3 section 18: one registration, many agents -- a construction-time capture fails."""
    agents, sessions = _world()
    seen: list[ToolExecutionContext | None] = []
    shared = _probe(seen)
    a = agents.create("a", AgentDefinition(name="a", model=ModelId("mock", "ma"))).instance
    b = agents.create("b", AgentDefinition(name="b", model=ModelId("mock", "mb"))).instance
    b.thinking_level = ThinkingLevel.MAX
    for instance in (a, b):
        loop = AgentLoop(
            instance=instance,
            llm=_llm(_call(), _stop()),
            tools=shared,
            artifacts=sessions.artifacts,
        )
        await _run(loop)
    assert [(c.session_id, c.model, c.reasoning_level) for c in seen if c is not None] == [
        (a.log.session_id, "ma", "off"),
        (b.log.session_id, "mb", "max"),
    ]
    assert a.log.session_id != b.log.session_id


async def test_control_a_registration_time_capture_fails_the_two_agent_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: a context fixed once (the first agent's) instead of per executing agent."""
    captured: list[ToolExecutionContext] = []
    original = AgentLoop._tool_execution_context

    def first_ever(self: AgentLoop) -> ToolExecutionContext:
        if not captured:
            captured.append(original(self))
        return captured[0]

    monkeypatch.setattr(AgentLoop, "_tool_execution_context", first_ever)
    with pytest.raises(AssertionError):
        await test_two_agents_sharing_one_registration_each_see_their_own_context()


async def test_two_calls_in_one_agent_batch_each_see_the_then_current_state() -> None:
    """`L0506-D004-I002`, agent-driven: ONE model reply with two calls to a sequential tool; the
    first call changes the agent's model, and the second call's context shows the change."""
    from minion_agent.tools.definition import ExecutionMode

    agents, sessions = _world()
    instance = agents.create(
        "ada", AgentDefinition(name="ada", model=ModelId("mock", "m-1"))
    ).instance
    seen: list[ToolExecutionContext | None] = []
    registry = ToolRegistry()

    async def execute(
        tool_call_id: str, args: dict[str, Any], *, context: ToolExecutionContext | None
    ) -> str:
        seen.append(context)
        instance.model = ModelId("mock", "m-2")
        return "ok"

    registry.register(
        ToolDefinition(
            name="probe",
            label="probe",
            description="probe",
            parameters={"type": "object", "properties": {}},
            execute=execute,
            wants_context=True,
            mode=ExecutionMode.SEQUENTIAL,
        )
    )
    both = ScriptedResponse(
        (
            ToolCallBlock(id="t1", name="probe", arguments={}),
            ToolCallBlock(id="t2", name="probe", arguments={}),
        ),
        StopReason.TOOL_USE,
    )
    loop = AgentLoop(
        instance=instance, llm=_llm(both, _stop()), tools=registry, artifacts=sessions.artifacts
    )
    await _run(loop)
    assert [c.model for c in seen if c is not None] == ["m-1", "m-2"]
