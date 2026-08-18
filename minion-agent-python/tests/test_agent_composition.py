"""The whole stack composes, in any mounting order."""

from typing import Any

from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.plugin import agents_plugin, tools_plugin
from minion_agent.agent_loop import agent_loop_plugin
from minion_agent.llm import ModelId, TextBlock, UserMessage, text_of
from minion_agent.llm.plugin import llm_plugin, mock_adapter_plugin
from minion_agent.runtime import Context, FiberState
from minion_agent.session import derive_messages
from minion_agent.session.service import session_plugin
from minion_agent.telemetry.plugin import telemetry_plugin


async def _stack(script: list[dict[str, Any]] | None = None) -> Context:
    ctx = Context()
    await ctx.plugin(agent_loop_plugin)  # mounted first, on purpose
    await ctx.plugin(session_plugin)
    await ctx.plugin(llm_plugin)
    await ctx.plugin(tools_plugin)
    await ctx.plugin(agents_plugin)
    await ctx.plugin(telemetry_plugin)
    await ctx.plugin(mock_adapter_plugin, {"script": script or []})
    return ctx


async def test_the_loop_plugin_waits_for_its_dependencies() -> None:
    ctx = Context()

    fiber = await ctx.plugin(agent_loop_plugin)
    assert fiber.state is FiberState.PENDING

    await ctx.plugin(session_plugin)
    await ctx.plugin(llm_plugin)
    await ctx.plugin(tools_plugin)
    await ctx.plugin(agents_plugin)

    assert fiber.state is FiberState.ACTIVE


async def test_a_full_turn_runs_through_the_composed_stack() -> None:
    ctx = await _stack([{"text": "hello there", "stop_reason": "stop"}])
    handle = ctx.agents.create(
        "room-a", AgentDefinition(name="ada", model=ModelId("mock", "mock-1"))
    )
    loop = ctx.agent_loop.for_instance(handle.instance)

    handle.instance.inbox.followup(UserMessage(content=(TextBlock(text="hi"),), timestamp=1))
    await loop.run_until_idle()

    assert [text_of(m) for m in derive_messages(handle.instance.log)] == [
        "hi",
        "hello there",
    ]


async def test_two_instances_of_one_definition_keep_separate_logs() -> None:
    ctx = await _stack(
        [
            {"text": "to a", "stop_reason": "stop"},
            {"text": "to b", "stop_reason": "stop"},
        ]
    )
    definition = AgentDefinition(name="ada", model=ModelId("mock", "mock-1"))
    first = ctx.agents.create("room-a", definition)
    second = ctx.agents.create("room-b", definition)

    for handle, text in ((first, "from a"), (second, "from b")):
        handle.instance.inbox.followup(UserMessage(content=(TextBlock(text=text),), timestamp=1))
        await ctx.agent_loop.for_instance(handle.instance).run_until_idle()

    assert text_of(derive_messages(first.instance.log)[0]) == "from a"
    assert text_of(derive_messages(second.instance.log)[0]) == "from b"


async def test_the_loop_picks_up_telemetry_when_it_is_mounted() -> None:
    """Observational, so it is resolved rather than injected -- but resolved."""
    ctx = await _stack([{"text": "ok", "stop_reason": "stop"}])
    handle = ctx.agents.create(
        "room-a", AgentDefinition(name="ada", model=ModelId("mock", "mock-1"))
    )

    assert ctx.agent_loop.for_instance(handle.instance).telemetry is not None


async def test_unmounting_the_llm_unloads_the_loop() -> None:
    """Reactive dependency reaches all the way up the stack."""
    ctx = Context()
    await ctx.plugin(session_plugin)
    llm_fiber = await ctx.plugin(llm_plugin)
    await ctx.plugin(tools_plugin)
    await ctx.plugin(agents_plugin)
    loop_fiber = await ctx.plugin(agent_loop_plugin)
    assert loop_fiber.state is FiberState.ACTIVE

    await ctx.plugins.unmount(llm_fiber)

    assert loop_fiber.state is FiberState.PENDING
