"""The tools plugin provides the registry and declares its events."""

from minion_agent.runtime import Context, DispatchMode, FiberState
from minion_agent.tools.events import TOOLS_POST_EXECUTE, TOOLS_PRE_EXECUTE
from minion_agent.tools.plugin import tools_plugin
from minion_agent.tools.registry import ToolRegistry


async def test_mounting_provides_the_registry() -> None:
    ctx = Context()

    fiber = await ctx.plugin(tools_plugin, None)

    assert fiber.state is FiberState.ACTIVE
    assert isinstance(ctx.tools, ToolRegistry)


async def test_mounting_declares_the_pipeline_events() -> None:
    """A listener cannot register for an undeclared event, so declaration has
    to happen when the service appears rather than at first dispatch."""
    ctx = Context()

    await ctx.plugin(tools_plugin, None)

    assert ctx.events.mode_of(TOOLS_PRE_EXECUTE) is DispatchMode.WATERFALL
    assert ctx.events.mode_of(TOOLS_POST_EXECUTE) is DispatchMode.WATERFALL


async def test_unmounting_withdraws_the_registry() -> None:
    ctx = Context()
    fiber = await ctx.plugin(tools_plugin, None)

    await ctx.plugins.unmount(fiber)

    assert not ctx.registry.has("tools")
