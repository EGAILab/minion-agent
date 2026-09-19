"""The execution plugins mount `fs`/`shell`/`subprocess` on the runtime (`L12-PY-R003`)."""

from minion_agent.execution.filesystem import LocalFileSystem
from minion_agent.execution.plugin import fs_plugin, shell_plugin, subprocess_plugin
from minion_agent.execution.shell import LocalShell
from minion_agent.execution.subprocess import LocalSubprocess
from minion_agent.runtime import Context, FiberState


async def test_fs_plugin_provides_a_local_filesystem() -> None:
    ctx = Context()

    fiber = await ctx.plugin(fs_plugin, None)

    assert fiber.state is FiberState.ACTIVE
    assert isinstance(ctx.fs, LocalFileSystem)


async def test_fs_plugin_unmounting_withdraws_the_service() -> None:
    ctx = Context()
    fiber = await ctx.plugin(fs_plugin, None)

    await ctx.plugins.unmount(fiber)

    assert not ctx.registry.has("fs")


async def test_shell_plugin_provides_a_local_shell() -> None:
    ctx = Context()

    fiber = await ctx.plugin(shell_plugin, None)

    assert fiber.state is FiberState.ACTIVE
    assert isinstance(ctx.shell, LocalShell)


async def test_shell_plugin_unmounting_withdraws_the_service() -> None:
    ctx = Context()
    fiber = await ctx.plugin(shell_plugin, None)

    await ctx.plugins.unmount(fiber)

    assert not ctx.registry.has("shell")


async def test_subprocess_plugin_provides_a_local_subprocess() -> None:
    ctx = Context()

    fiber = await ctx.plugin(subprocess_plugin, None)

    assert fiber.state is FiberState.ACTIVE
    assert isinstance(ctx.subprocess, LocalSubprocess)


async def test_subprocess_plugin_unmounting_withdraws_the_service() -> None:
    ctx = Context()
    fiber = await ctx.plugin(subprocess_plugin, None)

    await ctx.plugins.unmount(fiber)

    assert not ctx.registry.has("subprocess")


async def test_all_three_local_providers_share_one_execution_world_identity_by_construction() -> (
    None
):
    """`L12-PY-R003`: the three `Local*` providers, mounted independently through separate
    plugins with no shared state passed between them, still declare the SAME execution-world
    identity -- spec section 8's "all three local providers share one execution-world identity
    by construction, the simplest possible case" rule."""
    ctx = Context()
    await ctx.plugin(fs_plugin, None)
    await ctx.plugin(shell_plugin, None)
    await ctx.plugin(subprocess_plugin, None)

    assert ctx.fs.execution_world == ctx.shell.execution_world == ctx.subprocess.execution_world
