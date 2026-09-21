"""The execution plugins mount `fs`/`shell`/`subprocess` on the runtime (`L12-PY-R003`)."""

from __future__ import annotations

from collections.abc import Sequence

from minion_agent.execution.errors import SubprocessError, SubprocessErrorCode
from minion_agent.execution.filesystem import LocalFileSystem
from minion_agent.execution.plugin import fs_plugin, shell_plugin, subprocess_plugin
from minion_agent.execution.result import Err, Result
from minion_agent.execution.shell import LocalShell
from minion_agent.execution.subprocess import LocalSubprocess, Process, SpawnOptions
from minion_agent.execution.world import ExecutionWorldIdentity
from minion_agent.runtime import Context, FiberState


class _FakeOwner:
    """Matches `tests/runtime/test_context_access.py`'s own established convention for
    registering a service without a real plugin/fiber -- `ServiceRegistry.provide()`'s own
    visibility check only requires `owner.state is FiberState.ACTIVE`."""

    name = "fake-owner"
    state = FiberState.ACTIVE


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
    """`shell_plugin` depends on `subprocess` being mounted first (`L12-PY-R003`)."""
    ctx = Context()
    await ctx.plugin(subprocess_plugin, None)

    fiber = await ctx.plugin(shell_plugin, None)

    assert fiber.state is FiberState.ACTIVE
    assert isinstance(ctx.shell, LocalShell)


async def test_shell_plugin_unmounting_withdraws_the_service() -> None:
    ctx = Context()
    await ctx.plugin(subprocess_plugin, None)
    fiber = await ctx.plugin(shell_plugin, None)

    await ctx.plugins.unmount(fiber)

    assert not ctx.registry.has("shell")


async def test_shell_plugin_dispatches_through_the_mounted_subprocess_capability() -> None:
    """`L12-PY-R003` (refined, second review): a fake conforming `Subprocess` is mounted, then
    `shell_plugin` -- the resulting `ctx.shell` must actually dispatch through THAT mounted
    instance, not silently construct its own private `LocalSubprocess`. An earlier revision's
    `shell_plugin` declared `inject=["subprocess"]` (gating activation ordering only, per this
    codebase's own established convention) but never resolved `ctx.subprocess` in its own body,
    so `LocalShell()` fell back to constructing a second, unmounted `LocalSubprocess` -- this
    test proves the injected instance's own `spawn()` is the one actually called."""

    class FakeSubprocess:
        __service_name__ = "subprocess"

        def __init__(self) -> None:
            self.cwd = "/fake"
            self.execution_world = ExecutionWorldIdentity("fake-world")
            self.spawn_calls: list[Sequence[str]] = []

        async def spawn(
            self, argv: Sequence[str], options: SpawnOptions | None = None
        ) -> Result[Process, SubprocessError]:
            self.spawn_calls.append(argv)
            return Err(SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "fake: not spawned"))

    fake = FakeSubprocess()
    ctx = Context()
    ctx.registry.provide("subprocess", fake, _FakeOwner())

    await ctx.plugin(shell_plugin, None)
    result = await ctx.shell.exec("echo hi")

    assert fake.spawn_calls, "LocalShell must dispatch through the mounted ctx.subprocess"
    assert isinstance(result, Err)


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
    by construction, the simplest possible case" rule. `subprocess_plugin` mounts first, since
    `shell_plugin` now depends on it."""
    ctx = Context()
    await ctx.plugin(fs_plugin, None)
    await ctx.plugin(subprocess_plugin, None)
    await ctx.plugin(shell_plugin, None)

    assert ctx.fs.execution_world == ctx.shell.execution_world == ctx.subprocess.execution_world
