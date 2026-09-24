"""`read`/`ls` behavior the canonical scenarios cannot reach: the tools' own abort handling (the
Layer 06 pipeline answers an already-aborted signal before `execute` runs), a live abort mid-call,
provider results no local filesystem produces, and mounting through the plugin."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import (
    Err,
    FileInfo,
    FileKind,
    FsError,
    FsErrorCode,
    LocalFileSystem,
    Ok,
)
from minion_agent.execution.plugin import fs_plugin
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context, FiberState, RunAbortController
from minion_agent.tools.builtin import (
    FsQueryToolsConfig,
    create_ls_tool,
    create_read_tool,
    fs_query_tools_plugin,
)
from minion_agent.tools.builtin.paths import BuiltinToolError
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.plugin import tools_plugin
from minion_agent.tools.registry import ToolRegistry


class _Provider:
    """The real local ctx.fs with selected operations replaced."""

    def __init__(self, root: Path, **overrides: Any) -> None:
        self._local = LocalFileSystem(str(root))
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        return self._overrides.get(name) or getattr(self._local, name)


def _aborted_signal() -> Any:
    controller = RunAbortController()
    controller.abort()
    return controller.signal


@pytest.mark.parametrize("factory", [create_read_tool, create_ls_tool])
async def test_execute_itself_rejects_an_already_aborted_signal(
    tmp_path: Path, factory: Any
) -> None:
    """Pi's tools check `signal.aborted` at the top of `execute` (read.ts:232, ls.ts:111)."""
    tool = factory(LocalFileSystem(str(tmp_path)))
    with pytest.raises(BuiltinToolError, match=r"^Operation aborted$"):
        await tool.execute("id", {"path": "."}, _aborted_signal())


async def test_read_maps_an_aborted_read_to_operation_aborted(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")

    async def aborted_read(path: str, signal: Any = None) -> Any:
        return Err(FsError(FsErrorCode.ABORTED, "aborted", path))

    tool = create_read_tool(_Provider(tmp_path, read_binary_file=aborted_read))  # type: ignore[arg-type]
    with pytest.raises(BuiltinToolError, match=r"^Operation aborted$"):
        await tool.execute("id", {"path": "a.txt"}, None)


async def test_ls_maps_an_aborted_enumeration_to_operation_aborted(tmp_path: Path) -> None:
    async def aborted_list(path: str, signal: Any = None) -> Any:
        return Err(FsError(FsErrorCode.ABORTED, "aborted", path))

    tool = create_ls_tool(_Provider(tmp_path, list_dir_raw=aborted_list))  # type: ignore[arg-type]
    with pytest.raises(BuiltinToolError, match=r"^Operation aborted$"):
        await tool.execute("id", {}, None)


async def test_ls_skips_an_entry_whose_path_cannot_be_joined(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("x")
    (tmp_path / "b").write_text("x")
    local = LocalFileSystem(str(tmp_path))

    async def join_path(parts: list[str], signal: Any = None) -> Any:
        if parts[-1] == "a":
            return Err(FsError(FsErrorCode.INVALID, "invalid", None))
        return await local.join_path(parts, signal)

    tool = create_ls_tool(_Provider(tmp_path, join_path=join_path))  # type: ignore[arg-type]
    result = await tool.execute("id", {}, None)
    assert result.content == (TextBlock(text="b"),)


async def test_read_falls_back_to_the_pipeline_path_when_absolute_path_fails(
    tmp_path: Path,
) -> None:
    async def no_absolute(path: str, signal: Any = None) -> Any:
        return Err(FsError(FsErrorCode.NOT_SUPPORTED, "no", path))

    tool = create_read_tool(_Provider(tmp_path, absolute_path=no_absolute))  # type: ignore[arg-type]
    with pytest.raises(
        BuiltinToolError, match=r"^Cannot access missing.txt: no such file or directory$"
    ):
        await tool.execute("id", {"path": "missing.txt"}, None)


class _Recording(_Provider):
    """The real local ctx.fs; `blocked` operations wait for `release` before running, and every
    call is logged when it STARTS and when it FINISHES (a cancelled call never logs `done`)."""

    def __init__(self, root: Path, blocked: str) -> None:
        super().__init__(root)
        self.blocked = blocked
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.log: list[str] = []

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._local, name)

        async def recorded(*args: Any, **kwargs: Any) -> Any:
            self.log.append(
                f"start {name} signal={kwargs.get('signal') or (args[1:] or [None])[0]}"
            )
            if name == self.blocked:
                self.entered.set()
                await self.release.wait()
            result = await method(*args, **kwargs)
            self.log.append(f"done {name}")
            return result

        return recorded


async def _abort_while_blocked(tool_name: str, provider: _Recording) -> Any:
    registry = ToolRegistry()
    factory = create_read_tool if tool_name == "read" else create_ls_tool
    registry.register(factory(provider))  # type: ignore[arg-type]
    ctx = Context()
    declare_tools_events(ctx.events)
    controller = RunAbortController()
    arguments = {"path": "a.txt"} if tool_name == "read" else {}
    call = asyncio.ensure_future(
        execute_call(
            ToolCallBlock(id="t", name=tool_name, arguments=arguments),
            registry=registry,
            ctx=ctx,
            signal=controller.signal,
        )
    )
    await asyncio.wait_for(provider.entered.wait(), timeout=5)
    controller.abort()
    result = await asyncio.wait_for(call, timeout=5)  # answered while the provider is still blocked
    assert not provider.release.is_set()
    return result


async def _settle(provider: _Recording, last: str) -> None:
    provider.release.set()
    for _ in range(500):
        if provider.log and provider.log[-1] == last:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(provider.log)


async def test_read_abort_answers_at_once_leaves_the_access_check_running_then_stops(
    tmp_path: Path,
) -> None:
    """L13-WP131-I001: Pi rejects the moment the signal fires but does not cancel the pending
    `access` (read.ts:237-248); when it completes, the `if (aborted) return` checkpoint stops the
    work, so the file is never read. No ctx.fs call is given the signal."""
    (tmp_path / "a.txt").write_text("x")
    provider = _Recording(tmp_path, blocked="file_info")
    result = await _abort_while_blocked("read", provider)
    assert (result.is_error, result.content) == (True, (TextBlock(text="Operation aborted"),))
    await _settle(provider, "done file_info")
    await asyncio.sleep(0.05)
    assert provider.log == ["start file_info signal=None", "done file_info"]


async def test_ls_abort_answers_at_once_and_the_listing_work_runs_to_completion(
    tmp_path: Path,
) -> None:
    """ls.ts has no checkpoints: after the rejection its work carries on through enumeration and
    every per-entry probe, and the result is discarded (L13-WP131-I001)."""
    (tmp_path / "a.txt").write_text("x")
    provider = _Recording(tmp_path, blocked="probe_dir_entry")
    result = await _abort_while_blocked("ls", provider)
    assert (result.is_error, result.content) == (True, (TextBlock(text="Operation aborted"),))
    await _settle(provider, "done probe_dir_entry")
    for _ in range(500):
        if provider.log.count("done probe_dir_entry") == 2:
            break
        await asyncio.sleep(0.01)
    started = [line for line in provider.log if line.startswith("start")]
    assert started == [
        "start absolute_path signal=None",
        "start probe_dir_entry signal=None",
        "start list_dir_raw signal=None",
        "start join_path signal=None",
        "start probe_dir_entry signal=None",
    ]
    assert provider.log.count("done probe_dir_entry") == 2


async def test_read_stops_at_the_path_checkpoint_when_aborted_before_access(tmp_path: Path) -> None:
    """read.ts:246: an abort during path resolution stops before the access check."""
    from minion_agent.tools.builtin import read as read_module

    controller = RunAbortController()
    reader = read_module._Read(LocalFileSystem(str(tmp_path)), read_module.ReadToolOptions())
    controller.abort()
    assert await reader.run("a.txt", None, None, controller.signal) is None


async def test_execute_reports_abort_when_the_work_stopped_at_a_checkpoint(tmp_path: Path) -> None:
    """If the work reaches a checkpoint before the race observes the abort, the stopped work's
    `None` still becomes Pi's rejection."""
    (tmp_path / "a.txt").write_text("x")
    controller = RunAbortController()

    async def aborting_file_info(path: str, signal: Any = None) -> Any:
        # Aborts and answers without suspending, so the work reaches its checkpoint before the
        # race's poll can observe the abort.
        controller.abort()
        return Ok(FileInfo(name="a.txt", path=path, kind=FileKind.FILE, size=1, mtime_ms=0.0))

    tool = create_read_tool(_Provider(tmp_path, file_info=aborting_file_info))  # type: ignore[arg-type]
    with pytest.raises(BuiltinToolError, match=r"^Operation aborted$"):
        await tool.execute("id", {"path": "a.txt"}, controller.signal)


async def test_plugin_registers_read_and_ls_over_the_mounted_fs(tmp_path: Path) -> None:
    ctx = Context()
    await ctx.plugin(tools_plugin, None)
    await ctx.plugin(fs_plugin, None)
    fiber = await ctx.plugin(
        fs_query_tools_plugin, FsQueryToolsConfig(model_supports_images=lambda: False)
    )
    assert fiber.state is FiberState.ACTIVE
    registry: ToolRegistry = ctx.tools
    assert registry.resolve("read") is not None and registry.resolve("ls") is not None
    assert [schema.name for schema in registry.schemas()] == ["read", "ls"]

    await ctx.plugins.unmount(fiber)
    assert registry.resolve("read") is None and registry.resolve("ls") is None
