"""`read`/`ls` behavior the canonical scenarios cannot reach: the tools' own abort handling (the
Layer 06 pipeline answers an already-aborted signal before `execute` runs), a live abort mid-call,
provider results no local filesystem produces, and mounting through the plugin."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import Err, FsError, FsErrorCode, LocalFileSystem
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


@pytest.mark.parametrize("tool_name", ["read", "ls"])
async def test_a_live_abort_rejects_while_the_provider_is_still_working(
    tmp_path: Path, tool_name: str
) -> None:
    """Pi rejects the moment the abort event fires, even with I/O still in flight
    (read.ts:237-241, ls.ts:116); the late provider result is discarded."""
    (tmp_path / "a.txt").write_text("x")
    local = LocalFileSystem(str(tmp_path))
    controller = RunAbortController()

    async def slow_probe(path: str, signal: Any = None) -> Any:
        controller.abort()
        await asyncio.sleep(0.5)
        return await local.probe_dir_entry(path, signal)

    provider = _Provider(tmp_path, probe_dir_entry=slow_probe)
    factory = create_read_tool if tool_name == "read" else create_ls_tool
    registry = ToolRegistry()
    registry.register(factory(provider))  # type: ignore[arg-type]
    ctx = Context()
    from minion_agent.tools.events import declare_tools_events

    declare_tools_events(ctx.events)
    arguments = {"path": "a.txt"} if tool_name == "read" else {}
    result = await execute_call(
        ToolCallBlock(id="t", name=tool_name, arguments=arguments),
        registry=registry,
        ctx=ctx,
        signal=controller.signal,
    )
    assert result.is_error
    assert result.content == (TextBlock(text="Operation aborted"),)


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
