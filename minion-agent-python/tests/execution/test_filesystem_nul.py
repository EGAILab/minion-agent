"""`L12-D006` (spec/execution.md section 18; minion-agent#194, provenance #65 + #133-F1) binding
witnesses beside the canonical `fs-path-nul.json` corpus: temporary-file creation (no canonical
fixture can name the system temp location), the tool boundary (its texts embed the case directory),
and the Owner's "do not catch every ValueError" requirement."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import minion_agent.execution.filesystem as filesystem_module
from minion_agent.execution import Err, FsErrorCode, LocalFileSystem, Ok
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools.builtin import (
    create_edit_tool,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
)
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

NUL = "\x00"


async def test_a_nul_temp_dir_prefix_is_unknown_without_a_path(tmp_path: Path) -> None:
    result = await LocalFileSystem(str(tmp_path)).create_temp_dir(f"t{NUL}x")
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.UNKNOWN
    assert result.error.path is None


@pytest.mark.parametrize("where", ["prefix", "suffix"])
async def test_a_nul_temp_file_name_is_unknown_naming_the_would_be_file(
    tmp_path: Path, where: str
) -> None:
    result = await LocalFileSystem(str(tmp_path)).create_temp_file(**{where: f"t{NUL}x"})
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.UNKNOWN
    path = result.error.path
    assert path is not None and NUL in path
    # Pi: the temporary directory is created first; the file write is what rejects the name.
    directory = os.path.dirname(path)
    assert os.path.isdir(directory)
    os.rmdir(directory)


async def test_an_unrelated_value_error_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the host's NUL rejection of a NUL-containing argument becomes `unknown`."""
    (tmp_path / "f").write_text("x")

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("embedded null character in path")

    monkeypatch.setattr(filesystem_module, "_read_text_sync", broken)
    with pytest.raises(ValueError, match="embedded null"):
        await LocalFileSystem(str(tmp_path)).read_text_file("f")


async def _tool(root: Path, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    fs = LocalFileSystem(str(root))
    registry = ToolRegistry()
    for create in (create_read_tool, create_write_tool, create_edit_tool, create_ls_tool):
        registry.register(create(fs))
    ctx = Context()
    declare_tools_events(ctx.events)
    result = await execute_call(
        ToolCallBlock(id="c", name=name, arguments=arguments), registry=registry, ctx=ctx
    )
    first = result.content[0]
    return result.is_error, first.text if isinstance(first, TextBlock) else ""


async def test_read_with_a_nul_path_reports_unknown_at_the_access_site(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    absolute = os.path.join(str(root), f"f{NUL}x")
    assert await _tool(root, "read", {"path": f"f{NUL}x"}) == (
        True,
        f"Cannot access {absolute}: unknown filesystem error",
    )


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("write", {"path": f"new/a{NUL}b", "content": "w"}),
        ("edit", {"path": f"f{NUL}x", "edits": [{"oldText": "x", "newText": "y"}]}),
    ],
)
async def test_write_and_edit_with_a_nul_path_fail_at_the_queue_key_before_any_parent(
    tmp_path: Path, name: str, arguments: dict[str, Any]
) -> None:
    root = tmp_path.resolve()
    (root / "f").write_text("x")
    assert await _tool(root, name, arguments) == (
        True,
        f"Cannot resolve {arguments['path']}: unknown filesystem error",
    )
    assert not (root / "new").exists()  # Pi's getMutationQueueKey fails before mkdir


async def test_ls_with_a_nul_path_is_path_not_found(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    absolute = os.path.join(str(root), f"d{NUL}x")
    assert await _tool(root, "ls", {"path": f"d{NUL}x"}) == (True, f"Path not found: {absolute}")


async def test_a_lexical_operation_keeps_the_nul_string(tmp_path: Path) -> None:
    result = await LocalFileSystem(str(tmp_path)).absolute_path(f"f{NUL}x")
    assert isinstance(result, Ok)
    assert result.value == os.path.join(str(tmp_path), f"f{NUL}x")


async def test_an_unrelated_value_error_in_temp_creation_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_mkdtemp(*args: Any, **kwargs: Any) -> str:
        raise ValueError("embedded null character")

    monkeypatch.setattr(filesystem_module.tempfile, "mkdtemp", broken_mkdtemp)
    with pytest.raises(ValueError, match="embedded null"):
        await LocalFileSystem(str(tmp_path)).create_temp_dir("plain-")


async def test_an_unrelated_value_error_in_temp_file_creation_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_write(*args: Any, **kwargs: Any) -> None:
        raise ValueError("embedded null character")

    monkeypatch.setattr(filesystem_module, "_write_file_sync", broken_write)
    with pytest.raises(ValueError, match="embedded null"):
        await LocalFileSystem(str(tmp_path)).create_temp_file(prefix="plain")
