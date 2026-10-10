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


# L12D006-I001: the containment wrapper keeps every decorated operation's own call forms.
KEYWORD_CALLS: dict[str, Any] = {
    "read_text_file": lambda fs, p: fs.read_text_file(path=p),
    "read_text_lines": lambda fs, p: fs.read_text_lines(path=p, max_lines=None),
    "read_binary_file": lambda fs, p: fs.read_binary_file(path=p),
    "write_file": lambda fs, p: fs.write_file(path=p, content="w"),
    "append_file": lambda fs, p: fs.append_file(path=p, content="w"),
    "file_info": lambda fs, p: fs.file_info(path=p),
    "list_dir": lambda fs, p: fs.list_dir(path=p),
    "list_dir_raw": lambda fs, p: fs.list_dir_raw(path=p),
    "probe_dir_entry": lambda fs, p: fs.probe_dir_entry(path=p),
    "check_readable": lambda fs, p: fs.check_readable(path=p),
    "check_read_write": lambda fs, p: fs.check_read_write(path=p),
    "canonical_path": lambda fs, p: fs.canonical_path(path=p),
    "create_dir": lambda fs, p: fs.create_dir(path=p, recursive=True),
    "remove": lambda fs, p: fs.remove(path=p, recursive=True, force=False),
    "rename_file": lambda fs, p: fs.rename_file(source=p, destination=p + "-renamed"),
}
POSITIONAL_CALLS: dict[str, Any] = {
    "read_text_file": lambda fs, p: fs.read_text_file(p),
    "read_text_lines": lambda fs, p: fs.read_text_lines(p, None),
    "read_binary_file": lambda fs, p: fs.read_binary_file(p),
    "write_file": lambda fs, p: fs.write_file(p, "w"),
    "append_file": lambda fs, p: fs.append_file(p, "w"),
    "file_info": lambda fs, p: fs.file_info(p),
    "list_dir": lambda fs, p: fs.list_dir(p),
    "list_dir_raw": lambda fs, p: fs.list_dir_raw(p),
    "probe_dir_entry": lambda fs, p: fs.probe_dir_entry(p),
    "check_readable": lambda fs, p: fs.check_readable(p),
    "check_read_write": lambda fs, p: fs.check_read_write(p),
    "canonical_path": lambda fs, p: fs.canonical_path(p),
    "create_dir": lambda fs, p: fs.create_dir(p, True),
    "remove": lambda fs, p: fs.remove(p, True, False),
    "rename_file": lambda fs, p: fs.rename_file(p, p + "-renamed"),
}


def _target(operation: str) -> str:
    return "d" if operation in {"list_dir", "list_dir_raw", "create_dir"} else "f"


def _shape(result: Any, root: Path) -> Any:
    if isinstance(result, Err):
        return ("err", result.error.code, result.error.path)
    value = result.value
    if isinstance(value, str):
        return ("ok", os.path.relpath(value, root) if os.path.isabs(value) else value)
    if isinstance(value, list):
        return ("ok", sorted(getattr(item, "name", item) for item in value))
    return ("ok", type(value).__name__)


@pytest.mark.parametrize("operation", sorted(KEYWORD_CALLS))
async def test_a_keyword_call_behaves_exactly_like_the_positional_call(
    tmp_path: Path, operation: str
) -> None:
    """Ordinary paths: the keyword form gives the same Result as the positional form."""
    outcomes = []
    for name, calls in (("keyword", KEYWORD_CALLS), ("positional", POSITIONAL_CALLS)):
        root = (tmp_path / name).resolve()
        (root / "d").mkdir(parents=True)
        (root / "f").write_text("x")
        outcomes.append(
            _shape(await calls[operation](LocalFileSystem(str(root)), _target(operation)), root)
        )
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("operation", sorted(KEYWORD_CALLS))
@pytest.mark.parametrize("spelling", ["literal", "file-url"])
async def test_a_keyword_call_with_a_nul_path_is_contained(
    tmp_path: Path, operation: str, spelling: str
) -> None:
    root = tmp_path.resolve()
    (root / "f").write_text("x")
    if spelling == "literal":
        argument, logical = f"f{NUL}x", os.path.join(str(root), f"f{NUL}x")
    else:
        argument, logical = f"{root.as_uri()}/f%00x", os.path.join(str(root), f"f{NUL}x")
    result = await KEYWORD_CALLS[operation](LocalFileSystem(str(root)), argument)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.UNKNOWN
    assert result.error.path == logical


async def test_a_keyword_rename_with_a_nul_destination_names_the_source(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "f").write_text("x")
    result = await LocalFileSystem(str(root)).rename_file(
        source="f", destination=f"{root.as_uri()}/g%00x"
    )
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.UNKNOWN
    assert result.error.path == os.path.join(str(root), "f")
    assert (root / "f").read_text() == "x"
