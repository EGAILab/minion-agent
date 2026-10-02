"""`L12-D001-R001`: WHICH path an OS-originated `FsError` names -- pinned Pi's `toFileError` takes
Node's `err.path` (the path of the native call that failed), else its fallback (the logical path).
The canonical evidence is `conformance/agent/fs-path-domain/fs-path-error-origin.json`; these tests
pin the mechanisms the corpus cannot reach on every host: each branch of the Node recursive-`mkdir`
walk, and the logical fallback where Node's error carries no path."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from minion_agent.execution import filesystem as fs_module
from minion_agent.execution.errors import FsErrorCode
from minion_agent.execution.filesystem import LocalFileSystem, resolve_local_path
from minion_agent.execution.result import Err, Ok
from minion_agent.runtime.signal import RunAbortController

LONE = "a" + chr(0xD800)
DIRECTORY = os.stat_result((stat.S_IFDIR | 0o755, 0, 0, 0, 0, 0, 0, 0, 0, 0))
REGULAR = os.stat_result((stat.S_IFREG | 0o644, 0, 0, 0, 0, 0, 0, 0, 0, 0))


def _scripted(
    monkeypatch: pytest.MonkeyPatch,
    mkdir: dict[str, OSError | None],
    stats: dict[str, os.stat_result | OSError],
) -> list[str]:
    """Replace the walk's two native calls with a script; returns the mkdir call log."""
    calls: list[str] = []

    def fake_mkdir(path: str) -> None:
        calls.append(path)
        outcome = mkdir[path]
        if outcome is not None:
            raise outcome

    def fake_stat(path: str) -> os.stat_result:
        outcome = stats[path]
        if isinstance(outcome, OSError):
            raise outcome
        return outcome

    monkeypatch.setattr(fs_module.os, "mkdir", fake_mkdir)
    monkeypatch.setattr(fs_module.os, "stat", fake_stat)
    return calls


def _err(code: int, path: str) -> OSError:
    return OSError(code, os.strerror(code), path)


def test_the_walk_creates_every_missing_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    fs_module._node_mkdirp(str(target))
    assert target.is_dir()


def test_an_existing_directory_target_succeeds(tmp_path: Path) -> None:
    fs_module._node_mkdirp(str(tmp_path))
    assert tmp_path.is_dir()


@pytest.mark.parametrize("exc", [PermissionError, NotADirectoryError])
def test_eacces_eperm_enotdir_end_the_walk_naming_the_path(
    monkeypatch: pytest.MonkeyPatch, exc: Callable[..., OSError]
) -> None:
    _scripted(monkeypatch, {"a/b": exc(errno.EACCES, "x", "a/b")}, {})
    with pytest.raises(exc) as raised:
        fs_module._node_mkdirp("a/b")
    assert raised.value.filename == "a/b"


def _enoent(monkeypatch: pytest.MonkeyPatch, once: str, always: str = "") -> list[str]:
    """`os.mkdir` answering ENOENT for `once`'s first attempt and every `always` attempt."""
    calls: list[str] = []

    def mkdir(path: str) -> None:
        first = path not in calls
        calls.append(path)
        if path == always or (path == once and first):
            raise FileNotFoundError(errno.ENOENT, "x", path)

    monkeypatch.setattr(fs_module.os, "mkdir", mkdir)
    return calls


def test_enoent_pushes_the_parent_then_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _enoent(monkeypatch, once="a/b")
    fs_module._node_mkdirp("a/b")
    assert calls == ["a/b", "a", "a/b"]


def test_enoent_without_a_separator_and_more_to_do_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _enoent(monkeypatch, once="x/y", always="x")
    fs_module._node_mkdirp("x/y")  # Node: dirname("x") == "x" with paths left -> next path
    assert calls == ["x/y", "x", "x/y"]


def test_enoent_without_a_separator_as_the_last_path_becomes_eexist_then_stat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted(monkeypatch, {"x": FileNotFoundError(errno.ENOENT, "x", "x")}, {"x": REGULAR})
    with pytest.raises(FileExistsError) as raised:
        fs_module._node_mkdirp("x")
    assert raised.value.filename == "x"


def test_another_error_on_an_existing_directory_ends_the_walk_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted(monkeypatch, {"a": _err(errno.EROFS, "a")}, {"a": DIRECTORY})
    fs_module._node_mkdirp("a")  # Node: Done(0)


def test_an_existing_directory_ancestor_lets_the_walk_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    once = {"a/b": True}
    calls: list[str] = []

    def mkdir(path: str) -> None:
        calls.append(path)
        if once.pop(path, False):
            raise FileNotFoundError(errno.ENOENT, "x", path)
        if path == "a":
            raise FileExistsError(errno.EEXIST, "x", path)

    monkeypatch.setattr(fs_module.os, "mkdir", mkdir)
    monkeypatch.setattr(fs_module.os, "stat", lambda path: DIRECTORY)
    fs_module._node_mkdirp("a/b")
    assert calls == ["a/b", "a", "a/b"]


def test_a_file_ancestor_is_enotdir_naming_the_ancestor(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(
        monkeypatch,
        {"a/b": FileNotFoundError(errno.ENOENT, "x", "a/b"), "a": _err(errno.EEXIST, "a")},
        {"a": REGULAR},
    )
    with pytest.raises(NotADirectoryError) as raised:
        fs_module._node_mkdirp("a/b")
    assert raised.value.filename == "a"


def test_an_unstatable_ancestor_after_eexist_is_enotdir(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(
        monkeypatch,
        {"a/b": FileNotFoundError(errno.ENOENT, "x", "a/b"), "a": _err(errno.EEXIST, "a")},
        {"a": FileNotFoundError(errno.ENOENT, "x", "a")},
    )
    with pytest.raises(NotADirectoryError) as raised:
        fs_module._node_mkdirp("a/b")
    assert raised.value.filename == "a"


def test_an_unstatable_target_reports_the_stat_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(
        monkeypatch,
        {"a": _err(errno.EEXIST, "a")},
        {"a": PermissionError(errno.EACCES, "x", "a")},
    )
    with pytest.raises(PermissionError):
        fs_module._node_mkdirp("a")


def test_a_file_target_is_eexist(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, {"a": _err(errno.EEXIST, "a")}, {"a": REGULAR})
    with pytest.raises(FileExistsError) as raised:
        fs_module._node_mkdirp("a")
    assert raised.value.filename == "a"


@pytest.mark.parametrize("op", ["read_text_file", "read_text_lines", "read_binary_file"])
async def test_a_read_failing_as_a_directory_reports_the_logical_path(
    op: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Node opens a directory and fails the READ with a path-less `EISDIR` (Linux and Windows
    alike in pinned Pi); Python's open raises `IsADirectoryError` naming the native path (POSIX), so
    the provider must report Pi's fallback instead. Forced here so every host exercises it."""

    def is_a_directory(path: str, *_: object) -> str:
        raise IsADirectoryError(errno.EISDIR, "Is a directory", path)

    for name in ("_read_text_sync", "_read_text_lines_sync", "_read_binary_sync"):
        monkeypatch.setattr(fs_module, name, is_a_directory)
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await getattr(fs, op)(LONE)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.IS_DIRECTORY
    assert result.error.path == resolve_local_path(str(tmp_path), LONE)


async def test_removing_a_directory_without_recursive_reports_the_logical_path(
    tmp_path: Path,
) -> None:
    """Node's own `rm` validation names the string it was given. Only the path is asserted here:
    the code is the separately recorded Layer-12 finding (pinned Pi `unknown`)."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    assert isinstance(await fs.create_dir(LONE), Ok)
    result = await fs.remove(LONE)
    assert isinstance(result, Err)
    assert result.error.path == resolve_local_path(str(tmp_path), LONE)


async def test_an_os_failure_in_remove_reports_the_native_path(tmp_path: Path) -> None:
    fs = LocalFileSystem(cwd=str(tmp_path))
    result = await fs.remove(LONE)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.NOT_FOUND
    assert result.error.path == fs_module.native_path(resolve_local_path(str(tmp_path), LONE))


@pytest.mark.parametrize("op", ["read_text_file", "read_binary_file", "write_file"])
async def test_a_pre_aborted_operation_reports_the_resolved_logical_path(
    op: str, tmp_path: Path
) -> None:
    """Pinned Pi's `abortResult(signal, resolved)`: resolved, not the caller's raw string."""
    controller = RunAbortController()
    controller.abort()
    fs = LocalFileSystem(cwd=str(tmp_path))
    args = (LONE, "x") if op == "write_file" else (LONE,)
    result = await getattr(fs, op)(*args, signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == FsErrorCode.ABORTED
    assert result.error.path == resolve_local_path(str(tmp_path), LONE)
