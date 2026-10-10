"""L12-D007: a recursive `remove` of a directory follows pinned Node's `rimraf` order (v22.15.1
`lib/internal/fs/rimraf.js`, `_rmdir` / `_rmchildren`): `rmdir` FIRST; only ENOTEMPTY / EEXIST /
EPERM lists the directory and removes its children; ENOENT anywhere counts as removed."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from minion_agent.execution import LocalFileSystem, Ok
from minion_agent.execution import filesystem as fs_module

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="libuv's Win32 rmdir seam")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rmdir(2)")


@contextmanager
def _unlistable(path: Path) -> Iterator[None]:
    """Listing denied, everything else allowed: a deny-`RD` (`FILE_LIST_DIRECTORY`) ACE on
    Windows, mode 000 on POSIX. Restored if the directory is still there."""
    if sys.platform == "win32":
        user = os.environ["USERNAME"]
        subprocess.run(
            ["icacls", str(path), "/deny", f"{user}:(RD)"], check=True, capture_output=True
        )
    else:
        os.chmod(path, 0)
    try:
        yield
    finally:
        if path.exists():
            if sys.platform == "win32":
                subprocess.run(["icacls", str(path), "/reset"], check=True, capture_output=True)
            else:
                os.chmod(path, stat.S_IRWXU)


@pytest.mark.skipif(
    sys.platform != "win32" and os.geteuid() == 0, reason="root bypasses POSIX permission bits"
)
async def test_an_empty_unlistable_directory_is_removed(tmp_path: Path) -> None:
    """Pinned Pi removes it: `rmdir` succeeds before any listing is attempted. A walk that lists
    first (`shutil.rmtree`, `os.scandir`) fails with permission_denied instead."""
    target = tmp_path / "d"
    target.mkdir()
    with _unlistable(target):
        assert await LocalFileSystem(str(tmp_path)).remove("d", recursive=True) == Ok(None)
    assert not target.exists()


def _win32(code: int, path: str) -> OSError:
    return OSError(0, "scripted", path, code)


def _rmdir_plan(monkeypatch: pytest.MonkeyPatch, plan: list[Callable[[str], None]]) -> list[str]:
    """`_libuv_rmdir` answers by `plan`, one step per call."""
    calls: list[str] = []

    def rmdir(path: str) -> None:
        calls.append(path)
        plan.pop(0)(path)

    monkeypatch.setattr(fs_module, "_libuv_rmdir", rmdir)
    return calls


def _raise(code: int) -> Callable[[str], None]:
    def step(path: str) -> None:
        raise _win32(code, path)

    return step


@windows_only
def test_rmdir_finding_the_directory_gone_counts_as_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _rmdir_plan(monkeypatch, [_raise(2)])
    fs_module._rimraf(str(tmp_path / "d"))
    assert calls == [str(tmp_path / "d")]


@windows_only
def test_an_rmdir_failure_other_than_not_empty_or_eperm_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """32 (sharing violation) is libuv's EBUSY: rimraf does not list the directory."""
    (tmp_path / "d").mkdir()
    _rmdir_plan(monkeypatch, [_raise(32)])
    with pytest.raises(OSError) as caught:
        fs_module._rimraf(str(tmp_path / "d"))
    assert caught.value.winerror == 32


@windows_only
def test_a_directory_gone_before_its_listing_counts_as_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rmdir_plan(monkeypatch, [_raise(145)])
    fs_module._rimraf(str(tmp_path / "missing"))


@windows_only
def test_a_child_gone_before_its_unlink_counts_as_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("x")
    real = fs_module._unlink_entry

    def vanished(path: str) -> None:
        real(path)
        raise _win32(2, path)

    monkeypatch.setattr(fs_module, "_unlink_entry", vanished)
    fs_module._rimraf(str(tmp_path / "d"))
    assert not (tmp_path / "d").exists()


@windows_only
def test_a_child_unlink_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("x")

    def refused(path: str) -> None:
        raise _win32(32, path)

    monkeypatch.setattr(fs_module, "_unlink_entry", refused)
    with pytest.raises(OSError) as caught:
        fs_module._rimraf(str(tmp_path / "d"))
    assert (caught.value.winerror, caught.value.filename) == (32, str(tmp_path / "d" / "f"))


@windows_only
@pytest.mark.parametrize(("code", "removed"), [(2, True), (32, False)])
def test_the_final_rmdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, removed: bool
) -> None:
    """After the children: ENOENT counts as removed; any other failure is reported."""
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("x")
    real = fs_module._libuv_rmdir

    def removed_then(path: str) -> None:
        if code == 2:
            real(path)
        raise _win32(code, path)

    _rmdir_plan(monkeypatch, [_raise(145), removed_then])
    if removed:
        fs_module._rimraf(str(tmp_path / "d"))
    else:
        with pytest.raises(OSError) as caught:
            fs_module._rimraf(str(tmp_path / "d"))
        assert caught.value.winerror == code
    assert (tmp_path / "d").exists() is not removed


@posix_only
def test_posix_rmdir_enotdir_answers_lstats_absent_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rimraf `_rmdir`: ENOTDIR (the entry was replaced after `lstat`) calls back with `lstat`'s
    own error, which was none."""
    (tmp_path / "f").write_text("x")
    fs_module._rimraf(str(tmp_path / "f"))
    assert (tmp_path / "f").exists()
