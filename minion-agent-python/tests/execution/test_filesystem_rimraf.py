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
from typing import Any

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


@posix_only
def test_posix_rmdir_enotdir_after_an_unlink_error_reports_that_error(tmp_path: Path) -> None:
    """rimraf `_rmdir(path, originalErr)` from `unlink`'s EISDIR / EPERM: ENOTDIR answers the
    unlink's own error."""
    (tmp_path / "f").write_text("x")
    original = PermissionError(1, "Operation not permitted", str(tmp_path / "f"))
    with pytest.raises(PermissionError) as caught:
        fs_module._rimraf(str(tmp_path / "f"), original)
    assert caught.value is original


# --- L12D007-I002: each child is classified afresh, by name (rimraf `_rimraf`'s own lstat) -------


def _replace_after_listing(
    monkeypatch: pytest.MonkeyPatch, tree: Path, replace: Callable[[], None]
) -> list[bool]:
    """The real listing of `tree` runs; `replace` runs when the listing closes -- after enumeration,
    before any child is handled. No sleep or scheduling guess."""
    real_scandir = os.scandir
    fired: list[bool] = []

    class _Listing:
        def __init__(self, path: str) -> None:
            self._real = real_scandir(path)

        def __enter__(self) -> Any:
            return self._real.__enter__()

        def __exit__(self, *exc: object) -> None:
            self._real.__exit__(*exc)
            replace()
            fired.append(True)

    def scandir(path: Any = None) -> Any:
        if path is not None and os.path.normcase(os.fspath(path)) == os.path.normcase(str(tree)):
            return _Listing(os.fspath(path))
        return real_scandir(path) if path is not None else real_scandir()

    monkeypatch.setattr(fs_module.os, "scandir", scandir)
    return fired


def _directory_becomes_file(child: Path) -> Callable[[], None]:
    def replace() -> None:
        child.rmdir()
        child.write_bytes(b"replaced")

    return replace


async def _remove_tree_whose_child_dir_becomes_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, list[bool]]:
    tree = tmp_path / "tree"
    (tree / "child").mkdir(parents=True)
    fired = _replace_after_listing(monkeypatch, tree, _directory_becomes_file(tree / "child"))
    return await LocalFileSystem(str(tmp_path)).remove("tree", recursive=True), fired


async def test_a_child_directory_replaced_by_a_file_after_listing_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned Pi on Windows and Linux (`.tmp/codex-scratch/l12d007-impl-race.mjs`, review
    issuecomment-6102670593): `ok`, the tree removed."""
    result, fired = await _remove_tree_whose_child_dir_becomes_a_file(tmp_path, monkeypatch)
    assert fired == [True]
    assert result == Ok(None)
    assert not (tmp_path / "tree").exists()


async def test_control_cached_listing_type_fails_the_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutant: dispatch on the type the LISTING saw (the reviewed defect). The witness fails: the
    replacement file remains and the removal reports an error."""
    listed: dict[str, bool] = {}
    real_scandir = os.scandir

    def recording(path: Any = None) -> Any:
        for entry in real_scandir(path):
            listed[entry.path] = entry.is_dir(follow_symlinks=False)
        return real_scandir(path)

    def cached_child(path: str) -> None:
        if listed[path]:
            fs_module._rimraf(path)
        else:
            fs_module._unlink_entry(path)

    tree = tmp_path / "tree"
    (tree / "child").mkdir(parents=True)
    monkeypatch.setattr(fs_module.os, "scandir", recording)
    fired = _replace_after_listing(monkeypatch, tree, _directory_becomes_file(tree / "child"))
    monkeypatch.setattr(fs_module, "_rimraf_child", cached_child)
    result = await LocalFileSystem(str(tmp_path)).remove("tree", recursive=True)
    assert fired == [True]
    assert result != Ok(None)
    assert (tree / "child").is_file()


async def test_a_child_file_replaced_by_a_directory_after_listing_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inverse replacement: the child is removed as the directory it now is."""
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "child").write_text("x")

    def replace() -> None:
        (tree / "child").unlink()
        (tree / "child").mkdir()
        (tree / "child" / "inner").write_text("y")

    fired = _replace_after_listing(monkeypatch, tree, replace)
    assert await LocalFileSystem(str(tmp_path)).remove("tree", recursive=True) == Ok(None)
    assert fired == [True]
    assert not tree.exists()


async def test_a_child_replaced_by_a_directory_link_is_removed_without_following(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-follow neighbourhood: a child directory replaced by a link to a directory OUTSIDE
    the tree is unlinked as itself; the link's target and its content survive."""
    tree = tmp_path / "tree"
    (tree / "child").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("k")

    def replace() -> None:
        (tree / "child").rmdir()
        os.symlink(outside, tree / "child", target_is_directory=True)

    fired = _replace_after_listing(monkeypatch, tree, replace)
    assert await LocalFileSystem(str(tmp_path)).remove("tree", recursive=True) == Ok(None)
    assert fired == [True]
    assert not tree.exists()
    assert (outside / "keep").read_text() == "k"


def test_a_child_gone_before_its_lstat_counts_as_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs_module._rimraf_child(str(tmp_path / "missing"))


def test_an_lstat_failure_still_goes_on_to_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rimraf: an `lstat` error other than ENOENT falls through to `unlink`."""
    target = tmp_path / "f"
    target.write_text("x")
    real_lstat = os.lstat

    def refused(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if os.fspath(path) == str(target):
            raise PermissionError(13, "Permission denied", str(target))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(fs_module.os, "lstat", refused)
    fs_module._rimraf_child(str(target))
    assert not target.exists()


@windows_only
def test_an_unlink_that_meets_a_directory_removes_it_as_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child became a directory between its `lstat` and its `unlink`: libuv's unlink refuses
    a directory (EPERM), and `fixWinEPERM`'s `stat` finds a directory, so `_rmdir` removes it."""
    target = tmp_path / "d"
    (target / "inner").mkdir(parents=True)
    monkeypatch.setattr(fs_module, "_is_tree", lambda st: False)
    fs_module._rimraf_child(str(target))
    assert not target.exists()


@windows_only
@pytest.mark.parametrize(
    ("stat_answer", "removed"),
    [
        pytest.param(None, False, id="stat-finds-a-file"),
        pytest.param(32, False, id="stat-fails"),
        pytest.param(2, True, id="stat-finds-it-gone"),
    ],
)
def test_an_eperm_unlink_of_a_non_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stat_answer: int | None, removed: bool
) -> None:
    """`fixWinEPERM`: a `stat` that finds a file, or fails, reports the unlink's own error; one
    that finds the entry gone counts as removed."""
    target = tmp_path / "f"
    target.write_text("x")

    def refused(path: str) -> None:
        raise OSError(0, "Access is denied", path, 5)

    monkeypatch.setattr(fs_module, "_unlink_entry", refused)
    if stat_answer is not None:
        code = stat_answer

        def stat_fails(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
            raise OSError(0, "scripted", os.fspath(path), code)

        monkeypatch.setattr(fs_module.os, "stat", stat_fails)
    if removed:
        fs_module._rimraf_child(str(target))
        return
    with pytest.raises(OSError) as caught:
        fs_module._rimraf_child(str(target))
    assert (caught.value.winerror, caught.value.filename) == (5, str(target))


@windows_only
async def test_a_junction_in_the_tree_is_removed_without_following(tmp_path: Path) -> None:
    """libuv's `lstat` reports a junction as a link, so rimraf `unlink`s it as itself: the
    junction's target and its content survive."""
    import _winapi  # type: ignore[import-not-found,unused-ignore]

    tree = tmp_path / "tree"
    tree.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("k")
    _winapi.CreateJunction(str(outside), str(tree / "junction"))
    assert await LocalFileSystem(str(tmp_path)).remove("tree", recursive=True) == Ok(None)
    assert not tree.exists()
    assert (outside / "keep").read_text() == "k"
