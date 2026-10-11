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

    real_entry = fs_module._rimraf_entry

    def cached_child(path: str) -> Any:
        if path not in listed:  # the remove target itself: never listed
            return real_entry(path)
        if listed[path]:
            return (path, None)
        return fs_module._unlink_routed(path)

    tree = tmp_path / "tree"
    (tree / "child").mkdir(parents=True)
    monkeypatch.setattr(fs_module.os, "scandir", recording)
    fired = _replace_after_listing(monkeypatch, tree, _directory_becomes_file(tree / "child"))
    monkeypatch.setattr(fs_module, "_rimraf_entry", cached_child)
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


# --- L12D007-I005: the walk's interpreter stack does not grow with the tree's depth -------------


def _chain(root: Path, depth: int) -> Path:
    """`root/tree/d/d/.../d` with `depth` nested directories and a file at the bottom."""
    path = root / "tree"
    path.mkdir()
    for _ in range(depth):
        path = path / "d"
        path.mkdir()
    (path / "f").write_text("x")
    return root / "tree"


def _frames() -> int:
    frame, count = sys._getframe(), 0
    while frame is not None:
        count, frame = count + 1, frame.f_back  # type: ignore[assignment]
    return count


def _deepest_rmdir_stack(root: Path, depth: int, monkeypatch: pytest.MonkeyPatch) -> int:
    """The deepest interpreter stack at any `rmdir` while removing a `depth`-deep chain."""
    _chain(root, depth)
    real = fs_module._node_rmdir
    deepest = [0]

    def rmdir(path: str) -> None:
        deepest[0] = max(deepest[0], _frames())
        real(path)

    with monkeypatch.context() as patch:
        patch.setattr(fs_module, "_node_rmdir", rmdir)
        fs_module._remove_sync(str(root / "tree"), recursive=True, force=False)
    assert not (root / "tree").exists()
    return deepest[0]


def test_the_walk_stack_is_independent_of_the_tree_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All platforms: removing a 60-deep chain reaches no deeper interpreter stack than a 2-deep
    one (a recursive walk grows by its frames per level)."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    shallow = _deepest_rmdir_stack(tmp_path / "a", 2, monkeypatch)
    deep = _deepest_rmdir_stack(tmp_path / "b", 60, monkeypatch)
    assert deep == shallow


@posix_only
async def test_a_550_deep_tree_is_removed(tmp_path: Path) -> None:
    """The reviewer's witness (`.tmp/codex-scratch/l12d007-final-depth.py`): 550 nested ordinary
    directories, well within PATH_MAX. Pinned Pi removes them (`{"ok": true}`,
    `l12d007-final-depth-pi.mjs`); a recursive walk raises RecursionError."""
    _chain(tmp_path, 550)
    assert await LocalFileSystem(str(tmp_path)).remove("tree", recursive=True) == Ok(None)
    assert not (tmp_path / "tree").exists()


def _recursive_rimraf(path: str, original: OSError | None = None) -> None:
    """The control: the pre-I005 recursive walk (one interpreter frame pair per tree level)."""
    children = fs_module._rmdir_first(path, original)
    if children is None:
        return
    for child in children:
        descend = fs_module._rimraf_entry(child)
        if descend is not None:
            _recursive_rimraf(*descend)
    fs_module._rmdir_last(path)


def test_control_a_recursive_walk_fails_the_stack_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fs_module, "_rimraf", _recursive_rimraf)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    shallow = _deepest_rmdir_stack(tmp_path / "a", 2, monkeypatch)
    deep = _deepest_rmdir_stack(tmp_path / "b", 60, monkeypatch)
    assert deep > shallow


@posix_only
async def test_control_a_recursive_walk_fails_the_deep_tree_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fs_module, "_rimraf", _recursive_rimraf)
    _chain(tmp_path, 550)
    with pytest.raises(RecursionError):
        await LocalFileSystem(str(tmp_path)).remove("tree", recursive=True)
