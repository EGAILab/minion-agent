"""L12D007-C001: the canonical runner's fixture steps never reach outside their case directory.

Permanent rejecting controls for `fs_path_runner._fixture_target`. Links are created INSIDE the
test's own directory (some pointing outside) and only the guard is asked about them; nothing is ever
written, renamed or deleted through a link, and the outside targets are never created."""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tests.conformance.fs_path_runner as _runner
from tests.conformance.fs_path_runner import _fixture_target

OUTSIDE = (
    "C:\\l12d007-runner-never-created"
    if sys.platform == "win32"
    else "/l12d007-runner-never-created"
)


@pytest.mark.parametrize(
    "raw", ["", "..", "a/../b", "a\\..\\b", "C:x", "E:", "E:\\", "/abs", "\\abs", "\\\\srv\\share"]
)
def test_prohibited_raw_forms_are_refused_before_normalization(tmp_path: Path, raw: str) -> None:
    with pytest.raises(AssertionError, match="prohibited raw form"):
        _fixture_target(str(tmp_path), raw)


@pytest.mark.parametrize("name", ["f", "d/c", "./f:stream:bad", "x<y", "n" * 300])
def test_ordinary_fixture_names_are_accepted(tmp_path: Path, name: str) -> None:
    assert _fixture_target(str(tmp_path), name).startswith(str(tmp_path))


def test_a_dangling_link_pointing_outside_is_refused(tmp_path: Path) -> None:
    os.symlink(OUTSIDE, tmp_path / "out")
    with pytest.raises(AssertionError, match="through a link"):
        _fixture_target(str(tmp_path), "out")


def test_a_two_hop_escape_is_refused(tmp_path: Path) -> None:
    os.symlink("mid", tmp_path / "hop")
    os.symlink(OUTSIDE, tmp_path / "mid")
    with pytest.raises(AssertionError, match="through a link"):
        _fixture_target(str(tmp_path), "hop")


def test_a_directory_link_to_the_filesystem_root_is_refused(tmp_path: Path) -> None:
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction("C:\\", str(tmp_path / "j"))
    else:
        os.symlink("/", tmp_path / "j", target_is_directory=True)
    with pytest.raises(AssertionError, match="through a link"):
        _fixture_target(str(tmp_path), "j/x")


def test_a_loop_that_stays_inside_is_accepted(tmp_path: Path) -> None:
    os.symlink("b", tmp_path / "a")
    os.symlink("a", tmp_path / "b")
    assert _fixture_target(str(tmp_path), "a") == str(tmp_path / "a")


def test_a_symlink_text_is_checked_from_the_link_directory(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    assert _fixture_target(str(tmp_path), "x", str(tmp_path / "d")) == str(tmp_path / "d" / "x")
    with pytest.raises(AssertionError):
        _fixture_target(str(tmp_path), "..", str(tmp_path / "d"))


# --- CE-L12D007-01 synthetic controls: virtual link metadata below a root that is never created;
# --- nothing is written, renamed or deleted anywhere.

OUT_FILE = "C:\\ce01-never-created" if sys.platform == "win32" else "/ce01-never-created"


class _Virtual:
    """Answers lstat / readlink below `root` from a tree; delegates everything else."""

    def __init__(self, root: Path, tree: dict[str, dict[str, Any]]) -> None:
        self.root = os.path.abspath(root)
        self.tree = tree
        self.lstat, self.readlink = os.lstat, os.readlink

    def _rel(self, p: Any) -> str | None:
        a = os.path.abspath(str(p))
        if os.path.normcase(a) == os.path.normcase(self.root):
            return ""
        try:
            if os.path.commonpath(
                [os.path.normcase(self.root), os.path.normcase(a)]
            ) != os.path.normcase(self.root):
                return None
        except ValueError:
            return None
        return os.path.relpath(a, self.root).replace("\\", "/")

    def fake_lstat(self, p: Any, *a: Any, **k: Any) -> Any:
        r = self._rel(p)
        if r is None:
            return self.lstat(p, *a, **k)
        if r == "":
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_file_attributes=0)
        parts = r.split("/")
        for i in range(1, len(parts)):
            pre = self.tree.get("/".join(parts[:i]))
            if pre is None:
                raise FileNotFoundError(errno.ENOENT, "synthetic", str(p))
            if pre.get("err"):
                raise OSError(pre["err"], "synthetic", str(p))
            if pre["type"] == "file":
                raise NotADirectoryError(errno.ENOTDIR, "synthetic", str(p))
        e = self.tree.get(r)
        if e is None:
            raise FileNotFoundError(errno.ENOENT, "synthetic", str(p))
        if e.get("err"):
            raise OSError(e["err"], "synthetic", str(p))
        mode = {"link": stat.S_IFLNK, "dir": stat.S_IFDIR, "file": stat.S_IFREG}[e["type"]]
        return SimpleNamespace(st_mode=mode | 0o700, st_file_attributes=0)

    def fake_readlink(self, p: Any, *a: Any, **k: Any) -> Any:
        r = self._rel(p)
        if r is None:
            return self.readlink(p, *a, **k)
        return self.tree[r]["text"]


@pytest.fixture
def virtual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(tree: dict[str, dict[str, Any]]) -> str:
        v = _Virtual(tmp_path / "virtual", {_runner._project(k): val for k, val in tree.items()})
        monkeypatch.setattr(os, "lstat", v.fake_lstat)
        monkeypatch.setattr(os, "readlink", v.fake_readlink)
        return v.root

    return install


def _chain(n: int, last: str) -> dict[str, dict[str, Any]]:
    return {f"c{i}": {"type": "link", "text": f"c{i + 1}" if i + 1 < n else last} for i in range(n)}


def test_synthetic_long_acyclic_outward_chain_is_refused(virtual: Any) -> None:
    root = virtual(_chain(42, OUT_FILE))
    with pytest.raises(AssertionError):
        _fixture_target(root, "c0")


def test_synthetic_budget_exhaustion_without_a_repeat_is_refused(virtual: Any) -> None:
    root = virtual(_chain(_runner._BUDGET + 1, "missing"))
    with pytest.raises(AssertionError, match="budget"):
        _fixture_target(root, "c0")


def test_synthetic_outward_last_hop_is_refused(virtual: Any) -> None:
    root = virtual(_chain(_runner._BUDGET, OUT_FILE))
    with pytest.raises(AssertionError):
        _fixture_target(root, "c0")


def test_synthetic_contained_cycle_is_accepted(virtual: Any) -> None:
    root = virtual({f"k{i}": {"type": "link", "text": f"k{(i + 1) % 10}"} for i in range(10)})
    _fixture_target(root, "k0")


@pytest.mark.parametrize("code", [errno.EACCES, errno.EPERM, errno.EBUSY])
def test_synthetic_inspection_failure_is_refused_not_missing(virtual: Any, code: int) -> None:
    root = virtual({"d": {"type": "dir", "err": code}})
    with pytest.raises(AssertionError, match="cannot inspect"):
        _fixture_target(root, "d/x")


def test_synthetic_provider_write_through_outward_link_refused_entry_remove_allowed(
    virtual: Any,
) -> None:
    root = virtual({"j": {"type": "link", "text": OUT_FILE}})
    with pytest.raises(AssertionError):
        _runner._provider_target(root, "j", "write_file")  # REFERENT through the outward final link
    with pytest.raises(AssertionError):
        _runner._provider_target(root, "j/x", "write_file")
    _runner._provider_target(root, "j", "remove")  # ENTRY: the link itself
    with pytest.raises(AssertionError):
        _runner._provider_target(root, "j/x", "remove")  # its parent resolves outside


def test_synthetic_projection_alias_is_proven_as_the_native_spelling(virtual: Any) -> None:
    root = virtual({"a\ufffd": {"type": "link", "text": OUT_FILE}})
    for lone in ("a\ud800", "a\udc00"):
        with pytest.raises(AssertionError):
            _runner._provider_target(root, lone, "write_file")


def test_synthetic_restore_is_skipped_for_a_missing_or_linked_entry(virtual: Any) -> None:
    root = virtual({"l": {"type": "link", "text": "x"}})
    calls: list[str] = []
    _runner._restore_access(root, "gone", "file", calls.append)
    _runner._restore_access(root, "l", "file", calls.append)
    assert calls == []


async def test_runner_refuses_a_provider_step_through_an_outward_link_before_the_provider_runs(
    virtual: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Kills `runner-skips-provider-target`: the provider must never be reached."""
    reached: list[str] = []

    async def record(self: Any, path: str, *a: Any, **k: Any) -> Any:
        reached.append(path)
        raise AssertionError("provider reached")

    monkeypatch.setattr(_runner.LocalFileSystem, "write_file", record)
    root = virtual({"j": {"type": "link", "text": OUT_FILE}})
    case = {
        "id": "x",
        "steps": [{"op": "write_file", "path": {"utf16": [106, 47, 120]}, "content": [120]}],
    }
    with pytest.raises(AssertionError, match="containment"):
        await _runner.run_case(case, Path(root))
    assert reached == []
