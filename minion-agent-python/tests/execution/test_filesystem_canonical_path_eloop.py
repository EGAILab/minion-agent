"""`L12-D004` (`EXEC-002`): `canonical_path` resolves through the OS `realpath(3)` on POSIX, as
pinned Pi's `canonicalPath` does (`fs/promises.realpath` -> libuv `uv_fs_realpath`).

Characterization (minion-agent-docs assurance/layers/12-l12-d004-canonical-path.md): on Linux the
previous `os.path.realpath(strict=True)` differed from Pi on exactly two of 44 probed paths, both
symlink-traversal-limit (`ELOOP`) cases -- a cycle walked 41 levels deep and an acyclic chain of 41
links -- and agreed everywhere else; the OS `realpath(3)` agrees with Pi on all 44. Windows
resolution is unchanged.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

import minion_agent.execution.filesystem as filesystem_module
from minion_agent.execution import Err, FsErrorCode, LocalFileSystem, Ok


class _FakeLibc:
    """A stand-in for libc `realpath(3)`/`free(3)` that routes the POSIX branch on any host."""

    def __init__(self, result: str | None, errno_value: int = 0) -> None:
        self.result = result
        self.errno_value = errno_value
        self.calls: list[bytes] = []
        self.freed: list[int] = []
        self._buffer = (
            ctypes.create_string_buffer(os.fsencode(result)) if result is not None else None
        )

    def realpath(self, path: bytes, resolved: object) -> int | None:
        assert resolved is None  # always the allocating form
        self.calls.append(path)
        if self._buffer is None:
            ctypes.set_errno(self.errno_value)
            return None
        return ctypes.addressof(self._buffer)

    def free(self, pointer: int) -> None:
        self.freed.append(pointer)


def _route(monkeypatch: pytest.MonkeyPatch, fake: _FakeLibc) -> None:
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(filesystem_module, "_libc_realpath", lambda: (fake.realpath, fake.free))


async def test_posix_branch_is_one_realpath_call_on_the_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeLibc("/resolved/target")
    _route(monkeypatch, fake)
    result = await LocalFileSystem(cwd=str(tmp_path)).canonical_path("sub/x")
    assert result == Ok("/resolved/target")
    assert fake.calls == [os.fsencode(str(tmp_path / "sub" / "x"))]
    assert len(fake.freed) == 1  # the allocated result is released exactly once


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("ELOOP", FsErrorCode.UNKNOWN),
        ("ENOENT", FsErrorCode.NOT_FOUND),
        ("EACCES", FsErrorCode.PERMISSION_DENIED),
        ("ENOTDIR", FsErrorCode.NOT_DIRECTORY),
    ],
)
async def test_posix_branch_classifies_the_kept_errno(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, code: FsErrorCode
) -> None:
    import errno

    fake = _FakeLibc(None, getattr(errno, name))
    _route(monkeypatch, fake)
    result = await LocalFileSystem(cwd=str(tmp_path)).canonical_path("p")
    assert isinstance(result, Err)
    assert result.error.code == code
    assert fake.freed == []


def test_posix_branch_never_passes_a_nul_path_to_libc(monkeypatch: pytest.MonkeyPatch) -> None:
    """`L12D004-R001`: a C string ends at the NUL, so `realpath(3)` would resolve the prefix. A
    NUL-containing path keeps the previous `os.path.realpath(strict=True)`, never reaching libc."""
    fake = _FakeLibc("/the/prefix")
    _route(monkeypatch, fake)
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        filesystem_module.os.path,
        "realpath",
        lambda p, strict=False: calls.append((p, strict)) or "PREVIOUS",
    )
    assert filesystem_module._realpath("/the/prefix\0missing") == "PREVIOUS"
    assert calls == [("/the/prefix\0missing", True)]
    assert fake.calls == []
    assert fake.freed == []


async def _previous_canonical_path(cwd: Path, name: str) -> object:
    """The pre-`L12-D004` body: `os.path.realpath(strict=True)`; an `OSError` becomes `Err`."""
    native = filesystem_module.native_path(filesystem_module.resolve_local_path(str(cwd), name))
    try:
        return Ok(os.path.realpath(native, strict=True))
    except OSError as exc:
        return Err(filesystem_module.to_fs_error(exc, native))


async def _outcome(call: Callable[[], Awaitable[object]]) -> object:
    try:
        result = await call()
    except ValueError as exc:  # the previous POSIX rejection: embedded null character
        return ("raises", type(exc), str(exc))
    if isinstance(result, Err):  # the `cause` exception objects compare by identity
        return ("err", result.error.code, result.error.message, result.error.path)
    return result


@pytest.mark.parametrize("name", ["file\0missing", "missing\0file", "file\0"])
async def test_a_nul_path_keeps_the_previous_outcome_and_never_answers_for_its_prefix(
    tmp_path: Path, name: str
) -> None:
    """`L12D004-R001` real-host witness, with an existing prefix `file`: the outcome is exactly the
    previous resolution's on this host (POSIX: `ValueError`, embedded null character; Windows:
    unchanged), and never `file`'s canonical path. NUL's disposition is `minion-agent#133`'s and is
    not decided here."""
    (tmp_path / "file").write_text("x")
    fs = LocalFileSystem(cwd=str(tmp_path))
    candidate = await _outcome(lambda: fs.canonical_path(name))
    previous = await _outcome(lambda: _previous_canonical_path(tmp_path, name))
    assert candidate == previous
    assert candidate != Ok(os.path.realpath(tmp_path / "file"))


def test_windows_branch_keeps_os_path_realpath(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(
        filesystem_module.os.path,
        "realpath",
        lambda p, strict=False: calls.append((p, strict)) or "R",
    )
    assert filesystem_module._realpath("C:\\x") == "R"
    assert calls == [("C:\\x", True)]


# ---- real-host regressions (Owner: self-referential, multi-node cycle, valid chains) ------------

WINDOWS_CYCLE_CODE = pytest.mark.xfail(
    sys.platform == "win32",
    strict=True,
    reason="pre-existing Windows classification gap (#69): Pi reports `unknown` (ELOOP), Python "
    "`invalid`; out of L12-D004 scope (Windows resolution unchanged)",
)


def _chain(base: Path, links: int) -> Path:
    """`base/0 -> 1 -> ... -> links-1 -> end`: `links` symlinks before a regular file."""
    base.mkdir()
    (base / "end").write_text("x")
    for k in range(links):
        os.symlink(str(k + 1) if k + 1 < links else "end", base / str(k))
    return base / "0"


@WINDOWS_CYCLE_CODE
async def test_a_self_referential_symlink_is_unknown(tmp_path: Path) -> None:
    os.symlink("self", tmp_path / "self")
    result = await LocalFileSystem(cwd=str(tmp_path)).canonical_path("self")
    assert isinstance(result, Err) and result.error.code == FsErrorCode.UNKNOWN


@WINDOWS_CYCLE_CODE
async def test_a_multi_node_cycle_is_unknown(tmp_path: Path) -> None:
    os.symlink("b", tmp_path / "a")
    os.symlink("c", tmp_path / "b")
    os.symlink("a", tmp_path / "c")
    result = await LocalFileSystem(cwd=str(tmp_path)).canonical_path("a")
    assert isinstance(result, Err) and result.error.code == FsErrorCode.UNKNOWN


@pytest.mark.parametrize("links", [1, 5, 39, 40])
async def test_a_valid_symlink_chain_resolves_to_its_target(tmp_path: Path, links: int) -> None:
    start = _chain(tmp_path / "chain", links)
    result = await LocalFileSystem(cwd=str(tmp_path)).canonical_path(str(start))
    assert result == Ok(os.path.realpath(tmp_path / "chain" / "end"))


@pytest.mark.skipif(sys.platform == "win32", reason="the 40-link limit is the POSIX MAXSYMLINKS")
async def test_a_chain_beyond_the_symlink_limit_is_unknown_as_in_pi(tmp_path: Path) -> None:
    start = _chain(tmp_path / "chain", 41)
    result = await LocalFileSystem(cwd=str(tmp_path)).canonical_path(str(start))
    assert isinstance(result, Err) and result.error.code == FsErrorCode.UNKNOWN


@pytest.mark.parametrize(("depth", "fails"), [(40, False), (41, True)])
async def test_a_cycle_walked_deep_fails_at_the_traversal_limit(
    tmp_path: Path, depth: int, fails: bool
) -> None:
    """The WP-14.1 c01 shape: `loop/a/back -> ..`, probed `depth` levels deep."""
    (tmp_path / "loop" / "a").mkdir(parents=True)
    os.symlink("..", tmp_path / "loop" / "a" / "back", target_is_directory=True)
    rel = "/".join(["loop", "a"] + ["back", "a"] * depth)
    result = await LocalFileSystem(cwd=str(tmp_path)).canonical_path(rel)
    if sys.platform == "win32":
        # Windows: unchanged; its reparse-point limit differs from POSIX MAXSYMLINKS and both
        # depths resolve, as with Pi (characterization)
        assert result == Ok(os.path.realpath(tmp_path / "loop" / "a"))
    elif fails:
        assert isinstance(result, Err) and result.error.code == FsErrorCode.UNKNOWN
    else:
        assert result == Ok(os.path.realpath(tmp_path / "loop" / "a"))
