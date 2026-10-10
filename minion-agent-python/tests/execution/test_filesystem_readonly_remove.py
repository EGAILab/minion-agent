"""`L12-D005` (`spec/execution.md` section 17) binding witnesses for the concurrency clause, which
the canonical corpus cannot express: an entry that disappears while a removal is in progress counts
as removed, as pinned Node treats `ENOENT` during `rimraf` and during its attribute recovery."""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

import minion_agent.execution.filesystem as filesystem_module
from minion_agent.execution import LocalFileSystem, Ok

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows read-only attribute")


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _libuv_fails(monkeypatch: pytest.MonkeyPatch, target: Path, plan: list[Any]) -> list[str]:
    """L12-D007: Windows removal is libuv's handle-based unlink (`_libuv_unlink`), which deletes a
    read-only entry directly. Each call on `target` takes the next planned outcome (an exception
    to raise, or None to run the real unlink), so rimraf's fixWinEPERM layer above it is exercised.
    Returns the list of attempted paths."""
    real = filesystem_module._libuv_unlink
    attempts: list[str] = []

    def unlink(path: str) -> None:
        if path == str(target):
            attempts.append(path)
            outcome = plan.pop(0) if plan else None
            if outcome is not None:
                raise outcome
        real(path)

    monkeypatch.setattr(filesystem_module, "_libuv_unlink", unlink)
    return attempts


def _eperm(path: Path) -> OSError:
    return OSError(0, "Access is denied", str(path), 5)  # libuv's Win32 error: EPERM


@windows_only
async def test_an_entry_vanishing_before_its_attribute_is_cleared_counts_as_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "f"
    target.write_text("x")
    _readonly(target)
    real = filesystem_module._clear_own_readonly_windows
    calls: list[str] = []

    def vanish_first(path: str) -> bool:
        calls.append(path)
        os.chmod(path, stat.S_IWRITE)
        os.remove(path)
        return real(path)

    monkeypatch.setattr(filesystem_module, "_clear_own_readonly_windows", vanish_first)
    _libuv_fails(monkeypatch, target, [_eperm(target)])

    assert await LocalFileSystem(str(tmp_path)).remove("f") == Ok(None)
    assert calls == [str(target)]
    assert not target.exists()


@windows_only
async def test_a_failed_attribute_correction_reports_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "f"
    target.write_text("x")
    _readonly(target)
    monkeypatch.setattr(filesystem_module, "_clear_own_readonly_windows", lambda path: False)
    _libuv_fails(monkeypatch, target, [_eperm(target)])

    result = await LocalFileSystem(str(tmp_path)).remove("f")

    assert result.error.code.value == "permission_denied"  # type: ignore[union-attr]
    assert result.error.path == str(target)  # type: ignore[union-attr]
    os.chmod(target, stat.S_IWRITE)


async def test_an_entry_vanishing_during_the_recursive_walk_counts_as_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a", "b", "c"):
        (tmp_path / "t" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "t" / name).write_text("x")
        if sys.platform == "win32":
            _readonly(tmp_path / "t" / name)
    victim = tmp_path / "t" / "b"
    real_scandir = os.scandir
    fired: list[bool] = []

    def scandir_then_vanish(path: Any = None) -> Any:
        listing = real_scandir(path) if path is not None else real_scandir()
        if fired or Path(path if path is not None else ".").name != "t":
            return listing
        entries = list(listing)
        listing.close()
        os.chmod(victim, stat.S_IWRITE)
        os.remove(victim)
        fired.append(True)

        class _Listing:
            def __init__(self) -> None:
                self._inner = iter(entries)

            def __iter__(self) -> _Listing:
                return self

            def __next__(self) -> Any:
                return next(self._inner)

            def __enter__(self) -> _Listing:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def close(self) -> None:
                return None

        return _Listing()

    monkeypatch.setattr(os, "scandir", scandir_then_vanish)

    assert await LocalFileSystem(str(tmp_path)).remove("t", recursive=True) == Ok(None)
    assert fired == [True]
    assert not (tmp_path / "t").exists()


@windows_only
async def test_a_tree_entry_whose_retry_still_fails_reports_the_retry_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attribute is cleared but the retried deletion still fails: that retry's error is the
    result, naming the entry (pinned Node: `fixWinEPERM` returns the retried `unlink`'s error)."""
    target = tmp_path / "t" / "f"
    target.parent.mkdir()
    target.write_text("x")
    _readonly(target)
    retried: list[str] = []
    monkeypatch.setattr(
        filesystem_module, "_clear_readonly", lambda path: retried.append(path) or True
    )
    _libuv_fails(monkeypatch, target, [_eperm(target), _eperm(target)])

    result = await LocalFileSystem(str(tmp_path)).remove("t", recursive=True)

    assert retried == [str(target)]
    assert result.error.code.value == "permission_denied"  # type: ignore[union-attr]
    assert result.error.path == str(target)  # type: ignore[union-attr]
    os.chmod(target, stat.S_IWRITE)


@windows_only
async def test_a_tree_entry_vanishing_before_its_retry_counts_as_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "t" / "f"
    target.parent.mkdir()
    target.write_text("x")
    _readonly(target)

    def cleared_then_vanished(path: str) -> bool:
        os.chmod(path, stat.S_IWRITE)
        os.remove(path)
        return True

    monkeypatch.setattr(filesystem_module, "_clear_readonly", cleared_then_vanished)
    _libuv_fails(monkeypatch, target, [_eperm(target)])

    assert await LocalFileSystem(str(tmp_path)).remove("t", recursive=True) == Ok(None)
    assert not (tmp_path / "t").exists()


@windows_only
async def test_a_tree_entry_retry_failure_reports_the_retry_error_not_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`L12D005-I001`: the first `unlink` fails on the read-only attribute (`permission_denied`);
    the attribute is really cleared, and the retried `unlink` of the same entry fails DIFFERENTLY
    (`not_directory`). The result is the RETRY's error, mapped, naming the entry (section 14.8),
    as pinned Node's `fixWinEPERM` returns the retried call's error, never the first one."""
    target = tmp_path / "t" / "f"
    target.parent.mkdir()
    target.write_text("x")
    _readonly(target)
    attempts = _libuv_fails(
        monkeypatch,
        target,
        [
            _eperm(target),
            NotADirectoryError(errno.ENOTDIR, "retry failed differently", str(target)),
        ],
    )

    result = await LocalFileSystem(str(tmp_path)).remove("t", recursive=True)

    assert attempts == [str(target), str(target)]
    assert result.error.code.value == "not_directory"  # type: ignore[union-attr]
    assert result.error.path == str(target)  # type: ignore[union-attr]
    assert target.exists()
    assert not os.stat(target).st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
