"""L12-D007 (spec/execution.md section 19.2): the libuv-equivalent Win32 seam's own branches.

The `_libuv_win32` failure branches a real host cannot be made to produce on demand (a
`GetFileInformationByHandle` failure on an open handle, a volume without POSIX delete semantics)
are driven by replacing the one kernel32 entry point for the duration of a test; the outcome is
still the module's own decision on the Win32 code it receives."""

from __future__ import annotations

import ctypes
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# `_libuv_win32` (Windows only)
# ---------------------------------------------------------------------------

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="libuv's Win32 seam")


def _lib() -> Any:
    from minion_agent.execution import _libuv_win32

    return _libuv_win32


def _failing(code: int) -> Callable[..., int]:
    def call(*args: Any) -> int:
        ctypes.set_last_error(code)
        return 0

    return call


@windows_only
def test_read_failure_keeps_its_win32_code_and_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fs__read`: a `ReadFile` failure other than end-of-file (here 33, a byte-range lock) keeps
    its Win32 code and, as libuv's read error, no path."""
    lib = _lib()
    (tmp_path / "f").write_bytes(b"data")
    stream = lib.open_like_libuv(str(tmp_path / "f"), "r")
    try:
        monkeypatch.setattr(lib._k32, "ReadFile", _failing(33))
        with pytest.raises(OSError) as caught:
            stream.read()
    finally:
        monkeypatch.undo()
        stream.close()
    assert caught.value.winerror == 33
    assert caught.value.filename is None


@windows_only
@pytest.mark.parametrize("code", [38, 109])  # ERROR_HANDLE_EOF, ERROR_BROKEN_PIPE
def test_read_end_of_file_failure_is_end_of_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """`fs__read`: `ERROR_HANDLE_EOF` / `ERROR_BROKEN_PIPE` end the data; they are not errors."""
    lib = _lib()
    (tmp_path / "f").write_bytes(b"data")
    stream = lib.open_like_libuv(str(tmp_path / "f"), "r")
    try:
        monkeypatch.setattr(lib._k32, "ReadFile", _failing(code))
        assert stream.read() == b""
    finally:
        monkeypatch.undo()
        stream.close()


@windows_only
def test_listable_check_without_handle_information_is_not_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = _lib()
    monkeypatch.setattr(lib._k32, "GetFileInformationByHandle", _failing(6))
    with pytest.raises(NotADirectoryError):
        lib.check_listable(str(tmp_path))


@windows_only
def test_unlink_without_handle_information_reports_that_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = _lib()
    target = tmp_path / "f"
    target.write_bytes(b"")
    monkeypatch.setattr(lib._k32, "GetFileInformationByHandle", _failing(6))
    with pytest.raises(OSError) as caught:
        lib.unlink_like_libuv(str(target), isrmdir=False)
    assert (caught.value.winerror, caught.value.filename) == (6, str(target))
    monkeypatch.undo()
    assert target.exists()


@windows_only
def test_rmdir_of_a_file_is_error_directory(tmp_path: Path) -> None:
    """`fs__unlink_rmdir` with `isrmdir` on a non-directory: `ERROR_DIRECTORY` (267)."""
    target = tmp_path / "f"
    target.write_bytes(b"")
    with pytest.raises(OSError) as caught:
        _lib().unlink_like_libuv(str(target), isrmdir=True)
    assert caught.value.winerror == 267
    assert target.exists()


@windows_only
def test_unlink_of_a_directory_is_access_denied(tmp_path: Path) -> None:
    """`fs__unlink_rmdir` without `isrmdir` on a non-link directory: EPERM (Win32 5)."""
    target = tmp_path / "d"
    target.mkdir()
    with pytest.raises(OSError) as caught:
        _lib().unlink_like_libuv(str(target), isrmdir=False)
    assert caught.value.winerror == 5
    assert target.is_dir()


def _without_posix_delete(
    lib: Any, monkeypatch: pytest.MonkeyPatch, fail: dict[int, int] | None = None
) -> list[int]:
    """`SetFileInformationByHandle` refuses `FileDispositionInfoEx` (87, as a volume without POSIX
    delete semantics does); every other class is the real call unless `fail` scripts its code."""
    real = lib._k32.SetFileInformationByHandle
    classes: list[int] = []
    scripted = {lib._FILE_DISPOSITION_INFO_EX: 87, **(fail or {})}

    def call(handle: Any, info_class: int, buffer: Any, size: int) -> int:
        classes.append(info_class)
        if info_class in scripted:
            ctypes.set_last_error(scripted[info_class])
            return 0
        return int(real(handle, info_class, buffer, size))

    monkeypatch.setattr(lib._k32, "SetFileInformationByHandle", call)
    return classes


@windows_only
@pytest.mark.parametrize("readonly", [False, True])
def test_fallback_deletion_clears_readonly_then_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readonly: bool
) -> None:
    """libuv's fallback: a read-only entry has the attribute cleared (`FileBasicInfo`), then the
    entry is deleted with `FileDispositionInfo`."""
    lib = _lib()
    target = tmp_path / "f"
    target.write_bytes(b"")
    if readonly:
        os.chmod(target, stat.S_IREAD)
    classes = _without_posix_delete(lib, monkeypatch)
    lib.unlink_like_libuv(str(target), isrmdir=False)
    assert not target.exists()
    expected = [lib._FILE_DISPOSITION_INFO_EX]
    expected += [lib._FILE_BASIC_INFO] if readonly else []
    expected += [lib._FILE_DISPOSITION_INFO]
    assert classes == expected


@windows_only
@pytest.mark.parametrize(
    ("scripted", "code"),
    [
        pytest.param({21: 32}, 32, id="posix-delete-refused"),
        pytest.param({0: 5}, 5, id="attribute-clear-refused"),
        pytest.param({4: 32}, 32, id="fallback-delete-refused"),
    ],
)
def test_deletion_failures_keep_their_win32_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripted: dict[int, int], code: int
) -> None:
    lib = _lib()
    target = tmp_path / "f"
    target.write_bytes(b"")
    os.chmod(target, stat.S_IREAD)
    _without_posix_delete(lib, monkeypatch, scripted)  # a scripted class 21 overrides the 87
    try:
        with pytest.raises(OSError) as caught:
            lib.unlink_like_libuv(str(target), isrmdir=False)
    finally:
        monkeypatch.undo()
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
    assert (caught.value.winerror, caught.value.filename) == (code, str(target))
    assert target.exists()


# --- `clear_own_readonly`: fixWinEPERM's attribute correction (spec section 19.5 rule 5) ----------


def _readonly(path: Path) -> bool:
    return bool(os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)


@windows_only
@pytest.mark.parametrize("readonly", [True, False])
def test_the_correction_clears_the_attribute_or_succeeds_without_one(
    tmp_path: Path, readonly: bool
) -> None:
    target = tmp_path / "f"
    target.write_bytes(b"")
    if readonly:
        os.chmod(target, stat.S_IREAD)
    _lib().clear_own_readonly(str(target))
    assert not _readonly(target)


@windows_only
def test_the_correction_never_changes_a_link_target(tmp_path: Path) -> None:
    target = tmp_path / "t"
    target.write_bytes(b"")
    os.chmod(target, stat.S_IREAD)
    os.symlink(target, tmp_path / "link")
    try:
        _lib().clear_own_readonly(str(tmp_path / "link"))
        assert _readonly(target)
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)


@windows_only
def test_the_correction_of_a_missing_entry_fails_with_its_win32_code(tmp_path: Path) -> None:
    with pytest.raises(OSError) as caught:
        _lib().clear_own_readonly(str(tmp_path / "missing"))
    assert caught.value.winerror == 2


@windows_only
@pytest.mark.parametrize("stage", ["inspect", "update"])
def test_a_correction_failure_keeps_its_win32_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    lib = _lib()
    target = tmp_path / "f"
    target.write_bytes(b"")
    os.chmod(target, stat.S_IREAD)
    entry = "GetFileInformationByHandle" if stage == "inspect" else "SetFileInformationByHandle"
    monkeypatch.setattr(lib._k32, entry, _failing(5))
    try:
        with pytest.raises(OSError) as caught:
            lib.clear_own_readonly(str(target))
    finally:
        monkeypatch.undo()
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
    assert (caught.value.winerror, caught.value.filename) == (5, str(target))
