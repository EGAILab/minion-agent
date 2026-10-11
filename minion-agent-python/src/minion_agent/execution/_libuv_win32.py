"""L12-D007 (spec/execution.md section 19.2): the Win32 calls pinned libuv 1.49.2 makes for the
Pi-derived file operations, so a Windows failure carries the SAME Win32 error pinned Pi classifies.

CPython's `open()` goes through the C runtime's `_wopen`, whose errno mapping collapses distinct
Win32 causes (123 and 1921 -> EINVAL, 32/33 -> EACCES) and keeps no `winerror`; and it refuses to
open a directory at all, where libuv's open succeeds and the READ fails. This module reproduces
libuv `src/win/fs.c` (blob f2215bb3082178193d37f8429536bfe7b707dd0d):

- `fs__open`: `CreateFileW` with full sharing (`FILE_SHARE_READ | FILE_SHARE_WRITE |
  FILE_SHARE_DELETE`), `FILE_FLAG_BACKUP_SEMANTICS` (so a directory opens), the access and
  creation disposition libuv derives from Node's `r` / `w` / `a` flags, and its one special case:
  `ERROR_FILE_EXISTS` under create-without-exclusive is `EISDIR`;
- the read of a directory handle fails with `ERROR_INVALID_FUNCTION`, i.e. `EISDIR`, with no path;
- `fs__scandir`: the directory is opened with `FILE_LIST_DIRECTORY | SYNCHRONIZE`, the same
  sharing and backup semantics; a non-directory there is libuv's explicit `ENOTDIR`.

Windows only; every failure is an `OSError` carrying the Win32 code (`winerror`) and the path.
"""

from __future__ import annotations

import ctypes
import errno as _errno
import io
import os
import sys
from ctypes import wintypes
from typing import IO, Any

assert sys.platform == "win32"  # imported only on Windows; also scopes type checking to it

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_k32.CreateFileW.restype = wintypes.HANDLE
_k32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
_k32.GetFileAttributesW.restype = wintypes.DWORD
_k32.CloseHandle.argtypes = [wintypes.HANDLE]


class _ByHandleInfo(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", wintypes.FILETIME),
        ("ftLastAccessTime", wintypes.FILETIME),
        ("ftLastWriteTime", wintypes.FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


_k32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ByHandleInfo)]

_INVALID_HANDLE = wintypes.HANDLE(-1).value
_SHARE_ALL = 0x1 | 0x2 | 0x4  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
_BACKUP_SEMANTICS = 0x02000000
_ATTRIBUTE_NORMAL = 0x80
_ATTRIBUTE_DIRECTORY = 0x10
_ERROR_FILE_EXISTS = 80
# libuv fs__open, from Node's flags: `r` = O_RDONLY; `w` = O_WRONLY|O_CREAT|O_TRUNC;
# `a` = O_WRONLY|O_CREAT|O_APPEND.
_FILE_GENERIC_READ = 0x120089
_FILE_GENERIC_WRITE = 0x120116
_FILE_WRITE_DATA = 0x2
_FILE_APPEND_DATA = 0x4
_MODES = {
    # mode: (access, disposition, creates)
    "r": (_FILE_GENERIC_READ, 3, False),  # OPEN_EXISTING
    "w": (_FILE_GENERIC_WRITE, 2, True),  # CREATE_ALWAYS
    "a": ((_FILE_GENERIC_WRITE & ~_FILE_WRITE_DATA) | _FILE_APPEND_DATA, 4, True),  # OPEN_ALWAYS
}
_FILE_LIST_DIRECTORY = 0x1
_SYNCHRONIZE = 0x100000


_k32.ReadFile.argtypes = [
    wintypes.HANDLE,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPVOID,
]
_k32.WriteFile.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPVOID,
]
_ERROR_BROKEN_PIPE = 109
_ERROR_HANDLE_EOF = 38
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_FLAGS = 1004


def _win32_error(code: int, path: str | None) -> OSError:
    return OSError(0, ctypes.FormatError(code).strip(), path, code)


def _io_error_code(code: int) -> int:
    """libuv `fs__read` / `fs__write` (src/win/fs.c lines 870-873 / 1075-1079): an I/O syscall's
    `ERROR_ACCESS_DENIED` is reported as `ERROR_INVALID_FLAGS` (UV_EBADF, which pinned Pi's
    `toFileError` names `unknown`), L12D007-I004. Local to the read / write call: an OPEN refused
    with 5 keeps its own code."""
    return _ERROR_INVALID_FLAGS if code == _ERROR_ACCESS_DENIED else code


class _HandleIO(io.RawIOBase):
    """The read and write of libuv `fs__read` / `fs__write`: `ReadFile` / `WriteFile` on the
    handle, so a failure (for example 33 `ERROR_LOCK_VIOLATION` under a byte-range lock) keeps its
    Win32 code -- with libuv's own `ERROR_ACCESS_DENIED` override (`_io_error_code`). The C
    runtime's `_read` / `_write` would collapse it into `EACCES`. libuv's read and write errors
    carry no path, so neither does this one."""

    def __init__(self, handle: int, readable: bool) -> None:
        super().__init__()
        self._handle = handle
        self._readable = readable

    def readable(self) -> bool:
        return self._readable

    def writable(self) -> bool:
        return not self._readable

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        size = len(view)
        chunk = (ctypes.c_char * size).from_buffer(view)
        done = wintypes.DWORD(0)
        if not _k32.ReadFile(self._handle, chunk, size, ctypes.byref(done), None):
            code = _io_error_code(ctypes.get_last_error())  # libuv: before its EOF check
            if code in (_ERROR_HANDLE_EOF, _ERROR_BROKEN_PIPE):
                return 0
            raise _win32_error(code, None)
        return int(done.value)

    def write(self, data: Any) -> int:
        view = bytes(data)
        done = wintypes.DWORD(0)
        if not _k32.WriteFile(self._handle, view, len(view), ctypes.byref(done), None):
            raise _win32_error(_io_error_code(ctypes.get_last_error()), None)
        return int(done.value)

    def close(self) -> None:
        if not self.closed:
            _k32.CloseHandle(self._handle)
        super().close()


def _is_directory(handle: int) -> bool:
    info = _ByHandleInfo()
    if not _k32.GetFileInformationByHandle(handle, ctypes.byref(info)):
        return False
    return bool(info.dwFileAttributes & _ATTRIBUTE_DIRECTORY)


def _create(path: str, access: int, disposition: int) -> int:
    return _create_flags(path, access, _BACKUP_SEMANTICS, disposition, _ATTRIBUTE_NORMAL)


def _create_flags(
    path: str, access: int, flags: int, disposition: int = 3, attributes: int = 0
) -> int:
    if "\0" in path:
        # ctypes would silently cut a wide string at its first NUL and open a DIFFERENT path.
        # Reject it exactly as CPython's own path conversion does, so the L12-D006 containment
        # (section 18) classifies it unchanged.
        raise ValueError("embedded null character in path")
    handle = _k32.CreateFileW(path, access, _SHARE_ALL, None, disposition, attributes | flags, None)
    if handle == _INVALID_HANDLE or handle is None:
        raise _win32_error(ctypes.get_last_error(), path)
    return int(handle)


def open_like_libuv(path: str, mode: str, **text: Any) -> IO[Any]:
    """`mode` is Node's flag (`r` / `w` / `a`); `text` (encoding, errors) opens a text stream."""
    access, disposition, creates = _MODES[mode]
    try:
        handle = _create(path, access, disposition)
    except OSError as exc:
        if creates and exc.winerror == _ERROR_FILE_EXISTS:
            # libuv fs__open: ERROR_FILE_EXISTS with O_CREAT and without O_EXCL "means the path
            # referred to a directory" -- UV_EISDIR (it keeps the path).
            raise IsADirectoryError(_errno.EISDIR, os.strerror(_errno.EISDIR), path) from exc
        raise
    if mode == "r" and _is_directory(handle):
        # The open of a directory succeeds; libuv's READ then fails with ERROR_INVALID_FUNCTION,
        # i.e. EISDIR, an error that carries no path (pinned Pi reports its fallback path).
        _k32.CloseHandle(handle)
        raise IsADirectoryError(_errno.EISDIR, os.strerror(_errno.EISDIR))
    raw = _HandleIO(handle, readable=mode == "r")
    buffered: IO[bytes] = io.BufferedReader(raw) if mode == "r" else io.BufferedWriter(raw)
    return io.TextIOWrapper(buffered, **text) if text else buffered


class _BasicInfo(ctypes.Structure):
    _fields_ = [
        ("CreationTime", ctypes.c_longlong),
        ("LastAccessTime", ctypes.c_longlong),
        ("LastWriteTime", ctypes.c_longlong),
        ("ChangeTime", ctypes.c_longlong),
        ("FileAttributes", wintypes.DWORD),
    ]


_k32.SetFileInformationByHandle.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
]
_k32.GetFileInformationByHandleEx.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
]
_FILE_READ_ATTRIBUTES = 0x80
_FILE_WRITE_ATTRIBUTES = 0x100
_DELETE = 0x10000
_OPEN_REPARSE_POINT = 0x00200000
_ATTRIBUTE_READONLY = 0x1
_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_BASIC_INFO = 0
_FILE_DISPOSITION_INFO = 4
_FILE_DISPOSITION_INFO_EX = 21
# FILE_DISPOSITION_DELETE | FILE_DISPOSITION_POSIX_SEMANTICS |
# FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE
_POSIX_DELETE_FLAGS = 0x1 | 0x2 | 0x10
_POSIX_UNSUPPORTED = (
    50,
    87,
    1,
)  # ERROR_NOT_SUPPORTED, ERROR_INVALID_PARAMETER, ERROR_INVALID_FUNCTION


def unlink_like_libuv(path: str, isrmdir: bool) -> None:
    """libuv `fs__unlink_rmdir` (src/win/fs.c lines 1086-1206), the call Node's `unlink` / `rmdir`
    make: open the entry itself (`FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES | DELETE`, full
    sharing, the reparse point itself, backup semantics), then delete THROUGH that handle. So a
    read-denied file is refused by the open (5) exactly as under Pi, and a read-only entry is
    deleted (`IGNORE_READONLY`, or the attribute cleared on the fallback) -- L12-D005's outcome by
    libuv's own mechanism."""
    handle = _create_flags(
        path,
        _FILE_READ_ATTRIBUTES | _FILE_WRITE_ATTRIBUTES | _DELETE,
        _OPEN_REPARSE_POINT | _BACKUP_SEMANTICS,
    )
    try:
        info = _ByHandleInfo()
        if not _k32.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise _win32_error(ctypes.get_last_error(), path)
        attributes = info.dwFileAttributes
        is_dir = bool(attributes & _ATTRIBUTE_DIRECTORY)
        if isrmdir and not is_dir:
            raise _win32_error(267, path)  # SET_REQ_UV_ERROR(UV_ENOENT, ERROR_DIRECTORY)
        if not isrmdir and is_dir and not attributes & _ATTRIBUTE_REPARSE_POINT:
            raise _win32_error(5, path)  # a non-link directory: EPERM, as POSIX.1 mandates
        flags = wintypes.DWORD(_POSIX_DELETE_FLAGS)
        if _k32.SetFileInformationByHandle(
            handle, _FILE_DISPOSITION_INFO_EX, ctypes.byref(flags), ctypes.sizeof(flags)
        ):
            return
        error = ctypes.get_last_error()
        if error not in _POSIX_UNSUPPORTED:
            raise _win32_error(error, path)
        if attributes & _ATTRIBUTE_READONLY:
            basic = _BasicInfo(0, 0, 0, 0, (attributes & ~_ATTRIBUTE_READONLY) | 0x20)  # ARCHIVE
            if not _k32.SetFileInformationByHandle(
                handle, _FILE_BASIC_INFO, ctypes.byref(basic), ctypes.sizeof(basic)
            ):
                raise _win32_error(ctypes.get_last_error(), path)
        delete = ctypes.c_ubyte(1)
        if not _k32.SetFileInformationByHandle(
            handle, _FILE_DISPOSITION_INFO, ctypes.byref(delete), ctypes.sizeof(delete)
        ):
            raise _win32_error(ctypes.get_last_error(), path)
    finally:
        _k32.CloseHandle(handle)


def clear_own_readonly(path: str) -> None:
    """The attribute correction of pinned rimraf's `fixWinEPERM` (`chmod(path, 0o666)`), on the
    entry ITSELF: opened with `FILE_FLAG_OPEN_REPARSE_POINT`, so a link is corrected as itself and
    its target never changes (spec section 17 rule 3; pinned Pi measured to leave a read-only link
    target unchanged, `CE-L12D007-02` rows L9 / U7). A success whether or not the attribute was
    set; a failure (the entry gone, `WRITE_ATTRIBUTES` denied, ...) raises with its Win32 code."""
    handle = _create_flags(
        path,
        _FILE_READ_ATTRIBUTES | _FILE_WRITE_ATTRIBUTES,
        _OPEN_REPARSE_POINT | _BACKUP_SEMANTICS,
    )
    try:
        info = _ByHandleInfo()
        if not _k32.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise _win32_error(ctypes.get_last_error(), path)
        attributes = info.dwFileAttributes
        if not attributes & _ATTRIBUTE_READONLY:
            return
        basic = _BasicInfo(0, 0, 0, 0, (attributes & ~_ATTRIBUTE_READONLY) or _ATTRIBUTE_NORMAL)
        if not _k32.SetFileInformationByHandle(
            handle, _FILE_BASIC_INFO, ctypes.byref(basic), ctypes.sizeof(basic)
        ):
            raise _win32_error(ctypes.get_last_error(), path)
    finally:
        _k32.CloseHandle(handle)


def check_listable(path: str) -> None:
    """libuv `fs__scandir`'s own directory open, before CPython's `os.scandir` (which uses
    `FindFirstFileW` and answers differently, e.g. 267 rather than a sharing violation's 32)."""
    handle = _create(path, _FILE_LIST_DIRECTORY | _SYNCHRONIZE, 3)
    try:
        if not _is_directory(handle):
            raise NotADirectoryError(_errno.ENOTDIR, os.strerror(_errno.ENOTDIR), path)
    finally:
        _k32.CloseHandle(handle)
