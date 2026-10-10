"""Error vocabulary shared across the three execution capability seams (`EXEC-001`,
spec/execution.md section 2).

An execution-seam operation normalizes every EXPECTED operational/environmental failure
into a typed `Result` error (see `execution/result.py`). Framework/provider invariant
violations and programming errors remain ordinary Python exceptions -- an unnormalized
backend exception escaping a seam operation is itself a provider bug, never something this
module's own mapper is expected, or permitted, to catch and convert. Only genuine `OSError`
subclasses reach `to_fs_error`; anything else (an `AssertionError`, a broken invariant, a
`TypeError` from a provider's own bug) propagates unconverted, exactly as spec section 2's
own operational-vs-invariant boundary requires (`L12-R015`).
"""

from __future__ import annotations

import errno as _errno
import sys
from dataclasses import dataclass
from enum import StrEnum


class FsErrorCode(StrEnum):
    """`EXEC-001`/spec section 2.1, `DIRECT_PI_PARITY` (`FileErrorCode`, pinned Pi
    `types.ts:132-140`)."""

    ABORTED = "aborted"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    NOT_DIRECTORY = "not_directory"
    IS_DIRECTORY = "is_directory"
    INVALID = "invalid"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"


class ShellErrorCode(StrEnum):
    """`EXEC-001`/spec section 2.2, `DIRECT_PI_PARITY` (`ExecutionErrorCode`, pinned Pi
    `types.ts:158-164`)."""

    ABORTED = "aborted"
    TIMEOUT = "timeout"
    SHELL_UNAVAILABLE = "shell_unavailable"
    SPAWN_ERROR = "spawn_error"
    CALLBACK_ERROR = "callback_error"
    UNKNOWN = "unknown"


class SubprocessErrorCode(StrEnum):
    """`EXEC-001`/spec section 2.3, `MINION_EXTENSION`. No `timeout` member: `ctx.subprocess`
    has no native timeout concept at all -- removed at `L12-R006` after an earlier revision
    included one no operation in this contract could ever produce."""

    ABORTED = "aborted"
    SPAWN_ERROR = "spawn_error"
    PIPE_ERROR = "pipe_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FsError:
    """A `ctx.fs` operation's typed failure."""

    code: FsErrorCode
    message: str
    path: str | None = None
    cause: BaseException | None = None


@dataclass(frozen=True, slots=True)
class ShellError:
    """A `ctx.shell` operation's typed failure."""

    code: ShellErrorCode
    message: str
    cause: BaseException | None = None


@dataclass(frozen=True, slots=True)
class SubprocessError:
    """A `ctx.subprocess` operation's typed failure."""

    code: SubprocessErrorCode
    message: str
    cause: BaseException | None = None


# L12-D007 (spec/execution.md section 19): pinned libuv 1.49.2 `uv_translate_sys_error`
# (src/win/error.c, blob 7abf906bb5c82312aeb9f3f30f39ab2cadc07eae) reduced through pinned Pi's
# `toFileError`. Only the Win32 codes libuv sends to an errno Pi names are listed; every other code
# is `unknown` (libuv's own default is UV_UNKNOWN, and Pi maps any unnamed errno to `unknown`).
# libuv has no ENOTDIR entry: Windows `not_directory` only comes from explicit call sites.
_WIN32_PI_CODES: dict[int, FsErrorCode] = {
    # -> UV_ENOENT
    2: FsErrorCode.NOT_FOUND,  # ERROR_FILE_NOT_FOUND
    3: FsErrorCode.NOT_FOUND,  # ERROR_PATH_NOT_FOUND
    15: FsErrorCode.NOT_FOUND,  # ERROR_INVALID_DRIVE
    123: FsErrorCode.NOT_FOUND,  # ERROR_INVALID_NAME
    126: FsErrorCode.NOT_FOUND,  # ERROR_MOD_NOT_FOUND
    161: FsErrorCode.NOT_FOUND,  # ERROR_BAD_PATHNAME
    203: FsErrorCode.NOT_FOUND,  # ERROR_ENVVAR_NOT_FOUND
    267: FsErrorCode.NOT_FOUND,  # ERROR_DIRECTORY
    4392: FsErrorCode.NOT_FOUND,  # ERROR_INVALID_REPARSE_DATA
    11001: FsErrorCode.NOT_FOUND,  # WSAHOST_NOT_FOUND
    11004: FsErrorCode.NOT_FOUND,  # WSANO_DATA
    # -> UV_EACCES / UV_EPERM
    5: FsErrorCode.PERMISSION_DENIED,  # ERROR_ACCESS_DENIED (EPERM)
    740: FsErrorCode.PERMISSION_DENIED,  # ERROR_ELEVATION_REQUIRED
    1314: FsErrorCode.PERMISSION_DENIED,  # ERROR_PRIVILEGE_NOT_HELD (EPERM)
    1920: FsErrorCode.PERMISSION_DENIED,  # ERROR_CANT_ACCESS_FILE
    10013: FsErrorCode.PERMISSION_DENIED,  # WSAEACCES
    # -> UV_EISDIR
    1: FsErrorCode.IS_DIRECTORY,  # ERROR_INVALID_FUNCTION
    # -> UV_EINVAL
    13: FsErrorCode.INVALID,  # ERROR_INVALID_DATA
    87: FsErrorCode.INVALID,  # ERROR_INVALID_PARAMETER
    122: FsErrorCode.INVALID,  # ERROR_INSUFFICIENT_BUFFER
    1464: FsErrorCode.INVALID,  # ERROR_SYMLINK_NOT_SUPPORTED
    10022: FsErrorCode.INVALID,  # WSAEINVAL
    10046: FsErrorCode.INVALID,  # WSAEPFNOSUPPORT
}


def to_pi_fs_error(exc: OSError, path: str | None = None) -> FsError:
    """L12-D007: classify a failure of one of the Pi-derived filesystem operations exactly as
    pinned Pi does. On Windows the ORIGINAL Win32 error (`winerror`) goes through the pinned libuv
    translation (`_WIN32_PI_CODES`), never CPython's already-collapsed errno. Elsewhere, and for a
    Windows error that carries no Win32 code, this is `to_fs_error`. Section 19.2: the
    EXEC-007/008/009 operations classify their failures through this mapper too; their own
    dispositions (which call is made, what a success means) are unchanged."""
    winerror = getattr(exc, "winerror", None)
    if sys.platform == "win32" and isinstance(winerror, int):
        return FsError(_WIN32_PI_CODES.get(winerror, FsErrorCode.UNKNOWN), str(exc), path, exc)
    return to_fs_error(exc, path)


def to_fs_error(exc: OSError, path: str | None = None) -> FsError:
    """Map a caught `OSError` to `FsError`, matching pinned Pi's own `toFileError`
    (`nodejs.ts:97-121`) exactly: `FileNotFoundError`->`not_found`,
    `PermissionError`->`permission_denied`, `NotADirectoryError`->`not_directory`,
    `IsADirectoryError`->`is_directory`, `errno.EINVAL`->`invalid`, else `unknown`.

    Checked BEFORE `errno.EINVAL` because Python's built-in OSError subclasses are each
    keyed to a specific errno already (`ENOENT`, `EACCES`/`EPERM`, `ENOTDIR`, `EISDIR`) --
    matching pinned Pi's own `switch (nodeError.code)` order, where each case returns before
    the next is even considered.
    """
    if isinstance(exc, FileNotFoundError):
        return FsError(FsErrorCode.NOT_FOUND, str(exc), path, exc)
    if isinstance(exc, PermissionError):
        return FsError(FsErrorCode.PERMISSION_DENIED, str(exc), path, exc)
    if isinstance(exc, NotADirectoryError):
        return FsError(FsErrorCode.NOT_DIRECTORY, str(exc), path, exc)
    if isinstance(exc, IsADirectoryError):
        return FsError(FsErrorCode.IS_DIRECTORY, str(exc), path, exc)
    if exc.errno == _errno.EINVAL:
        return FsError(FsErrorCode.INVALID, str(exc), path, exc)
    return FsError(FsErrorCode.UNKNOWN, str(exc), path, exc)
