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
