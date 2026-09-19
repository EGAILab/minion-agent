"""`ctx.fs` (`EXEC-002`) and the `FsTarget` bridge (`EXEC-003`), spec/execution.md sections 3-4.

Cancellation is cooperative/poll-based throughout (`runtime.signal.RunSignal`, matching
pinned Pi's own `AbortSignal`) -- every operation ACCEPTS an optional `signal` in its typed
signature (the uniform public-API-shape rule, `L12-R001`), but only some operations actually
INSPECT it, exactly per the binding per-operation table in spec section 3.1: `read_text_file`/
`read_binary_file`/`write_file` check pre-aborted plus one underlying-call checkpoint (`write_file`
gets a THIRD checkpoint, immediately after its own parent-mkdir, before the write begins);
`read_text_lines` checks pre-aborted, at each loop iteration, AND once more after the loop
completes (four checkpoints); `list_dir` checks pre-aborted (including for an EMPTY directory,
independent of loop-entry count -- `L12-PY-R001`) and at each loop iteration only, with
deliberately NO post-loop check (fewer than `read_text_lines`, not a bug); `rename_file` checks
pre-aborted only; every other operation accepts `signal` but never reads it at all, matching
pinned Pi's own reference implementation exactly, not a gap.

Prompt mid-operation settlement (`L12-PY-R001`): Python's blocking `open().read()`/`.write()` run
as one `asyncio.to_thread` call with no native mid-syscall interruption point the way Node's
`fs.readFile({signal})` has. Python cannot forcibly kill a blocking OS thread -- but the
CALLER-OBSERVABLE result can and must still settle promptly when the signal fires while the
underlying call is still blocked (e.g. reading a slow/blocked source). `_race_signal` below races
the `to_thread` call against the signal; if the signal wins, `read_text_file`/`read_binary_file`/
`write_file`'s own final phase settles `Err(aborted)` immediately, abandoning the still-running
thread in the background (its eventual result/exception is discarded, not left to raise an
unretrieved-exception warning).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import stat as _stat
import tempfile
import uuid
from collections.abc import Coroutine, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.request import url2pathname

from ..runtime.signal import RunSignal
from .errors import FsError, FsErrorCode, to_fs_error
from .result import Err, Ok, Result
from .world import ExecutionWorldIdentity

_SIGNAL_POLL_INTERVAL_S = 0.01


class FileKind(StrEnum):
    """`EXEC-002`, `DIRECT_PI_PARITY` (`FileKind`, pinned Pi `types.ts:129`)."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


@dataclass(frozen=True, slots=True)
class FileInfo:
    """One addressed path's metadata, never following symlinks (spec section 3.2)."""

    name: str
    path: str
    kind: FileKind
    size: int
    mtime_ms: float


class _AbortedSignal(Exception):
    """Internal control-flow signal raised inside a worker thread to report a mid-loop
    cancellation checkpoint firing; caught by the owning async method and converted to
    `Err(FsError(aborted))`. Never escapes this module."""


class _UnsupportedFileType(Exception):
    """Internal control-flow signal for an addressed path whose kind is none of
    file/directory/symlink (a socket, FIFO, device, ...), matching pinned Pi's own
    `fileInfoFromStats` returning an `invalid` `Result` rather than throwing."""


def resolve_local_path(cwd: str, path: str) -> str:
    """Lexical path resolution shared by every local execution-seam provider (`ctx.fs`,
    `ctx.shell`, `ctx.subprocess`) -- matches pinned Pi's own `resolvePath` (`nodejs.ts:51-65`)
    exactly: bare `~` and `~/`-prefixed paths expand against the home directory; a `file://` URL
    is parsed to a filesystem path, with a malformed one kept as the literal string rather than
    raising; an absolute result is lexically normalized as-is; a relative result is resolved
    against `cwd`. Touches no filesystem object. `ctx.shell`/`ctx.subprocess` call this SAME
    function for their own cwd resolution (`EXEC-004`/`EXEC-005`) rather than reimplementing it --
    duplicating this logic with a bare path join was a genuine Rust-side defect (`L12-R012`) this
    Python implementation does not repeat.
    """
    normalized = path
    if normalized == "~":
        normalized = os.path.expanduser("~")
    elif normalized.startswith("~/") or (os.name == "nt" and normalized.startswith("~\\")):
        normalized = os.path.join(os.path.expanduser("~"), normalized[2:])
    elif normalized.startswith("file://"):
        # `L12-PY-R002` (refined, second review): pinned Node's own `resolvePath` does NOT
        # early-return the raw URL on a `fileURLToPath` failure -- it leaves `normalized`
        # UNCHANGED (still the literal "file://..." string) and falls through to the SAME
        # final isabs/resolve pipeline every other path goes through. An earlier revision of
        # this function early-returned the literal string, bypassing that pipeline entirely --
        # observably different from Pi (which then applies ordinary cwd-relative resolution TO
        # the literal string, producing e.g. `<cwd>\file:\%ZZ` on Windows, not the bare
        # `"file:///%ZZ"` string an early return would keep). Suppressing here (no reassignment
        # on failure) reproduces that exact fallthrough.
        with suppress(ValueError, OSError):
            normalized = _file_url_to_path(normalized)
    if os.path.isabs(normalized):
        return os.path.normpath(normalized)
    return os.path.normpath(os.path.join(cwd, normalized))


_WINDOWS_DRIVE_PATH_RE = re.compile(r"^/[A-Za-z]:")


def _file_url_to_path(url: str) -> str:
    """An equivalent-fidelity port of pinned Node's `fileURLToPath`, sufficient for this
    function's own two observable validation requirements (`L12-PY-R002`, second review):
    a non-empty, non-"localhost", non-hostname-shaped host is malformed on EVERY platform
    (Node's own WHATWG URL host parsing rejects a raw `%` in a host outright, e.g.
    `file://%zz-not-a-valid-escape` -- there is no such thing as a literal `%` in a real
    hostname); on Windows specifically, a host-less URL's own path must additionally look like
    a drive path (`/C:/...`) -- `file:///%ZZ` has no drive letter and is ALSO malformed, not a
    valid (if oddly-encoded) local path. Raises `ValueError` for either case, mirroring
    `fileURLToPath`'s own thrown `ERR_INVALID_FILE_URL_HOST`/`ERR_INVALID_FILE_URL_PATH`."""
    parsed = urlparse(url)
    if parsed.netloc and "%" in parsed.netloc:
        # No real hostname ever contains a raw "%" -- a percent-encoded (or malformed-percent)
        # host is invalid on every platform, never a legitimate UNC host on Windows either.
        raise ValueError(f"file:// URL host is not a valid hostname: {parsed.netloc!r}")
    if parsed.netloc and parsed.netloc != "localhost" and os.name != "nt":
        raise ValueError(f"file:// URL host is not local: {parsed.netloc!r}")
    if os.name == "nt" and not parsed.netloc and not _WINDOWS_DRIVE_PATH_RE.match(parsed.path):
        # A host-less Windows file URL's path must look like a drive path ("/C:/...") --
        # anything else (no drive letter at all) is malformed, not a valid local path.
        raise ValueError(f"file:// URL path has no drive letter on Windows: {parsed.path!r}")
    return url2pathname(parsed.path)


def _aborted(path: str | None = None) -> FsError:
    return FsError(FsErrorCode.ABORTED, "aborted", path)


async def _race_signal(op: Coroutine[Any, Any, Any], signal: RunSignal | None) -> Any:
    """`L12-PY-R001`. Races `op` (an `asyncio.to_thread(...)` coroutine) against `signal` firing.
    Returns `op`'s own result/re-raises its own exception if it finishes first. If `signal` fires
    first, raises `_AbortedSignal` promptly -- `op`'s own underlying thread, if still running, is
    abandoned in the background (Python cannot forcibly interrupt a blocking syscall); its
    eventual result or exception is discarded via a suppressing done-callback, never left to raise
    an "exception was never retrieved" warning."""
    if signal is None:
        return await op
    op_task: asyncio.Task[Any] = asyncio.ensure_future(op)

    async def _poll() -> None:
        while not op_task.done():
            if signal.aborted:
                return
            await asyncio.sleep(_SIGNAL_POLL_INTERVAL_S)

    poll_task = asyncio.ensure_future(_poll())
    await asyncio.wait({op_task, poll_task}, return_when=asyncio.FIRST_COMPLETED)
    if op_task.done():
        poll_task.cancel()
        with suppress(asyncio.CancelledError):
            await poll_task
        return op_task.result()
    # The signal won the race -- the to_thread call is still running; abandon it without
    # awaiting, but keep it from logging an unretrieved-exception warning once it does finish.
    op_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    raise _AbortedSignal


def _file_kind_from_stat(st: os.stat_result) -> FileKind | None:
    if _stat.S_ISREG(st.st_mode):
        return FileKind.FILE
    if _stat.S_ISDIR(st.st_mode):
        return FileKind.DIRECTORY
    if _stat.S_ISLNK(st.st_mode):
        return FileKind.SYMLINK
    return None


def _file_info_sync(path: str) -> FileInfo:
    st = os.lstat(path)
    kind = _file_kind_from_stat(st)
    if kind is None:
        raise _UnsupportedFileType
    return FileInfo(
        name=os.path.basename(path),
        path=path,
        kind=kind,
        size=st.st_size,
        mtime_ms=st.st_mtime * 1000,
    )


def _read_text_sync(path: str) -> str:
    # `L12-PY-R002`: `errors="replace"` matches pinned Node's own UTF-8 decoding, which never
    # throws for invalid bytes -- it substitutes the replacement character (U+FFFD). The default
    # `errors="strict"` raises `UnicodeDecodeError`, which is NOT an `OSError` subclass and would
    # therefore escape this operation's own `except OSError` entirely, violating the never-raise
    # `Result` contract outright.
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_binary_sync(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _read_text_lines_sync(path: str, max_lines: int | None, signal: RunSignal | None) -> list[str]:
    if signal is not None and signal.aborted:
        raise _AbortedSignal
    if max_lines is not None and max_lines <= 0:
        return []
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            if signal is not None and signal.aborted:
                raise _AbortedSignal
            lines.append(raw_line.rstrip("\n"))
            if max_lines is not None and len(lines) >= max_lines:
                break
    if signal is not None and signal.aborted:
        raise _AbortedSignal
    return lines


def _write_file_sync(path: str, content: str | bytes) -> None:
    if isinstance(content, str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(path, "wb") as f:
            f.write(content)


def _append_file_sync(path: str, content: str | bytes) -> None:
    if isinstance(content, str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(path, "ab") as f:
            f.write(content)


def _list_dir_sync(path: str, signal: RunSignal | None) -> list[FileInfo]:
    if signal is not None and signal.aborted:
        # `L12-PY-R001`: a check living only INSIDE the loop below never runs at all for an
        # empty directory (zero iterations), so a pre-aborted signal on an empty dir wrongly
        # returned Ok([]). This is the genuine pre-check, independent of entry count.
        raise _AbortedSignal
    infos: list[FileInfo] = []
    with os.scandir(path) as entries:
        for entry in entries:
            if signal is not None and signal.aborted:
                raise _AbortedSignal
            try:
                infos.append(_file_info_sync(entry.path))
            except _UnsupportedFileType:
                # Matches pinned Pi's own listDir exactly: an unsupported entry kind is
                # silently skipped (fileInfoFromStats returns an error Result Pi discards,
                # `info.ok ? push : skip`), not propagated as the whole call's own failure.
                continue
    return infos


def _remove_sync(path: str, recursive: bool, force: bool) -> None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        if force:
            return
        raise
    if _stat.S_ISLNK(st.st_mode):
        # Never follows: removes the addressed symlink's own directory entry regardless of
        # what it points to (spec section 3.2) -- matches Node's raw rm(), which unlinks a
        # symlink rather than recursing into its target.
        os.remove(path)
        return
    if _stat.S_ISDIR(st.st_mode):
        if not recursive:
            # Matches pinned Pi's own fs.rm exactly: ANY directory (even an empty one)
            # requires recursive=true, unlike POSIX rmdir's own more lenient default.
            raise IsADirectoryError(f"Path is a directory: {path}")
        shutil.rmtree(path)
        return
    os.remove(path)


class FileSystem(Protocol):
    """The filesystem capability seam -- an independently-swappable abstraction (`L12-PY-R003`).
    Every operation accepts an optional `signal` in its typed signature uniformly; whether it is
    inspected is per-operation (module docstring). Registered on the Runtime under
    `__service_name__` so `Context.require(FileSystem)`/`ctx.fs` resolves whichever conforming
    provider is mounted."""

    __service_name__: str = "fs"

    cwd: str
    execution_world: ExecutionWorldIdentity

    async def absolute_path(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[str, FsError]: ...
    async def join_path(
        self, parts: Sequence[str], signal: RunSignal | None = None
    ) -> Result[str, FsError]: ...
    async def read_text_file(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[str, FsError]: ...
    async def read_text_lines(
        self, path: str, max_lines: int | None = None, signal: RunSignal | None = None
    ) -> Result[list[str], FsError]: ...
    async def read_binary_file(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[bytes, FsError]: ...
    async def write_file(
        self, path: str, content: str | bytes, signal: RunSignal | None = None
    ) -> Result[None, FsError]: ...
    async def append_file(
        self, path: str, content: str | bytes, signal: RunSignal | None = None
    ) -> Result[None, FsError]: ...
    async def rename_file(
        self, source: str, destination: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]: ...
    async def file_info(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[FileInfo, FsError]: ...
    async def list_dir(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[list[FileInfo], FsError]: ...
    async def canonical_path(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[str, FsError]: ...
    async def exists(self, path: str, signal: RunSignal | None = None) -> Result[bool, FsError]: ...
    async def create_dir(
        self, path: str, recursive: bool = True, signal: RunSignal | None = None
    ) -> Result[None, FsError]: ...
    async def remove(
        self,
        path: str,
        recursive: bool = False,
        force: bool = False,
        signal: RunSignal | None = None,
    ) -> Result[None, FsError]: ...
    async def create_temp_dir(
        self, prefix: str = "tmp-", signal: RunSignal | None = None
    ) -> Result[str, FsError]: ...
    async def create_temp_file(
        self, prefix: str = "", suffix: str = "", signal: RunSignal | None = None
    ) -> Result[str, FsError]: ...
    async def cleanup(self) -> None: ...
    async def resolve(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[FsTarget, FsError]: ...
    async def process_path(
        self, target: FsTarget, signal: RunSignal | None = None
    ) -> Result[str, FsError]: ...


@dataclass(frozen=True, slots=True)
class FsTarget:
    """`EXEC-003`. `target_key` is opaque to callers -- do not parse it, compare its syntax to a
    path, or infer backend identity from its shape. `_provider` is INTERNAL, compared only by
    identity (`is`), never serialized or exposed -- it is how `process_path` detects a target
    produced by a different provider instance (spec section 4's provider-scoping rule)."""

    target_key: str
    _provider: object


class LocalFileSystem:
    """The local filesystem provider (`EXEC-002`/`EXEC-003`/spec section 8) -- `DIRECT_PI_PARITY`
    for `ctx.fs`'s own observable behavior, `MINION_ARCHITECTURAL_MAPPING` for the `FsTarget`
    bridge it also implements."""

    __service_name__: str = "fs"

    __slots__ = ("cwd", "execution_world")

    def __init__(
        self, cwd: str | None = None, execution_world: ExecutionWorldIdentity | None = None
    ) -> None:
        self.cwd = cwd if cwd is not None else os.getcwd()
        self.execution_world = (
            execution_world if execution_world is not None else ExecutionWorldIdentity.local()
        )

    async def absolute_path(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[str, FsError]:
        return Ok(resolve_local_path(self.cwd, path))

    async def join_path(
        self, parts: Sequence[str], signal: RunSignal | None = None
    ) -> Result[str, FsError]:
        # `L12-PY-R002`: matches Node's `path.join()` exactly, not `os.path.join`, which differs
        # in two ways Node does not: (1) `os.path.join()` requires at least one argument and has
        # no zero-arg return; Node's `path.join()` called with zero paths returns "." -- handled
        # by the `if parts else "."` branch. (2) `os.path.join` treats a LATER path-separator-
        # prefixed component as resetting the accumulated path (`os.path.join("a", "\\b") ==
        # "\\b")`; Node just concatenates every segment with the separator, then normalizes
        # (`path.join("a", "\\b") == "a\\b"`) -- reproduced here by joining with `os.sep`
        # directly rather than delegating to `os.path.join`'s own differing semantics.
        if not parts:
            return Ok(".")
        return Ok(os.path.normpath(os.sep.join(p for p in parts if p)))

    async def read_text_file(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[str, FsError]:
        if signal is not None and signal.aborted:
            return Err(_aborted(path))
        resolved = resolve_local_path(self.cwd, path)
        try:
            content = await _race_signal(asyncio.to_thread(_read_text_sync, resolved), signal)
        except _AbortedSignal:
            return Err(_aborted(resolved))
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(content)

    async def read_text_lines(
        self, path: str, max_lines: int | None = None, signal: RunSignal | None = None
    ) -> Result[list[str], FsError]:
        resolved = resolve_local_path(self.cwd, path)
        try:
            lines = await asyncio.to_thread(_read_text_lines_sync, resolved, max_lines, signal)
        except _AbortedSignal:
            return Err(_aborted(resolved))
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(lines)

    async def read_binary_file(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[bytes, FsError]:
        if signal is not None and signal.aborted:
            return Err(_aborted(path))
        resolved = resolve_local_path(self.cwd, path)
        try:
            content = await _race_signal(asyncio.to_thread(_read_binary_sync, resolved), signal)
        except _AbortedSignal:
            return Err(_aborted(resolved))
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(content)

    async def write_file(
        self, path: str, content: str | bytes, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        if signal is not None and signal.aborted:
            return Err(_aborted(path))
        resolved = resolve_local_path(self.cwd, path)
        parent = os.path.dirname(resolved)
        try:
            if parent:
                await asyncio.to_thread(os.makedirs, parent, exist_ok=True)
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        if signal is not None and signal.aborted:
            return Err(_aborted(resolved))
        try:
            await _race_signal(asyncio.to_thread(_write_file_sync, resolved, content), signal)
        except _AbortedSignal:
            return Err(_aborted(resolved))
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(None)

    async def append_file(
        self, path: str, content: str | bytes, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        parent = os.path.dirname(resolved)
        try:
            if parent:
                await asyncio.to_thread(os.makedirs, parent, exist_ok=True)
            await asyncio.to_thread(_append_file_sync, resolved, content)
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(None)

    async def rename_file(
        self, source: str, destination: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        dst = resolve_local_path(self.cwd, destination)
        if signal is not None and signal.aborted:
            return Err(_aborted(dst))
        src = resolve_local_path(self.cwd, source)
        try:
            # os.replace, not os.rename: spec section 3.5 requires an existing destination to
            # be REPLACED atomically. POSIX rename() already does this; os.rename on Windows
            # does NOT (it raises FileExistsError instead) -- os.replace uses MOVEFILE_REPLACE_
            # EXISTING there, giving the SAME cross-platform observable behavior pinned Pi's own
            # rename() (which wraps the OS primitive) provides. Still never follows a symlink at
            # either endpoint -- same underlying non-dereferencing primitive as os.rename.
            await asyncio.to_thread(os.replace, src, dst)
        except OSError as exc:
            return Err(to_fs_error(exc, src))
        return Ok(None)

    async def file_info(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[FileInfo, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        try:
            info = await asyncio.to_thread(_file_info_sync, resolved)
        except _UnsupportedFileType:
            return Err(FsError(FsErrorCode.INVALID, "Unsupported file type", resolved))
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(info)

    async def list_dir(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[list[FileInfo], FsError]:
        resolved = resolve_local_path(self.cwd, path)
        try:
            infos = await asyncio.to_thread(_list_dir_sync, resolved, signal)
        except _AbortedSignal:
            return Err(_aborted(resolved))
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(infos)

    async def canonical_path(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[str, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        try:
            real = await asyncio.to_thread(os.path.realpath, resolved, strict=True)
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(real)

    async def exists(self, path: str, signal: RunSignal | None = None) -> Result[bool, FsError]:
        info = await self.file_info(path)
        if isinstance(info, Ok):
            return Ok(True)
        if info.error.code == FsErrorCode.NOT_FOUND:
            return Ok(False)
        return info

    async def create_dir(
        self, path: str, recursive: bool = True, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        try:
            if recursive:
                await asyncio.to_thread(os.makedirs, resolved, exist_ok=True)
            else:
                await asyncio.to_thread(os.mkdir, resolved)
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(None)

    async def remove(
        self,
        path: str,
        recursive: bool = False,
        force: bool = False,
        signal: RunSignal | None = None,
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        try:
            await asyncio.to_thread(_remove_sync, resolved, recursive, force)
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(None)

    async def create_temp_dir(
        self, prefix: str = "tmp-", signal: RunSignal | None = None
    ) -> Result[str, FsError]:
        try:
            path = await asyncio.to_thread(tempfile.mkdtemp, prefix=prefix)
        except OSError as exc:
            return Err(to_fs_error(exc))
        return Ok(path)

    async def create_temp_file(
        self, prefix: str = "", suffix: str = "", signal: RunSignal | None = None
    ) -> Result[str, FsError]:
        dir_result = await self.create_temp_dir("tmp-")
        if isinstance(dir_result, Err):
            return dir_result
        file_path = os.path.join(dir_result.value, f"{prefix}{uuid.uuid4().hex}{suffix}")
        try:
            await asyncio.to_thread(_write_file_sync, file_path, "")
        except OSError as exc:
            return Err(to_fs_error(exc, file_path))
        return Ok(file_path)

    async def cleanup(self) -> None:
        """`EXEC-002`. No child-process claim (that is `ctx.shell`'s own concern, `L12-R017`); no
        temp-resource-removal claim (spec section 3.7). A true no-op is conforming."""
        return None

    async def resolve(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[FsTarget, FsError]:
        """`EXEC-003`. `target_key = canonical_path(path)` if it succeeds; falls back to
        `absolute_path(path)` ONLY when canonicalization fails with `not_found`/`not_supported`
        (`L12-R016`) -- any other canonicalization failure propagates as this call's own error.
        `resolve()` itself never inspects `signal`, a direct consequence of composing two
        operations that don't either."""
        canonical = await self.canonical_path(path)
        if isinstance(canonical, Ok):
            return Ok(FsTarget(target_key=canonical.value, _provider=self))
        if canonical.error.code in (FsErrorCode.NOT_FOUND, FsErrorCode.NOT_SUPPORTED):
            absolute = await self.absolute_path(path)
            if isinstance(absolute, Err):
                return absolute
            return Ok(FsTarget(target_key=absolute.value, _provider=self))
        return canonical

    async def process_path(
        self, target: FsTarget, signal: RunSignal | None = None
    ) -> Result[str, FsError]:
        """`EXEC-003`. Scoped to the producing provider only (`L12-R014`): returns the exact
        string `target_key` was derived from -- canonical once the resource exists, lexical-
        absolute for a not-yet-existing target (`L12-R021`), never a separately-computed path.
        A best-effort diagnostic (not a MUST): detects a target this instance did not produce and
        returns `invalid` rather than fabricating a path."""
        if target._provider is not self:
            return Err(FsError(FsErrorCode.INVALID, "FsTarget was not produced by this provider"))
        return Ok(target.target_key)
