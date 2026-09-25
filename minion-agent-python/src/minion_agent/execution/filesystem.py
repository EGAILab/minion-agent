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
deliberately NO post-loop check (fewer than `read_text_lines`, not a bug); `rename_file` and
`list_dir_raw` (`EXEC-007`, spec section 11.3, additive Layer-12 extension) each check
pre-aborted only, no per-entry checkpoint; every other operation, including `probe_dir_entry`
(`EXEC-007`, spec section 11.4 -- its own explicit `MINION_ARCHITECTURAL_MAPPING`, not a reuse of
`file_info`'s or `list_dir`'s cancellation precedent by analogy), accepts `signal` but never reads
it at all, matching pinned Pi's own reference implementation exactly, not a gap.

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
from collections.abc import Callable, Coroutine, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import urlparse

from ada_url import URL as _AdaURL
from ada_url import HostType as _AdaHostType
from ada_url import idna_to_unicode as _ada_idna_to_unicode

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


class DirEntryProbeKind(StrEnum):
    """`EXEC-007`, spec/execution.md section 11.4, `MINION_ARCHITECTURAL_MAPPING`. Richer than
    `FileKind`'s three-way split: distinguishes a symlink from what it resolves to, except for the
    disclosed `other` catch-all -- a symlink to an unclassifiable target (e.g. a FIFO) collapses to
    plain `other`, without a dedicated `symlink_to_other` value, mirroring a genuine limit in
    pinned Pi's own `ls.ts` (section 11.4's disclosed asymmetry)."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK_TO_FILE = "symlink_to_file"
    SYMLINK_TO_DIRECTORY = "symlink_to_directory"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class DirEntryProbe:
    """`EXEC-007`, spec/execution.md section 11.4. Describes the ADDRESSED entry -- the link
    itself, when the entry is a symlink -- never the resolved target. `path` is the RESOLVED path
    (the same section-3.2 rules `FileInfo.path` applies, via `resolve_local_path`); `name` is that
    resolved path's basename."""

    name: str
    path: str
    kind: DirEntryProbeKind


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
        resolved = os.path.normpath(normalized)
        if os.name == "nt":
            # `L12-PY-R002` (refined a third time): a bare drive letter (`C:`) or UNC share
            # (`\\host\share`) root, with nothing past it, gets a trailing separator here --
            # matching `path.win32.resolve`'s own root-canonicalization (live-verified:
            # `resolve('C:')` -> `"C:\\"`, `resolve('\\\\a\\share')` -> `"\\\\a\\share\\"`),
            # which `os.path.normpath` alone does not add.
            drive, tail = os.path.splitdrive(resolved)
            if drive and not tail:
                resolved += os.sep
        return resolved
    return os.path.normpath(os.path.join(cwd, normalized))


_WINDOWS_DRIVE_PATH_RE = re.compile(r"^/[A-Za-z]:")

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _strict_percent_decode(value: str) -> str:
    """Byte-accurate equivalent of JavaScript's `decodeURIComponent` -- percent-decodes `%XX`
    escapes to raw bytes (an unescaped character contributes its own UTF-8 bytes), then decodes
    the WHOLE byte sequence as UTF-8. Raises `ValueError` for a truncated/non-hex escape or
    invalid UTF-8, matching `decodeURIComponent`'s own thrown `URIError` (`UnicodeDecodeError`
    is itself a `ValueError` subclass, so both failure modes reach the caller identically)."""
    raw = bytearray()
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        if ch == "%":
            if i + 3 > n or value[i + 1] not in _HEX_DIGITS or value[i + 2] not in _HEX_DIGITS:
                raise ValueError(f"invalid percent-encoding in {value!r}")
            raw.append(int(value[i + 1 : i + 3], 16))
            i += 3
        else:
            raw.extend(ch.encode("utf-8"))
            i += 1
    return raw.decode("utf-8")


def _domain_to_unicode(host: str) -> str:
    """Decodes an already `ada_url.URL`-validated ASCII host's `"xn--..."` labels to their
    Unicode glyphs (`"xn--fa-hia"` -> `"faß"`), matching Node's own `domainToUnicode`
    (`L12-PY-R002`, root-characterization checkpoint -- `minion-agent-docs#121` @
    `00afd5178d5c1bed4ec5175eea873061a9928fb1`, `AGREED FOR IMPLEMENTATION: YES`).

    Node's `fileURLToPath`/`domainToUnicode` delegate this ENTIRE responsibility -- both the
    validation `_file_url_to_path`'s own `ada_url.URL(...)` construction already performed
    (bidi, leading-combining-mark, codepoint-assignment, ...) and this decode step -- to a
    single concrete engine: Ada (confirmed directly from Node v22.19.0's own source,
    `node_url.cc`: `ada::idna::to_unicode(get_hostname())`). Earlier revisions of this function
    hand-composed the third-party `idna` package's own IDNA2008/UTS46 validation with Python's
    `unicodedata` as a substitute for that engine -- independently proven, by a direct
    Ada-2.9.2 executable oracle built and differentially tested against an 8,246-case systematic
    corpus (`assurance/layers/data/12-python-r002-ada-oracle/`), NOT to be a faithful structural
    match: `idna`'s own validity table disagrees with Ada 2.9.2's actual (differently-shaped,
    and in at least one case genuinely buggy -- a verified LTR-bidi off-by-one in Ada 2.9.2's
    own `is_label_valid`) behavior in both directions, independent of which Unicode version
    either targets. `ada-url==1.15.3` (pinned exactly in `pyproject.toml`, NOT a floor) is Ada
    2.9.2's own contemporary PyPI release -- proven, not assumed, to match the direct Ada 2.9.2
    oracle EXACTLY across all 8,246 corpus cases, INCLUDING Ada 2.9.2's own bidi bug (required
    for, not in tension with, that exact match). Delegating to it directly, rather than
    hand-composing a competing implementation, is therefore the faithful choice, not merely the
    convenient one.

    `ada_url.idna_to_unicode()` itself never raises -- it returns a `"xn--..."` label UNCHANGED
    when it cannot decode it (matching Ada's own `to_unicode` C++ implementation, which falls
    back to the original input on failure rather than signaling an error). The caller
    (`_file_url_to_path`) treats an unchanged `"xn--..."` result as a rejected host, mirroring
    this exact convention -- verified as the correct signal against the full committed
    differential corpus, not merely assumed."""
    return _ada_idna_to_unicode(host)


def _file_url_to_path(url: str) -> str:
    """A characterized port of pinned Node's `fileURLToPath` (`L12-PY-R002`, root-characterization
    checkpoint -- `minion-agent-docs#121` @ `00afd5178d5c1bed4ec5175eea873061a9928fb1`, `AGREED
    FOR IMPLEMENTATION: YES`), verified against live Node 22 execution in both `windows: true`
    and `windows: false` modes (Node's own `fileURLToPath(url, {windows})` override) and against
    a direct Ada 2.9.2 executable oracle (Node v22.19.0's own vendored engine for this exact
    operation, confirmed from `node_url.cc` directly) over an 8,246-case systematic differential
    corpus (`assurance/layers/data/12-python-r002-ada-oracle/`), rather than guessed, trusted
    from a review's prose, or hand-composed from standards-adjacent Python libraries. This
    composes: `ada_url` (`ada-url==1.15.3`, pinned EXACTLY -- Ada 2.9.2's own contemporary
    release, proven exact behavioral parity with the pinned oracle, not merely assumed from
    sharing a project name) for host syntax/IPv4/IPv6/forbidden-code-point validation AND
    canonicalization AND, via `_domain_to_unicode`'s own thin wrapper over
    `ada_url.idna_to_unicode()`, the Punycode-to-Unicode decode step -- see `_domain_to_unicode`'s
    own docstring for why this now delegates the WHOLE host-conversion responsibility to Ada
    directly, superseding an earlier hand-composed `idna`-package-based validation layer this
    project's own differential-oracle investigation proved was not a faithful structural match --
    plus a hand-written layer below for the `fileURLToPath`-SPECIFIC rules neither `ada_url` nor
    Ada's own `to_unicode`/`to_ascii` primitives attempt: the encoded-separator guard,
    drive-letter validation, and backslash-to-`/` pre-normalization. Node's real algorithm, as
    observed:

    0. A RAW (unescaped) backslash ANYWHERE in the URL text is normalized to `/` BEFORE any
       other parsing -- a WHATWG "special scheme" URL-parsing-stage rule (`file:`, like `http:`,
       treats `\\` as a path-separator equivalent to `/`), universal across BOTH platform modes,
       not Windows-specific (`file://host\\share\\file` -> `\\\\host\\share\\file` even under
       `{windows: false}`, where it becomes a POSIX-style path since no separator conversion
       applies there -- confirmed by live probe). An *encoded* `%5c` is NOT touched by this step
       (it is a different 3-character sequence, not a literal backslash byte) -- that is a
       SEPARATE, later check (step 3).
    1. The host is validated and canonicalized by `ada_url.URL` (`ada-url==1.15.3`, pinned
       EXACTLY -- Ada 2.9.2's own contemporary release, proven to match Node v22.19.0's own
       vendored Ada 2.9.2 engine exactly across an 8,246-case differential corpus, including
       IPv4/IPv6 rejection/canonicalization, forbidden-code-point rejection, AND full bidi/
       leading-combining-mark/codepoint-assignment validation for domain hosts -- confirmed
       `ada-url==1.15.3`'s own `URL(...)` constructor raises directly for a host that fails any
       of these, unlike the newer `ada-url==4.0.0` this project used to depend on, which passed
       several of these through unvalidated). An invalid-range IPv4-shaped host
       (`256.256.256.256`, `1.2.3.4.5`) is REJECTED, not passed through as a literal domain
       label; an IPv6 literal (`[::ffff:192.168.1.1]`) is CANONICALIZED
       (`-> [::ffff:c0a8:101]`), not kept as typed; a decoded host containing any WHATWG
       "forbidden host code point" (space/control/``#%/:<>?@[\\]^|``) is rejected
       (`file://%2541/share` -- decodes to the literal string `%41`, still containing `%` -- is
       rejected; `file://%41/share` -- decodes cleanly to `A` -- is accepted); a domain host is
       ASCII-lowercased and non-ASCII input is converted to its Punycode (`xn--...`) ASCII form.
       `ada_url`'s own `host_type` distinguishes an IPv4/IPv6 literal (exempt from the
       domain-specific step below -- an IPv6 host's brackets are exactly what delimits it, not a
       forbidden character on that host type) from an ordinary domain. A domain-typed
       (`ada_url.HostType.DEFAULT`) host that survived construction is additionally passed
       through `_domain_to_unicode` (a thin wrapper over `ada_url.idna_to_unicode()` itself,
       Ada's own decode-for-display primitive -- see its own docstring for why this now
       delegates the WHOLE decode responsibility to Ada directly, rather than hand-composing a
       validation surface on top) -- an `"xn--..."` punycode label decodes to its Unicode glyphs
       (`"xn--fa-hia"` -> `"faß"`). Since `ada_url.idna_to_unicode()` returns a `"xn--..."`
       label UNCHANGED rather than raising when it cannot decode it, an unchanged result is
       treated as rejection here, matching Ada's own C++ `to_unicode` fallback convention
       (verified as the correct signal against the full committed differential corpus).
    2. On Windows specifically, a host that is (after decoding) exactly `"localhost"` is treated
       identically to an EMPTY host -- routed to the drive-letter branch (6), not the UNC branch
       (5) (`file://localhost/C:/foo` resolves to `C:\foo`, not a UNC path to a literal
       `localhost` share).
    3. The RAW (undecoded, POST-backslash-normalization) pathname is scanned for the literal
       case-insensitive substrings `%2f` (encoded `/`) and, on Windows only, `%5c` (encoded
       `\\`) -- either one, anywhere, rejects the whole URL BEFORE decoding (a path-traversal/
       ambiguity guard Node applies even to a host-qualified UNC path).
    4. The pathname is THEN percent-decoded (UTF-8, strict) -- an incomplete/invalid escape
       (`%ZZ`, a bare trailing `%`, invalid UTF-8 continuation bytes) rejects the whole URL.
    5. Non-empty, non-`"localhost"` host (Windows) / ANY non-empty host (this function is only
       ever reached from a Windows-or-POSIX-routed caller, see below): the result is a UNC path,
       ``\\\\<decoded-host><decoded-pathname>``, with every `/` in the FINAL combined string
       converted to `\\`.
    6. Empty (or `"localhost"`-collapsed) host, Windows: the DECODED pathname must start with
       `/<ascii-letter>:` (checked AFTER decoding, not before -- `file:///C%3A/foo` decodes to
       `/C:/foo` and IS valid, even though the raw path `/C%3A/foo` would not match a raw-text
       check) or the whole URL is rejected; the leading `/` is dropped and remaining `/`
       converted to `\\`.
    7. Empty host, POSIX: the host must be `""` or `"localhost"` (any other host rejects,
       independent of its own encoding validity); the decoded pathname is returned as-is (an
       EMPTY decoded pathname -- `"file://"` alone, with no trailing `/` at all -- becomes `"/"`
       rather than the empty string), with NO backslash-to-slash conversion (POSIX has no such
       convention -- but any RAW backslash was already normalized to `/` at step 0, universally).

    A bare drive letter (`C:`) or UNC share (`\\\\host\\share`) RESULT with nothing past it is
    given a trailing separator by `resolve_local_path`'s own final `os.path.normpath` step
    (mirroring `path.win32.resolve`'s own root-canonicalization, empirically confirmed via live
    probes of `resolve('C:')` -> `"C:\\\\"` and `resolve('\\\\\\\\a\\\\share')` ->
    `"\\\\\\\\a\\\\share\\\\"`), not here -- this function returns the bare, un-rooted form."""
    windows = os.name == "nt"
    normalized_url = url.replace("\\", "/")
    parsed = urlparse(normalized_url)

    host = ""
    if parsed.netloc:
        try:
            ada = _AdaURL(normalized_url)
        except ValueError as exc:
            raise ValueError(
                f"file:// URL host is not a valid hostname: {parsed.netloc!r}"
            ) from exc
        host = ada.host
        has_punycode_label = ada.host_type == _AdaHostType.DEFAULT and any(
            label.startswith("xn--") for label in host.split(".")
        )
        if has_punycode_label:
            decoded = _domain_to_unicode(host)
            if decoded == host:  # pragma: no cover
                # Defensive: matches _domain_to_unicode's own documented "unchanged means
                # Ada's to_unicode could not decode it" convention, but empirically
                # unreachable with the exact-pinned ada-url==1.15.3 -- every witness that
                # would make idna_to_unicode() fail to decode (bidi, leading-combining-mark,
                # malformed Punycode, disallowed/unassigned codepoint, ANY label position in
                # a multi-label host, live-probe-confirmed) already makes the ada_url.URL(...)
                # construction above raise first, so this branch never observably fires
                # against that dependency. Retained rather than removed: it is the correct
                # safety net if a future exact-pin change ever narrows what ada_url.URL(...)
                # itself rejects at parse time.
                raise ValueError(f"file:// URL host is not a valid hostname: {parsed.netloc!r}")
            host = decoded
    windows_empty_host = windows and host == "localhost"

    raw_pathname = parsed.path
    lowered_pathname = raw_pathname.lower()
    if "%2f" in lowered_pathname:
        raise ValueError(
            f"file:// URL path must not include an encoded / character: {raw_pathname!r}"
        )
    if windows and "%5c" in lowered_pathname:
        raise ValueError(
            f"file:// URL path must not include an encoded \\ character: {raw_pathname!r}"
        )
    pathname = _strict_percent_decode(raw_pathname)

    if windows:
        if host and not windows_empty_host:
            return ("\\\\" + host + pathname).replace("/", "\\")
        if not _WINDOWS_DRIVE_PATH_RE.match(pathname):
            raise ValueError(f"file:// URL path has no drive letter on Windows: {pathname!r}")
        return pathname[1:].replace("/", "\\")

    if host and host != "localhost":
        raise ValueError(f"file:// URL host is not local: {parsed.netloc!r}")
    return pathname if pathname else "/"


file_url_to_path = _file_url_to_path
"""The same certified conversion under a public name, for callers that need its failure UNWRAPPED
(spec/tools.md `TOOL-026` step 4, `R002-A`: a malformed `file://` path is rejected before any
`ctx.fs` access). Visibility only -- no behavior change; `resolve_local_path` keeps its own
suppress-and-fall-through wrapper. Raises `ValueError` (or `OSError`) on a malformed URL."""


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


def _list_dir_raw_sync(path: str, signal: RunSignal | None) -> list[str]:
    """`EXEC-007`, spec section 11.3. Analogous in shape to `_list_dir_sync` minus its own
    per-entry classification loop: ONE pre-aborted checkpoint, no per-entry checkpoint (there is
    no per-entry loop to checkpoint within). Returns raw provider/OS enumeration order,
    deliberately unsorted -- ordering is Layer 13's own responsibility (section 11.5), not this
    operation's."""
    if signal is not None and signal.aborted:
        raise _AbortedSignal
    return os.listdir(path)


def _probe_dir_entry_sync(path: str) -> DirEntryProbe:
    """`EXEC-007`, spec section 11.4. `path` is already the RESOLVED path (the caller applies
    `resolve_local_path` before invoking this). Reuses `_file_kind_from_stat` for both the
    non-following classification (mirroring `file_info`'s own `lstat`-based check) and, only when
    the addressed entry is itself a symlink, a SECOND following `stat` of its target -- rather than
    duplicating the stat-bit logic `_file_kind_from_stat` already owns. A target kind
    `_file_kind_from_stat` does not recognize (FIFO, socket, device, ...) collapses to `other` in
    both branches: directly for a non-symlink entry, and via the disclosed symlink-to-other
    asymmetry (no `symlink_to_other` value) when the entry is a symlink to such a target. A broken
    symlink's following `stat` raises `FileNotFoundError` (an `OSError`), left to the caller to
    convert via `to_fs_error` -- this is this call's OWN `Result` error, never a raised exception
    escaping the seam."""
    st = os.lstat(path)
    if _stat.S_ISLNK(st.st_mode):
        target_kind = _file_kind_from_stat(os.stat(path))
        if target_kind is FileKind.FILE:
            kind = DirEntryProbeKind.SYMLINK_TO_FILE
        elif target_kind is FileKind.DIRECTORY:
            kind = DirEntryProbeKind.SYMLINK_TO_DIRECTORY
        else:
            kind = DirEntryProbeKind.OTHER
    else:
        direct_kind = _file_kind_from_stat(st)
        if direct_kind is FileKind.FILE:
            kind = DirEntryProbeKind.FILE
        elif direct_kind is FileKind.DIRECTORY:
            kind = DirEntryProbeKind.DIRECTORY
        else:
            kind = DirEntryProbeKind.OTHER
    return DirEntryProbe(name=os.path.basename(path), path=path, kind=kind)


def _libc_access() -> Callable[[bytes, int], int]:  # pragma: no cover -- POSIX-only (libc)
    """The host C library's own `access(2)`, called with `use_errno` so its failure errno survives.
    `os.access` makes the same call but reports only a boolean, discarding why it failed."""
    import ctypes

    access = ctypes.CDLL(None, use_errno=True).access
    access.argtypes = [ctypes.c_char_p, ctypes.c_int]
    access.restype = ctypes.c_int
    return access


def _check_readable_posix(path: str) -> None:
    """`EXEC-008` on POSIX, spec section 12.4: exactly one `access(path, R_OK)` -- the call Node's
    `fs.access` makes -- evaluated with the process's real user/group IDs and following symlinks.
    Its own errno is kept (`ENOENT`, `ENOTDIR`, `EACCES`, `ELOOP`, `EIO`, ...) and classified by
    `to_fs_error`, never replaced by a fabricated one, so there is no second call whose failure
    could be misattributed. `access` does not open the target: no content is consumed and a FIFO
    without a writer cannot block."""
    import ctypes

    if _libc_access()(os.fsencode(path), os.R_OK) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), path)


# Win32: FILE_READ_DATA is also FILE_LIST_DIRECTORY; backup semantics lets CreateFileW open a
# directory at all (it does not bypass the ACL check unless the backup privilege is enabled).
_FILE_READ_DATA = 0x0001
_FILE_SHARE_ALL = 0x0001 | 0x0002 | 0x0004
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000


def _check_readable_windows(path: str) -> None:
    """`EXEC-008` on Windows, spec section 12.4 (owner decision, `MINION_ARCHITECTURAL_MAPPING`):
    the target's actual readability, not libuv's attribute-only `access`. One `CreateFileW` asking
    for `FILE_READ_DATA` -- read access to a file, list access to a directory -- following
    symlinks (no `FILE_FLAG_OPEN_REPARSE_POINT`); the handle is closed at once and nothing is
    read. A failure is raised as the matching `OSError` (built from the Win32 code, which Python
    maps to an errno and `OSError` subclass), which `to_fs_error` classifies like every other
    operation's."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateFileW(
        path,
        _FILE_READ_DATA,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        code = ctypes.get_last_error()
        raise OSError(None, ctypes.FormatError(code), path, code)
    kernel32.CloseHandle(handle)


class _EmbeddedNulPath(ValueError):
    """A path containing NUL cannot be passed to a native C-string API: the host would see only the
    prefix before the NUL and answer for a DIFFERENT target."""


def _check_readable_sync(path: str) -> None:
    """`EXEC-008`, spec section 12.3. `path` is already the RESOLVED path. An embedded NUL is
    rejected before either native call (`WP12E2-I002`), as pinned Node's `fs.access` rejects it
    (`ERR_INVALID_ARG_VALUE`) before reaching the host."""
    if "\x00" in path:
        raise _EmbeddedNulPath("embedded null character in path")
    if os.name == "nt":
        _check_readable_windows(path)
    else:
        _check_readable_posix(path)


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
    async def list_dir_raw(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[list[str], FsError]: ...
    async def probe_dir_entry(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[DirEntryProbe, FsError]: ...
    # `EXEC-008` (spec section 12), additive. A provider that cannot supply it returns
    # `Err(not_supported)` for every call -- a capability answer, never a target failure.
    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]: ...
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

    async def list_dir_raw(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[list[str], FsError]:
        """`EXEC-007`, spec section 11.3. Additive Layer-12 extension -- `list_dir` above is
        UNCHANGED by this method's addition."""
        resolved = resolve_local_path(self.cwd, path)
        try:
            names = await asyncio.to_thread(_list_dir_raw_sync, resolved, signal)
        except _AbortedSignal:
            return Err(_aborted(resolved))
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(names)

    async def probe_dir_entry(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[DirEntryProbe, FsError]:
        """`EXEC-007`, spec section 11.4. Additive Layer-12 extension -- `file_info` above is
        UNCHANGED by this method's addition. `signal` is accepted (uniform typed API shape) but
        deliberately never inspected, matching `file_info`'s own established behavior and this
        operation's own explicit `MINION_ARCHITECTURAL_MAPPING` cancellation classification."""
        resolved = resolve_local_path(self.cwd, path)
        try:
            probe = await asyncio.to_thread(_probe_dir_entry_sync, resolved)
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        return Ok(probe)

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        """`EXEC-008`, spec section 12. Additive Layer-12 extension -- no existing operation
        changes. Resolves with section 3.2's rules, follows symlinks, and answers whether the
        target exists and is readable without consuming content. `signal` is accepted but never
        inspected (section 12.3), like `file_info`/`probe_dir_entry`."""
        resolved = resolve_local_path(self.cwd, path)
        try:
            await asyncio.to_thread(_check_readable_sync, resolved)
        except OSError as exc:
            return Err(to_fs_error(exc, resolved))
        except _EmbeddedNulPath as exc:
            # Section 2.1's mapping of that rejection: Node's `ERR_INVALID_ARG_VALUE` is none of
            # `toFileError`'s listed codes (in particular not the `EINVAL` errno), so `unknown`.
            return Err(FsError(FsErrorCode.UNKNOWN, str(exc), resolved, exc))
        return Ok(None)

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
