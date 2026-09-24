"""`EXEC-008` (`WP-12.E2`): `check_readable`, spec/execution.md section 12.

Every section 12.6 witness is a plain `_witness_*` coroutine taking the provider class under test,
so the same witness body runs twice: once against `LocalFileSystem` (the positive tests below),
and once against each section-7 negative-control mutant (the `test_negative_control_*` tests),
where it MUST fail. A witness that passes both the correct implementation and a mutant would not
be discriminating evidence."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import stat as _stat
import subprocess
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from unittest import mock

import pytest

from minion_agent.execution import filesystem as filesystem_module
from minion_agent.execution.errors import FsError, FsErrorCode
from minion_agent.execution.filesystem import LocalFileSystem, resolve_local_path
from minion_agent.execution.result import Err, Ok, Result
from minion_agent.runtime.signal import RunAbortController, RunSignal

_WINDOWS = os.name == "nt"
_POSIX_ROOT = not _WINDOWS and os.geteuid() == 0

posix_only = pytest.mark.skipif(_WINDOWS, reason="POSIX-only section 12.6 row")
windows_only = pytest.mark.skipif(not _WINDOWS, reason="Windows-only section 12.6 row")
# root bypasses POSIX permission bits, so the permission rows are only meaningful unprivileged.
unprivileged = pytest.mark.skipif(_POSIX_ROOT, reason="root bypasses POSIX permission checks")

Provider = type[LocalFileSystem]
Witness = Callable[[Provider, Path], Awaitable[None]]


# ---------------------------------------------------------------------------
# Fixture helpers: host permission setup with guaranteed restoration
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _unreadable(path: Path, *, posix_mode: int = 0o000) -> Iterator[None]:
    """Make `path` unreadable to this process: on Windows a deny ACE for `RD` -- `FILE_READ_DATA`
    on a file, `FILE_LIST_DIRECTORY` on a directory, exactly the right readability is about, and
    unlike generic `R` it leaves `READ_CONTROL` so the ACE can be removed again; on POSIX
    `posix_mode`. Always restored."""
    if _WINDOWS:
        user = os.environ["USERNAME"]
        subprocess.run(
            ["icacls", str(path), "/deny", f"{user}:(RD)"], check=True, capture_output=True
        )
        try:
            yield
        finally:
            subprocess.run(["icacls", str(path), "/reset"], check=True, capture_output=True)
    else:
        original = _stat.S_IMODE(os.stat(path).st_mode)
        os.chmod(path, posix_mode)
        try:
            yield
        finally:
            os.chmod(path, original)


@contextlib.contextmanager
def _posix_mode(path: Path, mode: int) -> Iterator[None]:
    original = _stat.S_IMODE(os.stat(path).st_mode)
    os.chmod(path, mode)
    try:
        yield
    finally:
        os.chmod(path, original)


def _assert_ok(result: Result[None, FsError]) -> None:
    assert isinstance(result, Ok), f"expected Ok(None), got {result!r}"
    assert result.value is None


def _assert_err(result: Result[None, FsError], code: FsErrorCode) -> None:
    assert isinstance(result, Err), f"expected Err({code.value}), got {result!r}"
    assert result.error.code == code, f"expected {code.value}, got {result.error.code.value}"


# ---------------------------------------------------------------------------
# Section 12.6 witnesses (provider-parameterized)
# ---------------------------------------------------------------------------


async def _witness_readable_file_direct_and_through_symlink(fs_cls: Provider, tmp: Path) -> None:
    """READABLE REGULAR FILE, DIRECTLY AND THROUGH A SYMLINK."""
    (tmp / "f").write_text("x")
    os.symlink(tmp / "f", tmp / "link")
    fs = fs_cls(cwd=str(tmp))
    _assert_ok(await fs.check_readable("f"))
    _assert_ok(await fs.check_readable("link"))


async def _witness_unreadable_file_direct_and_through_symlink(fs_cls: Provider, tmp: Path) -> None:
    """UNREADABLE REGULAR FILE, DIRECTLY AND THROUGH A SYMLINK: POSIX mode 000, Windows a
    deny-read ACE -> `permission_denied` for both, on both hosts."""
    (tmp / "f000").write_text("x")
    os.symlink(tmp / "f000", tmp / "link")
    fs = fs_cls(cwd=str(tmp))
    with _unreadable(tmp / "f000"):
        direct = await fs.check_readable("f000")
        through_link = await fs.check_readable("link")
    _assert_err(direct, FsErrorCode.PERMISSION_DENIED)
    _assert_err(through_link, FsErrorCode.PERMISSION_DENIED)


async def _witness_dangling_symlink(fs_cls: Provider, tmp: Path) -> None:
    """DANGLING SYMLINK -> `not_found`, on both hosts."""
    os.symlink(tmp / "missing-target", tmp / "dangling")
    fs = fs_cls(cwd=str(tmp))
    _assert_err(await fs.check_readable("dangling"), FsErrorCode.NOT_FOUND)


async def _witness_missing_path_and_non_directory_component(fs_cls: Provider, tmp: Path) -> None:
    """MISSING PATH / NON-DIRECTORY COMPONENT: `not_found`; for `file/x` the host's section 2.1
    mapping (POSIX `not_directory`, Windows `not_found`)."""
    (tmp / "file").write_text("x")
    fs = fs_cls(cwd=str(tmp))
    _assert_err(await fs.check_readable("missing"), FsErrorCode.NOT_FOUND)
    component = FsErrorCode.NOT_FOUND if _WINDOWS else FsErrorCode.NOT_DIRECTORY
    _assert_err(await fs.check_readable("file/x"), component)


async def _witness_readable_directory(fs_cls: Provider, tmp: Path) -> None:
    """READABLE DIRECTORY -> `Ok(None)`: a directory is readable, not an error here."""
    (tmp / "d").mkdir()
    fs = fs_cls(cwd=str(tmp))
    _assert_ok(await fs.check_readable("d"))


async def _witness_read_but_not_search_directory(fs_cls: Provider, tmp: Path) -> None:
    """READ-BUT-NOT-SEARCH DIRECTORY (POSIX, 0444) -> `Ok(None)`: search permission on the
    target directory itself is not required."""
    (tmp / "dr").mkdir()
    fs = fs_cls(cwd=str(tmp))
    with _posix_mode(tmp / "dr", 0o444):
        result = await fs.check_readable("dr")
    _assert_ok(result)


async def _witness_unreadable_directory(fs_cls: Provider, tmp: Path) -> None:
    """UNREADABLE DIRECTORY (POSIX 000) / DENY-READ DIRECTORY (Windows deny-list ACE) ->
    `permission_denied`."""
    (tmp / "d000").mkdir()
    fs = fs_cls(cwd=str(tmp))
    with _unreadable(tmp / "d000"):
        result = await fs.check_readable("d000")
    _assert_err(result, FsErrorCode.PERMISSION_DENIED)


async def _witness_search_only_directory(fs_cls: Provider, tmp: Path) -> None:
    """SEARCH-ONLY DIRECTORY (POSIX, 0111) -> `permission_denied`."""
    (tmp / "dx").mkdir()
    fs = fs_cls(cwd=str(tmp))
    with _posix_mode(tmp / "dx", 0o111):
        result = await fs.check_readable("dx")
    _assert_err(result, FsErrorCode.PERMISSION_DENIED)


async def _witness_unsearchable_parent(fs_cls: Provider, tmp: Path) -> None:
    """UNSEARCHABLE PARENT (POSIX): `dr/inner` where `dr` is 0444 -> `permission_denied`."""
    (tmp / "dr").mkdir()
    (tmp / "dr" / "inner").write_text("x")
    fs = fs_cls(cwd=str(tmp))
    with _posix_mode(tmp / "dr", 0o444):
        result = await fs.check_readable("dr/inner")
    _assert_err(result, FsErrorCode.PERMISSION_DENIED)


async def _witness_symlink_loop(fs_cls: Provider, tmp: Path) -> None:
    """SYMLINK LOOP: `a -> b`, `b -> a` -> the section 2.1 mapping of the host error: `ELOOP ->
    unknown` on POSIX (the section 12.6 row); on Windows the host reports
    `ERROR_CANT_RESOLVE_FILENAME` (`EINVAL -> invalid`), exactly as the unchanged Layer-12
    `read_binary_file` and `canonical_path` already classify the same loop."""
    os.symlink(tmp / "b", tmp / "a")
    os.symlink(tmp / "a", tmp / "b")
    fs = fs_cls(cwd=str(tmp))
    expected = FsErrorCode.INVALID if _WINDOWS else FsErrorCode.UNKNOWN
    _assert_err(await fs.check_readable("a"), expected)


async def _witness_fifo_no_content_consumed(fs_cls: Provider, tmp: Path) -> None:
    """NO CONTENT CONSUMED: a FIFO with no writer -> `Ok(None)` promptly; the call does not
    block. If an implementation does block (it opened the FIFO to read), a writer is attached so
    the stuck worker thread is released before the witness fails."""
    fifo = tmp / "fifo"
    os.mkfifo(fifo)
    fs = fs_cls(cwd=str(tmp))
    task = asyncio.ensure_future(fs.check_readable("fifo"))
    done, _ = await asyncio.wait({task}, timeout=5.0)
    if not done:
        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
        await task
        raise AssertionError("check_readable blocked on a FIFO without a writer")
    _assert_ok(task.result())


async def _witness_relative_path_resolution(fs_cls: Provider, tmp: Path) -> None:
    """RELATIVE PATH RESOLUTION: with cwd `<tmp>/workspace`, `check_readable("sub/f")` answers
    for `<tmp>/workspace/sub/f` (section 3.2, as `file_info`), not for the process's own cwd; a
    failure reports the resolved path."""
    workspace = tmp / "workspace"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "sub" / "f").write_text("x")
    fs = fs_cls(cwd=str(workspace))
    _assert_ok(await fs.check_readable("sub/f"))
    _assert_ok(await fs.check_readable("sub/../sub/f"))
    missing = await fs.check_readable("sub/g")
    _assert_err(missing, FsErrorCode.NOT_FOUND)
    assert isinstance(missing, Err)
    assert missing.error.path == resolve_local_path(str(workspace), "sub/g")


async def _witness_target_vanishes_before_the_readability_query(
    fs_cls: Provider, tmp: Path
) -> None:
    """WP12E2-I001 (POSIX): the target disappears after any preliminary metadata query has
    succeeded and before the readability query runs. The answer is the readability query's own
    `ENOENT -> not_found` (pinned Node `fs.access(R_OK)` on an absent target is `ENOENT`), never a
    `permission_denied` inferred from a boolean. Both hooks remove the target at most once: a
    successful `os.stat` of it removes it afterwards; the `access(2)` query removes it first."""
    target = tmp / "f"
    target.write_text("x")
    real_stat = os.stat
    real_libc_access = filesystem_module._libc_access

    def remove_target() -> None:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()

    def stat_then_remove(path: object, *args: object, **kwargs: object) -> os.stat_result:
        result = real_stat(path, *args, **kwargs)  # type: ignore[arg-type]
        if os.fspath(path) == str(target):  # type: ignore[arg-type]
            remove_target()
        return result

    def libc_access() -> Callable[[bytes, int], int]:
        access = real_libc_access()

        def remove_then_access(path: bytes, mode: int) -> int:
            remove_target()
            return access(path, mode)

        return remove_then_access

    fs = fs_cls(cwd=str(tmp))
    with (
        mock.patch.object(os, "stat", stat_then_remove),
        mock.patch.object(filesystem_module, "_libc_access", libc_access),
    ):
        result = await fs.check_readable("f")
    assert not target.exists()
    _assert_err(result, FsErrorCode.NOT_FOUND)


async def _witness_pre_aborted_signal_not_inspected(fs_cls: Provider, tmp: Path) -> None:
    """SIGNAL ACCEPTED, NOT INSPECTED: a pre-aborted signal on a readable file -> `Ok(None)`."""
    (tmp / "f").write_text("x")
    controller = RunAbortController()
    controller.abort()
    fs = fs_cls(cwd=str(tmp))
    _assert_ok(await fs.check_readable("f", signal=controller.signal))
    _assert_ok(await fs.check_readable("f", controller.signal))


async def _witness_provider_without_extension(fs_cls: Provider, tmp: Path) -> None:
    """PROVIDER WITHOUT THE EXTENSION: `Err(not_supported)` for every path, including readable
    ones and directories; never a silent success, never a fallback to `file_info`/`exists` (a
    fallback would answer `not_found` for the missing path)."""
    (tmp / "f").write_text("x")
    (tmp / "d").mkdir()
    fs = fs_cls(cwd=str(tmp))
    for path in ("f", "d", "missing"):
        _assert_err(await fs.check_readable(path), FsErrorCode.NOT_SUPPORTED)


class _ProviderWithoutExec008(LocalFileSystem):
    """A provider that cannot supply `EXEC-008` answers the capability question, as section 12.2
    requires. `LocalFileSystem` itself always implements it (a first-party `not_supported` would
    be a defect, section 12.5); this stand-in exists only to hold the capability answer to the
    contract and to be distinguishable from every target failure above."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        return Err(FsError(FsErrorCode.NOT_SUPPORTED, "check_readable is not supported", path))


# ---------------------------------------------------------------------------
# Positive witnesses: LocalFileSystem (and the capability stand-in)
# ---------------------------------------------------------------------------


async def test_readable_regular_file_directly_and_through_symlink(tmp_path: Path) -> None:
    await _witness_readable_file_direct_and_through_symlink(LocalFileSystem, tmp_path)


@unprivileged
async def test_unreadable_regular_file_directly_and_through_symlink(tmp_path: Path) -> None:
    await _witness_unreadable_file_direct_and_through_symlink(LocalFileSystem, tmp_path)


async def test_dangling_symlink_is_not_found(tmp_path: Path) -> None:
    await _witness_dangling_symlink(LocalFileSystem, tmp_path)


async def test_missing_path_and_non_directory_component(tmp_path: Path) -> None:
    await _witness_missing_path_and_non_directory_component(LocalFileSystem, tmp_path)


async def test_readable_directory_is_ok(tmp_path: Path) -> None:
    await _witness_readable_directory(LocalFileSystem, tmp_path)


@posix_only
@unprivileged
async def test_posix_read_but_not_search_directory_is_ok(tmp_path: Path) -> None:
    await _witness_read_but_not_search_directory(LocalFileSystem, tmp_path)


@unprivileged
async def test_unreadable_directory_posix_000_or_windows_deny_list(tmp_path: Path) -> None:
    await _witness_unreadable_directory(LocalFileSystem, tmp_path)


@posix_only
@unprivileged
async def test_posix_search_only_directory_is_permission_denied(tmp_path: Path) -> None:
    await _witness_search_only_directory(LocalFileSystem, tmp_path)


@posix_only
@unprivileged
async def test_posix_unsearchable_parent_is_permission_denied(tmp_path: Path) -> None:
    await _witness_unsearchable_parent(LocalFileSystem, tmp_path)


async def test_symlink_loop_uses_the_host_error_mapping(tmp_path: Path) -> None:
    await _witness_symlink_loop(LocalFileSystem, tmp_path)


@windows_only
async def test_windows_symlink_loop_matches_the_unchanged_layer12_operations(
    tmp_path: Path,
) -> None:
    """The Windows loop classification is not EXEC-008's own choice: it is the same host-error
    mapping the certified `read_binary_file`/`canonical_path` apply to the same loop."""
    os.symlink(tmp_path / "b", tmp_path / "a")
    os.symlink(tmp_path / "a", tmp_path / "b")
    fs = LocalFileSystem(cwd=str(tmp_path))
    check = await fs.check_readable("a")
    read = await fs.read_binary_file("a")
    canonical = await fs.canonical_path("a")
    assert isinstance(check, Err) and isinstance(read, Err) and isinstance(canonical, Err)
    assert check.error.code == read.error.code == canonical.error.code


@posix_only
async def test_fifo_without_writer_is_ok_and_does_not_block(tmp_path: Path) -> None:
    await _witness_fifo_no_content_consumed(LocalFileSystem, tmp_path)


async def test_relative_path_resolution(tmp_path: Path) -> None:
    await _witness_relative_path_resolution(LocalFileSystem, tmp_path)


@posix_only
async def test_target_vanishing_before_the_readability_query_is_not_found(tmp_path: Path) -> None:
    await _witness_target_vanishes_before_the_readability_query(LocalFileSystem, tmp_path)


async def test_pre_aborted_signal_is_accepted_but_not_inspected(tmp_path: Path) -> None:
    await _witness_pre_aborted_signal_not_inspected(LocalFileSystem, tmp_path)


async def test_provider_without_exec008_answers_not_supported(tmp_path: Path) -> None:
    await _witness_provider_without_extension(_ProviderWithoutExec008, tmp_path)


async def test_local_filesystem_never_answers_not_supported(tmp_path: Path) -> None:
    """Section 12.5: the certified first-party provider MUST implement `check_readable`, so its
    answers are target answers, never the capability answer."""
    (tmp_path / "f").write_text("x")
    fs = LocalFileSystem(cwd=str(tmp_path))
    for path in ("f", ".", "missing"):
        result = await fs.check_readable(path)
        assert not (isinstance(result, Err) and result.error.code == FsErrorCode.NOT_SUPPORTED)


async def test_check_readable_leaves_content_and_existing_operations_unchanged(
    tmp_path: Path,
) -> None:
    """EXISTING OPERATIONS UNCHANGED (regression, alongside the unchanged section 10/11.6 suite in
    `test_filesystem.py`): checking a file, a directory and a symlink changes neither their
    content nor what the certified operations report for them afterwards."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("f.txt", "content")
    await fs.create_dir("d")
    os.symlink(tmp_path / "f.txt", tmp_path / "link")

    before = [
        await fs.file_info("f.txt"),
        await fs.file_info("d"),
        await fs.file_info("link"),
        await fs.canonical_path("link"),
        await fs.read_binary_file("f.txt"),
        await fs.list_dir("."),
    ]
    for path in ("f.txt", "d", "link"):
        _assert_ok(await fs.check_readable(path))
    after = [
        await fs.file_info("f.txt"),
        await fs.file_info("d"),
        await fs.file_info("link"),
        await fs.canonical_path("link"),
        await fs.read_binary_file("f.txt"),
        await fs.list_dir("."),
    ]
    assert before == after
    assert (tmp_path / "f.txt").read_text() == "content"


# ---------------------------------------------------------------------------
# POSIX branch, exercised on any host (host-independent coverage of `_check_readable_posix`)
# ---------------------------------------------------------------------------


def _fake_libc_access(
    monkeypatch: pytest.MonkeyPatch, errno_value: int | None
) -> list[tuple[bytes, int]]:
    """Route the POSIX branch on any host through a stand-in for libc `access(2)`: success when
    `errno_value` is None, else `-1` with that errno set exactly as `use_errno` would."""
    import ctypes

    calls: list[tuple[bytes, int]] = []

    def access(path: bytes, mode: int) -> int:
        calls.append((path, mode))
        if errno_value is None:
            return 0
        ctypes.set_errno(errno_value)
        return -1

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(filesystem_module, "_libc_access", lambda: access)
    return calls


async def test_posix_branch_ok_is_one_access_r_ok_on_the_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The POSIX check is exactly one `access(resolved, R_OK)` -- no preliminary metadata call."""
    calls = _fake_libc_access(monkeypatch, None)
    stat_calls: list[object] = []
    real_stat = os.stat
    monkeypatch.setattr(
        filesystem_module.os, "stat", lambda *a, **k: stat_calls.append(a) or real_stat(*a, **k)
    )
    _assert_ok(await LocalFileSystem(cwd=str(tmp_path)).check_readable("sub/f"))
    assert calls == [(os.fsencode(str(tmp_path / "sub" / "f")), os.R_OK)]
    assert stat_calls == []


@pytest.mark.parametrize(
    ("errno_value", "code"),
    [
        (errno.ENOENT, FsErrorCode.NOT_FOUND),
        (errno.ENOTDIR, FsErrorCode.NOT_DIRECTORY),
        (errno.EACCES, FsErrorCode.PERMISSION_DENIED),
        (errno.ELOOP, FsErrorCode.UNKNOWN),
        (errno.EIO, FsErrorCode.UNKNOWN),
        (errno.EINVAL, FsErrorCode.INVALID),
    ],
)
async def test_posix_branch_keeps_the_access_calls_own_errno(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, errno_value: int, code: FsErrorCode
) -> None:
    """WP12E2-I001: the failure reason is `access(2)`'s own errno, classified by section 2.1 --
    never a fabricated `EACCES` for whatever made the check fail."""
    _fake_libc_access(monkeypatch, errno_value)
    result = await LocalFileSystem(cwd=str(tmp_path)).check_readable("f")
    _assert_err(result, code)
    assert isinstance(result, Err)
    assert isinstance(result.error.cause, OSError)
    assert result.error.cause.errno == errno_value
    assert result.error.path == str(tmp_path / "f")


# ---------------------------------------------------------------------------
# Section 7 negative controls: each mutant MUST fail at least one witness above
# ---------------------------------------------------------------------------


class _FileInfoExistenceOnly(LocalFileSystem):
    """Mutant: readability == `file_info` (lstat) succeeded."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        info = await self.file_info(path)
        return Ok(None) if isinstance(info, Ok) else Err(info.error)


class _CanonicalPathExistenceOnly(LocalFileSystem):
    """Mutant: readability == `canonical_path` succeeded."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        canonical = await self.canonical_path(path)
        return Ok(None) if isinstance(canonical, Ok) else Err(canonical.error)


class _NodeWindowsAttributeOnlyAccess(LocalFileSystem):
    """Mutant: libuv's Windows `fs__access` -- `GetFileAttributesW` only, which checks existence
    of the (non-followed) final component and never read permission."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        import ctypes
        from ctypes import wintypes

        resolved = resolve_local_path(self.cwd, path)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetFileAttributesW.restype = wintypes.DWORD
        kernel32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
        if kernel32.GetFileAttributesW(resolved) == 0xFFFFFFFF:
            code = ctypes.get_last_error()
            exc = OSError(None, ctypes.FormatError(code), resolved, code)
            return Err(FsError(FsErrorCode.NOT_FOUND, str(exc), resolved, exc))
        return Ok(None)


class _OpenAndReadOneByte(LocalFileSystem):
    """Mutant: readability by opening and consuming one byte."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)

        def read_one_byte() -> None:
            with open(resolved, "rb") as handle:
                handle.read(1)

        try:
            await asyncio.to_thread(read_one_byte)
        except OSError as exc:
            return Err(filesystem_module.to_fs_error(exc, resolved))
        return Ok(None)


class _FinalSymlinkNotFollowed(LocalFileSystem):
    """Mutant: the correct check, except that a final-component symlink is answered for the link
    itself (it exists, and links are not permission-checked) instead of its target."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        if os.path.islink(resolved):
            return Ok(None)
        return await super().check_readable(path, signal)


class _StatThenBooleanAccess(LocalFileSystem):
    """Mutant (the WP12E2-I001 candidate): a symlink-following `os.stat`, then boolean
    `os.access(R_OK)`, fabricating `EACCES` for every false answer."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)

        def stat_then_access() -> None:
            os.stat(resolved)
            if not os.access(resolved, os.R_OK):
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), resolved)

        try:
            await asyncio.to_thread(stat_then_access)
        except OSError as exc:
            return Err(filesystem_module.to_fs_error(exc, resolved))
        return Ok(None)


class _PreAbortRejecting(LocalFileSystem):
    """Mutant: inspects the signal and rejects a pre-aborted one."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        if signal is not None and signal.aborted:
            return Err(FsError(FsErrorCode.ABORTED, "aborted", path))
        return await super().check_readable(path, signal)


class _SilentSuccessWithoutExec008(LocalFileSystem):
    """Mutant: a provider lacking the extension that silently answers `Ok`."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        return Ok(None)


class _ExistsFallbackWithoutExec008(LocalFileSystem):
    """Mutant: a provider lacking the extension that falls back to `exists`."""

    async def check_readable(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        exists = await self.exists(path)
        if isinstance(exists, Ok) and exists.value:
            return Ok(None)
        return Err(FsError(FsErrorCode.NOT_FOUND, "not found", path))


_NEGATIVE_CONTROLS: list[tuple[str, Provider, Witness, list[pytest.MarkDecorator]]] = [
    ("file_info-existence/unreadable-file", _FileInfoExistenceOnly,
     _witness_unreadable_file_direct_and_through_symlink, [unprivileged]),
    ("file_info-existence/dangling-link", _FileInfoExistenceOnly,
     _witness_dangling_symlink, []),
    ("file_info-existence/unreadable-directory", _FileInfoExistenceOnly,
     _witness_unreadable_directory, [unprivileged]),
    ("canonical_path-existence/unreadable-file", _CanonicalPathExistenceOnly,
     _witness_unreadable_file_direct_and_through_symlink, [unprivileged]),
    ("canonical_path-existence/unreadable-directory", _CanonicalPathExistenceOnly,
     _witness_unreadable_directory, [unprivileged]),
    ("node-windows-attributes/unreadable-file", _NodeWindowsAttributeOnlyAccess,
     _witness_unreadable_file_direct_and_through_symlink, [windows_only]),
    ("node-windows-attributes/dangling-link", _NodeWindowsAttributeOnlyAccess,
     _witness_dangling_symlink, [windows_only]),
    ("node-windows-attributes/unreadable-directory", _NodeWindowsAttributeOnlyAccess,
     _witness_unreadable_directory, [windows_only]),
    ("open-read-one-byte/readable-directory", _OpenAndReadOneByte,
     _witness_readable_directory, []),
    ("open-read-one-byte/fifo", _OpenAndReadOneByte,
     _witness_fifo_no_content_consumed, [posix_only]),
    ("final-symlink-not-followed/dangling-link", _FinalSymlinkNotFollowed,
     _witness_dangling_symlink, []),
    ("final-symlink-not-followed/unreadable-through-link", _FinalSymlinkNotFollowed,
     _witness_unreadable_file_direct_and_through_symlink, [unprivileged]),
    ("stat-then-boolean-access/target-vanishes-before-query", _StatThenBooleanAccess,
     _witness_target_vanishes_before_the_readability_query, [posix_only]),
    ("pre-abort-rejection/signal-not-inspected", _PreAbortRejecting,
     _witness_pre_aborted_signal_not_inspected, []),
    ("silent-success-without-exec008/provider-capability", _SilentSuccessWithoutExec008,
     _witness_provider_without_extension, []),
    ("exists-fallback-without-exec008/provider-capability", _ExistsFallbackWithoutExec008,
     _witness_provider_without_extension, []),
]  # fmt: skip


@pytest.mark.parametrize(
    ("mutant", "witness"),
    [
        pytest.param(mutant, witness, id=name, marks=marks)
        for name, mutant, witness, marks in _NEGATIVE_CONTROLS
    ],
)
async def test_negative_control_is_rejected_by_its_witness(
    mutant: Provider, witness: Witness, tmp_path: Path
) -> None:
    """Section 7: the named witness passes for `LocalFileSystem` (above) and MUST fail for the
    mutant; a witness that accepted both would not be discriminating."""
    with pytest.raises(AssertionError):
        await witness(mutant, tmp_path)
