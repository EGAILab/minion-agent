"""`EXEC-009` (`WP-12.E3`): `check_read_write`, spec/execution.md section 13.

Every section 13.6 witness is a plain `_witness_*` coroutine taking the provider class under test,
so the same witness body runs against `LocalFileSystem` (the positive tests) and against each
negative-control mutant (`test_negative_control_*`), where it MUST fail. A witness that passes
both the correct implementation and a mutant would not be discriminating evidence."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import stat as _stat
import subprocess
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from minion_agent.execution import filesystem as filesystem_module
from minion_agent.execution.errors import FsError, FsErrorCode
from minion_agent.execution.filesystem import LocalFileSystem, resolve_local_path
from minion_agent.execution.result import Err, Ok, Result
from minion_agent.runtime.signal import RunAbortController, RunSignal

_WINDOWS = os.name == "nt"
_POSIX_ROOT = not _WINDOWS and os.geteuid() == 0

posix_only = pytest.mark.skipif(_WINDOWS, reason="POSIX-only section 13.6 row")
windows_only = pytest.mark.skipif(not _WINDOWS, reason="Windows-only section 13.6 row")
# root bypasses POSIX permission bits, so the permission rows are only meaningful unprivileged
# (section 13.6: skipped and reported, never faked).
unprivileged = pytest.mark.skipif(_POSIX_ROOT, reason="root bypasses POSIX permission checks")

Provider = type[LocalFileSystem]
Witness = Callable[[Provider, Path], Awaitable[None]]


# ---------------------------------------------------------------------------
# Fixture helpers: host permission setup with guaranteed restoration
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _windows_deny(path: Path, rights: str) -> Iterator[None]:
    """A deny ACE for the current user with `rights` (icacls notation: `RD` read-data/list,
    `WD` write-data/add-file, `DC` delete-child), always removed afterwards."""
    user = os.environ["USERNAME"]
    subprocess.run(
        ["icacls", str(path), "/deny", f"{user}:({rights})"], check=True, capture_output=True
    )
    try:
        yield
    finally:
        subprocess.run(["icacls", str(path), "/reset"], check=True, capture_output=True)


@contextlib.contextmanager
def _posix_mode(path: Path, mode: int) -> Iterator[None]:
    original = _stat.S_IMODE(os.stat(path).st_mode)
    os.chmod(path, mode)
    try:
        yield
    finally:
        os.chmod(path, original)


@contextlib.contextmanager
def _not_writable(path: Path) -> Iterator[None]:
    """Readable but not writable: POSIX 0444, Windows a deny-write-data ACE."""
    if _WINDOWS:
        with _windows_deny(path, "WD"):
            yield
    else:
        with _posix_mode(path, 0o444):
            yield


@contextlib.contextmanager
def _not_readable(path: Path) -> Iterator[None]:
    """Writable but not readable: POSIX 0222, Windows a deny-read-data ACE."""
    if _WINDOWS:
        with _windows_deny(path, "RD"):
            yield
    else:
        with _posix_mode(path, 0o222):
            yield


@contextlib.contextmanager
def _windows_readonly_attribute(path: Path) -> Iterator[None]:
    """The Windows read-only attribute (`os.chmod` with `S_IREAD` sets it), always cleared."""
    os.chmod(path, _stat.S_IREAD)
    try:
        yield
    finally:
        os.chmod(path, _stat.S_IREAD | _stat.S_IWRITE)


def _assert_ok(result: Result[None, FsError]) -> None:
    assert isinstance(result, Ok), f"expected Ok(None), got {result!r}"
    assert result.value is None


def _assert_err(result: Result[None, FsError], code: FsErrorCode) -> None:
    assert isinstance(result, Err), f"expected Err({code.value}), got {result!r}"
    assert result.error.code == code, f"expected {code.value}, got {result.error.code.value}"


def _snapshot(path: Path) -> tuple[bytes, int, int]:
    st = os.stat(path)
    return path.read_bytes(), st.st_size, st.st_mtime_ns


# ---------------------------------------------------------------------------
# Section 13.6 witnesses (provider-parameterized)
# ---------------------------------------------------------------------------


async def _witness_rw_file_direct_and_through_symlink(fs_cls: Provider, tmp: Path) -> None:
    """READ+WRITE ACCESSIBLE REGULAR FILE, DIRECTLY AND THROUGH A SYMLINK: `Ok(None)` for both;
    the file's bytes, size and mtime are unchanged afterwards."""
    (tmp / "f").write_bytes(b"content\n")
    os.symlink(tmp / "f", tmp / "link")
    before = _snapshot(tmp / "f")
    fs = fs_cls(cwd=str(tmp))
    _assert_ok(await fs.check_read_write("f"))
    _assert_ok(await fs.check_read_write("link"))
    assert _snapshot(tmp / "f") == before


async def _witness_readable_not_writable(fs_cls: Provider, tmp: Path) -> None:
    """READABLE BUT NOT WRITABLE FILE: POSIX 0444 / Windows deny-write ACE -> `permission_denied`,
    directly and through a symlink -- the row that separates EXEC-009 from EXEC-008."""
    (tmp / "f").write_text("x")
    os.symlink(tmp / "f", tmp / "link")
    fs = fs_cls(cwd=str(tmp))
    with _not_writable(tmp / "f"):
        direct = await fs.check_read_write("f")
        through_link = await fs.check_read_write("link")
    _assert_err(direct, FsErrorCode.PERMISSION_DENIED)
    _assert_err(through_link, FsErrorCode.PERMISSION_DENIED)


async def _witness_windows_readonly_attribute_file(fs_cls: Provider, tmp: Path) -> None:
    """READABLE BUT NOT WRITABLE FILE, Windows read-only attribute (permissive ACL) ->
    `permission_denied`."""
    (tmp / "f").write_text("x")
    fs = fs_cls(cwd=str(tmp))
    with _windows_readonly_attribute(tmp / "f"):
        result = await fs.check_read_write("f")
    _assert_err(result, FsErrorCode.PERMISSION_DENIED)


async def _witness_writable_not_readable(fs_cls: Provider, tmp: Path) -> None:
    """WRITABLE BUT NOT READABLE FILE: POSIX 0222 / Windows deny-read ACE -> `permission_denied`."""
    (tmp / "f").write_text("x")
    fs = fs_cls(cwd=str(tmp))
    with _not_readable(tmp / "f"):
        result = await fs.check_read_write("f")
    _assert_err(result, FsErrorCode.PERMISSION_DENIED)


async def _witness_dangling_symlink(fs_cls: Provider, tmp: Path) -> None:
    """DANGLING SYMLINK -> `not_found`, on both hosts."""
    os.symlink(tmp / "missing-target", tmp / "dangling")
    fs = fs_cls(cwd=str(tmp))
    _assert_err(await fs.check_read_write("dangling"), FsErrorCode.NOT_FOUND)


async def _witness_missing_path_and_non_directory_component(fs_cls: Provider, tmp: Path) -> None:
    """MISSING PATH / NON-DIRECTORY COMPONENT: `not_found`; for `file/x` the host's section 2.1
    mapping (POSIX `not_directory`, Windows `not_found`)."""
    (tmp / "file").write_text("x")
    fs = fs_cls(cwd=str(tmp))
    _assert_err(await fs.check_read_write("missing"), FsErrorCode.NOT_FOUND)
    component = FsErrorCode.NOT_FOUND if _WINDOWS else FsErrorCode.NOT_DIRECTORY
    _assert_err(await fs.check_read_write("file/x"), component)


async def _witness_rw_directory(fs_cls: Provider, tmp: Path) -> None:
    """READ+WRITE DIRECTORY -> `Ok(None)`: a directory is not an error here."""
    (tmp / "d").mkdir()
    fs = fs_cls(cwd=str(tmp))
    _assert_ok(await fs.check_read_write("d"))


async def _witness_rw_but_not_search_directory(fs_cls: Provider, tmp: Path) -> None:
    """READ+WRITE-BUT-NOT-SEARCH DIRECTORY (POSIX 0666) -> `Ok(None)`, and a following create in
    it still fails `permission_denied`: Ok is an access answer, not a mutation guarantee
    (WP12E3-C001)."""
    (tmp / "drw").mkdir()
    fs = fs_cls(cwd=str(tmp))
    with _posix_mode(tmp / "drw", 0o666):
        result = await fs.check_read_write("drw")
        create = await fs.write_file("drw/child", "x")
    _assert_ok(result)
    assert isinstance(create, Err) and create.error.code == FsErrorCode.PERMISSION_DENIED


async def _witness_readonly_directory(fs_cls: Provider, tmp: Path) -> None:
    """READ-ONLY DIRECTORY (POSIX 0555) / DENY-WRITE DIRECTORY (Windows deny add-file) ->
    `permission_denied`."""
    (tmp / "d").mkdir()
    fs = fs_cls(cwd=str(tmp))
    with _not_writable(tmp / "d"):
        result = await fs.check_read_write("d")
    _assert_err(result, FsErrorCode.PERMISSION_DENIED)


async def _witness_windows_directory_denying_delete_child(fs_cls: Provider, tmp: Path) -> None:
    """WINDOWS DIRECTORY GRANTING LIST + ADD-FILE BUT DENYING DELETE-CHILD -> `Ok(None)`: the
    probe checks exactly list + add-file (WP12E3-C001)."""
    (tmp / "d").mkdir()
    fs = fs_cls(cwd=str(tmp))
    with _windows_deny(tmp / "d", "DC"):
        result = await fs.check_read_write("d")
    _assert_ok(result)


async def _witness_windows_readonly_attribute_directory(fs_cls: Provider, tmp: Path) -> None:
    """WINDOWS DIRECTORY WITH THE READ-ONLY ATTRIBUTE (permissive ACL) -> `Ok(None)`: the
    attribute does not deny adding entries to a directory."""
    (tmp / "d").mkdir()
    fs = fs_cls(cwd=str(tmp))
    with _windows_readonly_attribute(tmp / "d"):
        result = await fs.check_read_write("d")
    _assert_ok(result)


async def _witness_unsearchable_parent(fs_cls: Provider, tmp: Path) -> None:
    """UNSEARCHABLE PARENT (POSIX): `dr/inner` where `dr` is 0444 -> `permission_denied`."""
    (tmp / "dr").mkdir()
    (tmp / "dr" / "inner").write_text("x")
    fs = fs_cls(cwd=str(tmp))
    with _posix_mode(tmp / "dr", 0o444):
        result = await fs.check_read_write("dr/inner")
    _assert_err(result, FsErrorCode.PERMISSION_DENIED)


async def _witness_symlink_loop(fs_cls: Provider, tmp: Path) -> None:
    """SYMLINK LOOP -> the section 2.1 mapping of the host error: `ELOOP -> unknown` on POSIX;
    on Windows the same classification the unchanged Layer-12 operations give the loop."""
    os.symlink(tmp / "b", tmp / "a")
    os.symlink(tmp / "a", tmp / "b")
    fs = fs_cls(cwd=str(tmp))
    # L12-D007: Win32 1921 -> libuv ELOOP -> `unknown`, as on POSIX (formerly `invalid` on Windows).
    expected = FsErrorCode.UNKNOWN
    _assert_err(await fs.check_read_write("a"), expected)


async def _witness_fifo_no_blocking(fs_cls: Provider, tmp: Path) -> None:
    """NO CONTENT CONSUMED, NO BLOCKING: a 0666 FIFO with no reader and no writer -> `Ok(None)`
    promptly. If an implementation blocks (it opened the FIFO), both ends are attached so the
    stuck worker thread is released before the witness fails."""
    fifo = tmp / "fifo"
    os.mkfifo(fifo, 0o666)
    fs = fs_cls(cwd=str(tmp))
    task = asyncio.ensure_future(fs.check_read_write("fifo"))
    done, _ = await asyncio.wait({task}, timeout=5.0)
    if not done:
        reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
        os.close(reader)
        with contextlib.suppress(Exception):
            await task
        raise AssertionError("check_read_write blocked on a FIFO")
    _assert_ok(task.result())


async def _witness_relative_path_resolution(fs_cls: Provider, tmp: Path) -> None:
    """RELATIVE PATH RESOLUTION: with cwd `<tmp>/workspace`, `check_read_write("sub/f")` answers
    for `<tmp>/workspace/sub/f` (section 3.2); a failure reports the resolved path."""
    workspace = tmp / "workspace"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "sub" / "f").write_text("x")
    fs = fs_cls(cwd=str(workspace))
    _assert_ok(await fs.check_read_write("sub/f"))
    _assert_ok(await fs.check_read_write("sub/../sub/f"))
    missing = await fs.check_read_write("sub/g")
    _assert_err(missing, FsErrorCode.NOT_FOUND)
    assert isinstance(missing, Err)
    assert missing.error.path == resolve_local_path(str(workspace), "sub/g")


_NUL = chr(0)


async def _witness_embedded_nul_path_is_rejected(fs_cls: Provider, tmp: Path) -> None:
    """EMBEDDED NUL: rejected before any native call as `unknown`, as `check_readable` rejects it
    (WP12E2-I002); a native C-string call would see only the accessible prefix and answer `Ok`."""
    (tmp / "prefix").write_text("x")
    fs = fs_cls(cwd=str(tmp))
    for path in ("prefix" + _NUL + "missing", "prefix" + _NUL, str(tmp / "prefix") + _NUL + "x"):
        result = await fs.check_read_write(path)
        _assert_err(result, FsErrorCode.UNKNOWN)
        assert isinstance(result, Err)
        assert result.error.path == resolve_local_path(str(tmp), path)
    _assert_ok(await fs.check_read_write("prefix"))


async def _witness_pre_aborted_signal_not_inspected(fs_cls: Provider, tmp: Path) -> None:
    """SIGNAL ACCEPTED, NOT INSPECTED: a pre-aborted signal on an accessible file -> `Ok(None)`."""
    (tmp / "f").write_text("x")
    controller = RunAbortController()
    controller.abort()
    fs = fs_cls(cwd=str(tmp))
    _assert_ok(await fs.check_read_write("f", signal=controller.signal))
    _assert_ok(await fs.check_read_write("f", controller.signal))


async def _witness_provider_without_extension(fs_cls: Provider, tmp: Path) -> None:
    """PROVIDER WITHOUT THE EXTENSION: `Err(not_supported)` for every path, including accessible
    ones and directories; never a silent success, never a fallback inside the provider (a
    fallback would answer `not_found` for the missing path)."""
    (tmp / "f").write_text("x")
    (tmp / "d").mkdir()
    fs = fs_cls(cwd=str(tmp))
    for path in ("f", "d", "missing"):
        _assert_err(await fs.check_read_write(path), FsErrorCode.NOT_SUPPORTED)


class _ProviderWithoutExec009(LocalFileSystem):
    """A provider that cannot supply `EXEC-009` answers the capability question (section 13.2).
    `LocalFileSystem` itself always implements it; this stand-in only holds the capability answer
    to the contract."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        return Err(FsError(FsErrorCode.NOT_SUPPORTED, "check_read_write is not supported", path))


# ---------------------------------------------------------------------------
# Positive witnesses
# ---------------------------------------------------------------------------


async def test_rw_regular_file_directly_and_through_symlink(tmp_path: Path) -> None:
    await _witness_rw_file_direct_and_through_symlink(LocalFileSystem, tmp_path)


@unprivileged
async def test_readable_but_not_writable_file_is_permission_denied(tmp_path: Path) -> None:
    await _witness_readable_not_writable(LocalFileSystem, tmp_path)


@windows_only
async def test_windows_readonly_attribute_file_is_permission_denied(tmp_path: Path) -> None:
    await _witness_windows_readonly_attribute_file(LocalFileSystem, tmp_path)


@unprivileged
async def test_writable_but_not_readable_file_is_permission_denied(tmp_path: Path) -> None:
    await _witness_writable_not_readable(LocalFileSystem, tmp_path)


async def test_dangling_symlink_is_not_found(tmp_path: Path) -> None:
    await _witness_dangling_symlink(LocalFileSystem, tmp_path)


async def test_missing_path_and_non_directory_component(tmp_path: Path) -> None:
    await _witness_missing_path_and_non_directory_component(LocalFileSystem, tmp_path)


async def test_rw_directory_is_ok(tmp_path: Path) -> None:
    await _witness_rw_directory(LocalFileSystem, tmp_path)


@posix_only
@unprivileged
async def test_posix_rw_but_not_search_directory_is_ok_yet_refuses_creation(
    tmp_path: Path,
) -> None:
    await _witness_rw_but_not_search_directory(LocalFileSystem, tmp_path)


@unprivileged
async def test_readonly_or_deny_write_directory_is_permission_denied(tmp_path: Path) -> None:
    await _witness_readonly_directory(LocalFileSystem, tmp_path)


@windows_only
async def test_windows_directory_denying_delete_child_is_ok(tmp_path: Path) -> None:
    await _witness_windows_directory_denying_delete_child(LocalFileSystem, tmp_path)


@windows_only
async def test_windows_readonly_attribute_directory_is_ok(tmp_path: Path) -> None:
    await _witness_windows_readonly_attribute_directory(LocalFileSystem, tmp_path)


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
    """The Windows loop classification is the same host-error mapping the certified
    `check_readable`/`read_binary_file`/`canonical_path` apply to the same loop."""
    os.symlink(tmp_path / "b", tmp_path / "a")
    os.symlink(tmp_path / "a", tmp_path / "b")
    fs = LocalFileSystem(cwd=str(tmp_path))
    checks = [await fs.check_read_write("a"), await fs.check_readable("a"),
              await fs.read_binary_file("a"), await fs.canonical_path("a")]  # fmt: skip
    assert all(isinstance(r, Err) for r in checks)
    assert len({r.error.code for r in checks if isinstance(r, Err)}) == 1


@windows_only
async def test_windows_sharing_violation_matches_the_bindings_other_operations(
    tmp_path: Path,
) -> None:
    """SHARING VIOLATION (section 13.6, WP12E3-R001): with the file held open by another handle
    with no sharing, `check_read_write` gives exactly the classification this binding's shared
    mapper gives `check_readable` and `read_binary_file` for the same held file -- no
    operation-specific special case. The cross-language value is minion-agent#69's."""
    import ctypes
    from ctypes import wintypes

    target = tmp_path / "held.txt"
    target.write_text("x")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    dword = wintypes.DWORD
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, dword, dword, wintypes.LPVOID, dword, dword, wintypes.HANDLE,
    ]  # fmt: skip
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateFileW(str(target), 0xC0000000, 0, None, 3, 0, None)
    assert handle != wintypes.HANDLE(-1).value
    fs = LocalFileSystem(cwd=str(tmp_path))
    try:
        results = [await fs.check_read_write("held.txt"), await fs.check_readable("held.txt"),
                   await fs.read_binary_file("held.txt")]  # fmt: skip
    finally:
        kernel32.CloseHandle(handle)
    assert all(isinstance(r, Err) for r in results)
    assert len({r.error.code for r in results if isinstance(r, Err)}) == 1


@posix_only
async def test_fifo_is_ok_and_does_not_block(tmp_path: Path) -> None:
    await _witness_fifo_no_blocking(LocalFileSystem, tmp_path)


async def test_relative_path_resolution(tmp_path: Path) -> None:
    await _witness_relative_path_resolution(LocalFileSystem, tmp_path)


async def test_embedded_nul_path_is_rejected_as_unknown(tmp_path: Path) -> None:
    await _witness_embedded_nul_path_is_rejected(LocalFileSystem, tmp_path)


@pytest.mark.parametrize("host", ["nt", "posix"])
def test_embedded_nul_is_rejected_before_either_native_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    """Direct witness on the sync helper for both host branches: the NUL guard runs before the
    Windows `CreateFileW` path and before libc `access(2)` -- neither is called."""
    (tmp_path / "prefix").write_text("x")
    native_calls: list[str] = []
    monkeypatch.setattr(os, "name", host)
    monkeypatch.setattr(filesystem_module, "_check_read_write_windows", native_calls.append)
    monkeypatch.setattr(filesystem_module, "_check_read_write_posix", native_calls.append)
    with pytest.raises(ValueError, match="embedded null character"):
        filesystem_module._check_read_write_sync(str(tmp_path / "prefix") + _NUL + "missing")
    assert native_calls == []
    filesystem_module._check_read_write_sync(str(tmp_path / "prefix"))
    assert native_calls == [str(tmp_path / "prefix")]


async def test_pre_aborted_signal_is_accepted_but_not_inspected(tmp_path: Path) -> None:
    await _witness_pre_aborted_signal_not_inspected(LocalFileSystem, tmp_path)


async def test_provider_without_exec009_answers_not_supported(tmp_path: Path) -> None:
    await _witness_provider_without_extension(_ProviderWithoutExec009, tmp_path)


async def test_local_filesystem_never_answers_not_supported(tmp_path: Path) -> None:
    """Section 13.5: the certified first-party provider MUST implement `check_read_write`."""
    (tmp_path / "f").write_text("x")
    fs = LocalFileSystem(cwd=str(tmp_path))
    for path in ("f", ".", "missing"):
        result = await fs.check_read_write(path)
        assert not (isinstance(result, Err) and result.error.code == FsErrorCode.NOT_SUPPORTED)


async def test_check_read_write_leaves_content_and_existing_operations_unchanged(
    tmp_path: Path,
) -> None:
    """EXISTING OPERATIONS UNCHANGED (regression, alongside the unchanged section 10/11.6/12.6
    suites): checking a file, a directory and a symlink changes neither their content nor what the
    certified operations -- `check_readable` included -- report for them afterwards."""
    fs = LocalFileSystem(cwd=str(tmp_path))
    await fs.write_file("f.txt", "content")
    await fs.create_dir("d")
    os.symlink(tmp_path / "f.txt", tmp_path / "link")

    async def observe() -> list[object]:
        return [await fs.file_info("f.txt"), await fs.file_info("d"), await fs.file_info("link"),
                await fs.canonical_path("link"), await fs.read_binary_file("f.txt"),
                await fs.list_dir("."), await fs.check_readable("f.txt")]  # fmt: skip

    before = await observe()
    for path in ("f.txt", "d", "link"):
        _assert_ok(await fs.check_read_write(path))
    assert await observe() == before
    assert (tmp_path / "f.txt").read_text() == "content"


# ---------------------------------------------------------------------------
# Host branches exercised on any host (ONE combined decision; errno kept)
# ---------------------------------------------------------------------------


def _fake_libc_access(
    monkeypatch: pytest.MonkeyPatch, errno_value: int | None
) -> list[tuple[bytes, int]]:
    """Route the POSIX branch on any host through a stand-in for libc `access(2)`."""
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


async def test_posix_branch_is_one_access_r_ok_w_ok_on_the_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONE COMBINED OPERATION (POSIX): exactly one `access(resolved, R_OK | W_OK)` -- no
    `check_readable` call, no second probe, no preliminary metadata call."""
    calls = _fake_libc_access(monkeypatch, None)
    stat_calls: list[object] = []
    real_stat = os.stat
    monkeypatch.setattr(
        filesystem_module.os, "stat", lambda *a, **k: stat_calls.append(a) or real_stat(*a, **k)
    )
    _assert_ok(await LocalFileSystem(cwd=str(tmp_path)).check_read_write("sub/f"))
    assert calls == [(os.fsencode(str(tmp_path / "sub" / "f")), os.R_OK | os.W_OK)]
    assert stat_calls == []


def test_windows_branch_is_one_open_for_read_and_write_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONE COMBINED OPERATION (Windows): exactly one open asking for
    `FILE_READ_DATA | FILE_WRITE_DATA` (list + add-file on a directory) -- never
    `FILE_DELETE_CHILD`, never a second probe."""
    probes: list[tuple[str, int]] = []
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(
        filesystem_module, "_windows_open_probe", lambda path, access: probes.append((path, access))
    )
    filesystem_module._check_read_write_sync(str(tmp_path / "f"))
    assert probes == [(str(tmp_path / "f"), 0x0001 | 0x0002)]
    assert not probes[0][1] & 0x0040  # FILE_DELETE_CHILD


@pytest.mark.parametrize(
    ("errno_value", "code"),
    [
        (errno.ENOENT, FsErrorCode.NOT_FOUND),
        (errno.ENOTDIR, FsErrorCode.NOT_DIRECTORY),
        (errno.EACCES, FsErrorCode.PERMISSION_DENIED),
        (errno.EPERM, FsErrorCode.PERMISSION_DENIED),
        (errno.EROFS, FsErrorCode.UNKNOWN),
        (errno.ETXTBSY, FsErrorCode.UNKNOWN),
        (errno.ELOOP, FsErrorCode.UNKNOWN),
        (errno.EIO, FsErrorCode.UNKNOWN),
        (errno.EINVAL, FsErrorCode.INVALID),
    ],
)
async def test_posix_branch_keeps_the_access_calls_own_errno(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, errno_value: int, code: FsErrorCode
) -> None:
    """The failure reason is `access(2)`'s own errno classified by section 2.1 -- including
    `EROFS` (read-only filesystem) and `ETXTBSY`, both `unknown` -- never a fabricated one."""
    _fake_libc_access(monkeypatch, errno_value)
    result = await LocalFileSystem(cwd=str(tmp_path)).check_read_write("f")
    _assert_err(result, code)
    assert isinstance(result, Err)
    assert isinstance(result.error.cause, OSError)
    assert result.error.cause.errno == errno_value
    assert result.error.path == str(tmp_path / "f")


# ---------------------------------------------------------------------------
# Negative controls: each mutant MUST fail at least one witness above
# ---------------------------------------------------------------------------


class _ReadabilityOnly(LocalFileSystem):
    """Mutant: `check_read_write` answered by `check_readable` (EXEC-008) alone."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        return await self.check_readable(path, signal)


class _WritabilityOnly(LocalFileSystem):
    """Mutant: only the write half of the predicate (POSIX `access(W_OK)`, Windows write-data)."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)

        def write_only() -> None:
            if _WINDOWS:
                filesystem_module._windows_open_probe(resolved, filesystem_module._FILE_WRITE_DATA)
            elif not os.access(resolved, os.W_OK):
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), resolved)

        try:
            await asyncio.to_thread(write_only)
        except OSError as exc:
            return Err(filesystem_module.to_pi_fs_error(exc, resolved))
        return Ok(None)


class _TwoProbes(LocalFileSystem):
    """Mutant: `check_readable` then a separate write probe -- two host decisions."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        readable = await self.check_readable(path, signal)
        if isinstance(readable, Err):
            return readable
        return await super().check_read_write(path, signal)


class _TruncatingOpen(LocalFileSystem):
    """Mutant: proves writability by opening for writing with truncation."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)

        def truncate_open() -> None:
            with open(resolved, "wb"):
                pass

        try:
            await asyncio.to_thread(truncate_open)
        except OSError as exc:
            return Err(filesystem_module.to_pi_fs_error(exc, resolved))
        return Ok(None)


class _CreateEntryToProve(LocalFileSystem):
    """Mutant: proves directory writability by creating (and removing) an entry in it."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        result = await super().check_read_write(path, signal)
        resolved = resolve_local_path(self.cwd, path)
        if isinstance(result, Ok) and os.path.isdir(resolved):
            probe = os.path.join(resolved, ".probe")
            try:
                await asyncio.to_thread(Path(probe).write_text, "")
                await asyncio.to_thread(os.remove, probe)
            except OSError as exc:
                return Err(filesystem_module.to_pi_fs_error(exc, resolved))
        return result


class _RequestsDeleteChild(LocalFileSystem):
    """Mutant (Windows): also requests `FILE_DELETE_CHILD`, overclaiming removal (WP12E3-C001)."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        access = filesystem_module._FILE_READ_DATA | filesystem_module._FILE_WRITE_DATA | 0x0040
        try:
            await asyncio.to_thread(filesystem_module._windows_open_probe, resolved, access)
        except OSError as exc:
            return Err(filesystem_module.to_pi_fs_error(exc, resolved))
        return Ok(None)


class _NodeWindowsAttributeOnlyAccess(LocalFileSystem):
    """Mutant: libuv's Windows `fs__access` for `W_OK` -- `GetFileAttributesW` on the unfollowed
    final component, failing only for a non-directory with the read-only attribute."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        import ctypes
        from ctypes import wintypes

        resolved = resolve_local_path(self.cwd, path)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetFileAttributesW.restype = wintypes.DWORD
        kernel32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
        attr = kernel32.GetFileAttributesW(resolved)
        if attr == 0xFFFFFFFF:
            return Err(FsError(FsErrorCode.NOT_FOUND, "not found", resolved))
        if attr & 0x1 and not attr & 0x10:
            return Err(FsError(FsErrorCode.PERMISSION_DENIED, "read-only", resolved))
        return Ok(None)


class _FinalSymlinkNotFollowed(LocalFileSystem):
    """Mutant: a final-component symlink is answered for the link itself."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        if os.path.islink(resolved):
            return Ok(None)
        return await super().check_read_write(path, signal)


class _OpenForReadWrite(LocalFileSystem):
    """Mutant (POSIX): proves access by actually opening read+write, which blocks on a FIFO."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)

        def open_rw() -> None:
            with open(resolved, "rb"):
                pass

        try:
            await asyncio.to_thread(open_rw)
        except OSError as exc:
            return Err(filesystem_module.to_pi_fs_error(exc, resolved))
        return Ok(None)


class _NativeCallWithoutNulGuard(LocalFileSystem):
    """Mutant: the per-host native check reached without the embedded-NUL guard."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        resolved = resolve_local_path(self.cwd, path)
        native = (
            filesystem_module._check_read_write_windows
            if _WINDOWS
            else filesystem_module._check_read_write_posix
        )
        try:
            await asyncio.to_thread(native, resolved)
        except OSError as exc:
            return Err(filesystem_module.to_pi_fs_error(exc, resolved))
        return Ok(None)


class _PreAbortRejecting(LocalFileSystem):
    """Mutant: inspects the signal and rejects a pre-aborted one."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        if signal is not None and signal.aborted:
            return Err(FsError(FsErrorCode.ABORTED, "aborted", path))
        return await super().check_read_write(path, signal)


class _SilentSuccessWithoutExec009(LocalFileSystem):
    """Mutant: a provider lacking the extension that silently answers `Ok`."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        return Ok(None)


class _ReadableFallbackWithoutExec009(LocalFileSystem):
    """Mutant: a provider lacking the extension that falls back to `check_readable` itself."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        return await self.check_readable(path, signal)


class _ExistenceOnly(LocalFileSystem):
    """Mutant: accessibility == `file_info` (lstat) succeeded."""

    async def check_read_write(
        self, path: str, signal: RunSignal | None = None
    ) -> Result[None, FsError]:
        info = await self.file_info(path)
        return Ok(None) if isinstance(info, Ok) else Err(info.error)


_NEGATIVE_CONTROLS: list[tuple[str, Provider, Witness, list[pytest.MarkDecorator]]] = [
    ("readability-only/readable-not-writable", _ReadabilityOnly,
     _witness_readable_not_writable, [unprivileged]),
    ("readability-only/readonly-directory", _ReadabilityOnly,
     _witness_readonly_directory, [unprivileged]),
    ("readable-fallback-inside-provider/readable-not-writable", _ReadableFallbackWithoutExec009,
     _witness_readable_not_writable, [unprivileged]),
    ("writability-only/writable-not-readable", _WritabilityOnly,
     _witness_writable_not_readable, [unprivileged]),
    ("truncating-open/content-unchanged", _TruncatingOpen,
     _witness_rw_file_direct_and_through_symlink, []),
    ("create-entry-to-prove/rw-but-not-search-directory", _CreateEntryToProve,
     _witness_rw_but_not_search_directory, [posix_only, unprivileged]),
    ("requests-delete-child/deny-delete-child-directory", _RequestsDeleteChild,
     _witness_windows_directory_denying_delete_child, [windows_only]),
    ("node-windows-attributes/deny-write-acl", _NodeWindowsAttributeOnlyAccess,
     _witness_readable_not_writable, [windows_only]),
    ("node-windows-attributes/dangling-link", _NodeWindowsAttributeOnlyAccess,
     _witness_dangling_symlink, [windows_only]),
    ("final-symlink-not-followed/dangling-link", _FinalSymlinkNotFollowed,
     _witness_dangling_symlink, []),
    ("existence-only/readable-not-writable", _ExistenceOnly,
     _witness_readable_not_writable, [unprivileged]),
    ("existence-only/dangling-link", _ExistenceOnly, _witness_dangling_symlink, []),
    ("open-for-read-write/fifo", _OpenForReadWrite, _witness_fifo_no_blocking, [posix_only]),
    ("native-call-without-nul-guard/embedded-nul", _NativeCallWithoutNulGuard,
     _witness_embedded_nul_path_is_rejected, []),
    ("pre-abort-rejection/signal-not-inspected", _PreAbortRejecting,
     _witness_pre_aborted_signal_not_inspected, []),
    ("silent-success-without-exec009/provider-capability", _SilentSuccessWithoutExec009,
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
    """The named witness passes for `LocalFileSystem` (above) and MUST fail for the mutant."""
    with pytest.raises(AssertionError):
        await witness(mutant, tmp_path)


def test_two_probe_mutant_makes_two_host_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONE COMBINED OPERATION, negative control: the two-probe mutant reaches the host twice
    (`R_OK`, then `R_OK | W_OK`), where the implementation reaches it once."""
    calls = _fake_libc_access(monkeypatch, None)
    asyncio.run(_TwoProbes(cwd=str(tmp_path)).check_read_write("f"))
    assert len(calls) == 2
