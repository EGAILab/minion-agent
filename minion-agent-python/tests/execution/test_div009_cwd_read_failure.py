"""`DIV-009` (spec/execution.md section 22; minion-agent-docs assurance/pi-divergences.md;
checkpoint `assurance/layers/12-l12-d013-convergence-ce01.md`, agreed in review 6): a failed native
process-cwd read during path resolution is a typed failure, not an escaping exception.

Every failure is INJECTED at the native read boundary (`_native_process_cwd_view`); no real
working directory is removed or invalidated. Read rows come from the checkpoint's characterized
read-domain matrix (section 2): every NP cell class reads exactly once, every no-read cell class
must not read. The inherited `=X:` entry is injected too."""

from __future__ import annotations

import errno
import os
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import filesystem
from minion_agent.execution import shell as shell_module
from minion_agent.execution import subprocess as subprocess_module
from minion_agent.execution.errors import (
    FsError,
    FsErrorCode,
    ShellErrorCode,
    SubprocessErrorCode,
)
from minion_agent.execution.filesystem import LocalFileSystem, ProcessCwdReadError
from minion_agent.execution.result import Err, Ok
from minion_agent.execution.shell import LocalShell
from minion_agent.execution.subprocess import LocalSubprocess, SpawnOptions
from minion_agent.runtime.signal import RunAbortController, RunSignal

WIN = pytest.mark.skipif(sys.platform != "win32", reason="Python reads the cwd only on win32")
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="the POSIX no-read witness")
PROC = "E:\\proc"


class Reader:
    """A scripted native process-cwd read: each call takes the next step (a value, or an
    exception to raise); past the script every call FAILS THE TEST (not an `OSError`)."""

    def __init__(self, *script: str | BaseException) -> None:
        self.script = list(script)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if not self.script:
            raise AssertionError("a native process-cwd read where none is allowed")
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


ENTRY_STATES: dict[str, dict[str, str]] = {
    "absent": {},
    "empty": {"A:": "", "C:": ""},
    "same": {"A:": "A:\\deep", "C:": "C:\\deep"},
    "other": {"A:": "X:\\other", "C:": "X:\\other"},
}


def install(mp: pytest.MonkeyPatch, reader: Reader, entries: str = "absent") -> Reader:
    state = ENTRY_STATES[entries]
    mp.setattr(filesystem, "_native_process_cwd_view", reader)
    mp.setattr(filesystem, "_native_inherited_view", lambda device: state.get(device.upper()))
    return reader


ERRORS: dict[str, tuple[Callable[[], OSError], FsErrorCode]] = {
    "enoent": (lambda: FileNotFoundError(errno.ENOENT, "process cwd gone"), FsErrorCode.NOT_FOUND),
    "denied": (
        lambda: PermissionError(errno.EACCES, "process cwd denied"),
        FsErrorCode.PERMISSION_DENIED,
    ),
    "win87": (lambda: OSError(errno.EINVAL, "bad parameter", None, 87), FsErrorCode.INVALID),
    "eio": (lambda: OSError(errno.EIO, "device error"), FsErrorCode.UNKNOWN),
}

# The checkpoint's section 2 classes (win32). NP: Node and Python read; the rest read in neither.
ROOTED = ["\\leaf", "/leaf", "\\\\leaf", "\\\\srv\\"]
ALL_PROVIDERS = [
    "C:\\base",
    "\\\\srv\\share\\base",
    "\\\\?\\C:\\base",
    "\\base",
    "C:base",
    "base",
    "",
]
READING = (
    [(p, i, s) for p in ALL_PROVIDERS for i in ROOTED for s in ("absent", "other")]
    + [
        (p, "a:leaf", s)
        for p in ["C:\\base", "\\\\srv\\share\\base", "\\\\?\\C:\\base", "C:base", "base", ""]
        for s in ("absent", "empty")
    ]
    + [
        (p, "c:leaf", s)
        for p in ["\\\\srv\\share\\base", "\\\\?\\C:\\base", "base", ""]
        for s in ("absent", "empty")
    ]
)
NOT_READING = (
    [
        (p, i, "absent")
        for p in ALL_PROVIDERS
        for i in ["C:\\x", "\\\\srv\\share\\x", "file:///C:/x", "file://srv/share/x"]
    ]
    + [
        (p, i, s)
        for p in ["C:\\base", "\\\\srv\\share\\base", "\\\\?\\C:\\base"]
        for i in ["leaf", ""]
        for s in ENTRY_STATES
    ]
    + [
        (p, "a:leaf", s)
        for p in ["C:\\base", "\\\\srv\\share\\base", "C:base", "base", ""]
        for s in ("same", "other")
    ]
    + [("C:\\base", "c:leaf", s) for s in ENTRY_STATES]
    + [("\\base", i, s) for i in ["a:leaf", "c:leaf"] for s in ENTRY_STATES]
)


def aborted() -> RunSignal:
    controller = RunAbortController()
    controller.abort()
    return controller.signal


def expect_div009(result: Any, native: OSError, code: FsErrorCode, argument: str) -> bool:
    if not isinstance(result, Err) or not isinstance(result.error, FsError):
        return False
    err = result.error
    return (
        err.code == code
        and err.message == str(native)
        and err.path == argument
        and err.cause is native
    )


# --- read rows and no-read rows (checkpoint sections 2 and 5) ------------------------------------


@WIN
@pytest.mark.parametrize("kind", sorted(ERRORS))
@pytest.mark.parametrize(("provider", "source", "entries"), READING)
async def test_reading_row_returns_the_typed_failure(
    provider: str, source: str, entries: str, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    make, code = ERRORS[kind]
    native = make()
    reader = install(monkeypatch, Reader(native), entries)
    result = await LocalFileSystem(provider).absolute_path(source)
    assert expect_div009(result, native, code, source)
    assert reader.calls == 1


@WIN
@pytest.mark.parametrize(("provider", "source", "entries"), NOT_READING)
async def test_no_read_row_never_reads(
    provider: str, source: str, entries: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = install(monkeypatch, Reader(), entries)
    assert isinstance(await LocalFileSystem(provider).absolute_path(source), Ok)
    assert reader.calls == 0


@POSIX
@pytest.mark.parametrize("provider", ["/base", "base", "", "//base"])
@pytest.mark.parametrize("source", ["/x", "//x", "x", "", "\\leaf", "file:///x"])
def test_python_posix_never_reads(
    provider: str, source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merged Python's POSIX branch performs no process-cwd read (checkpoint section 2; its
    relative-provider success path is recorded, not certified, spec section 22.5)."""
    reader = install(monkeypatch, Reader())
    filesystem.resolve_local_path(provider, source)
    assert reader.calls == 0


@WIN
@pytest.mark.parametrize("source", ["\\le\ud800af", "file://srv/", "/leaf\\"])
async def test_caller_path_is_the_raw_argument(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`path` is the argument exactly as passed: a lone surrogate stays raw, a `file://` spelling
    is not converted, a forward slash is not normalized."""
    native = FileNotFoundError(errno.ENOENT, "gone")
    install(monkeypatch, Reader(native))
    result = await LocalFileSystem("C:\\base").absolute_path(source)
    assert expect_div009(result, native, FsErrorCode.NOT_FOUND, source)


# --- every resolver consumer through the real seam, with no further filesystem call ------------


Call = Callable[[LocalFileSystem, str, RunSignal | None], Coroutine[Any, Any, Any]]
CONSUMERS: dict[str, Call] = {
    "absolute_path": lambda fs, p, s: fs.absolute_path(p, s),
    "canonical_path": lambda fs, p, s: fs.canonical_path(p, s),
    "file_info": lambda fs, p, s: fs.file_info(p, s),
    "read_text_file": lambda fs, p, s: fs.read_text_file(p, s),
    "read_text_lines": lambda fs, p, s: fs.read_text_lines(p, None, s),
    "read_binary_file": lambda fs, p, s: fs.read_binary_file(p, s),
    "write_file": lambda fs, p, s: fs.write_file(p, "x", s),
    "append_file": lambda fs, p, s: fs.append_file(p, "x", s),
    "rename_file": lambda fs, p, s: fs.rename_file(p, "E:\\abs\\dst", s),
    "list_dir": lambda fs, p, s: fs.list_dir(p, s),
    "list_dir_raw": lambda fs, p, s: fs.list_dir_raw(p, s),
    "probe_dir_entry": lambda fs, p, s: fs.probe_dir_entry(p, s),
    "check_readable": lambda fs, p, s: fs.check_readable(p, s),
    "check_read_write": lambda fs, p, s: fs.check_read_write(p, s),
    "create_dir": lambda fs, p, s: fs.create_dir(p, True, s),
    "remove": lambda fs, p, s: fs.remove(p, False, False, s),
    "exists": lambda fs, p, s: fs.exists(p, s),
}
ABORT_CHECKING = ["read_text_file", "read_text_lines", "read_binary_file", "write_file", "list_dir"]
ABORT_CHECKING += ["list_dir_raw", "rename_file"]


def no_native_calls(mp: pytest.MonkeyPatch) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("a native filesystem call after a failed process-cwd read")

    mp.setattr(filesystem.asyncio, "to_thread", forbidden)


@WIN
@pytest.mark.parametrize("name", sorted(CONSUMERS))
@pytest.mark.parametrize("signal_state", ["none", "aborted"])
async def test_every_consumer_returns_the_typed_failure(
    name: str, signal_state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolution precedes the pre-abort check (Pi's order): the read failure wins even under an
    aborted signal, and nothing native follows it."""
    if signal_state == "aborted" and name not in ABORT_CHECKING:
        pytest.skip("the operation does not inspect its signal")
    native = FileNotFoundError(errno.ENOENT, "gone")
    reader = install(monkeypatch, Reader(native))
    no_native_calls(monkeypatch)
    signal = aborted() if signal_state == "aborted" else None
    result = await CONSUMERS[name](LocalFileSystem("C:\\base"), "\\leaf", signal)
    assert expect_div009(result, native, FsErrorCode.NOT_FOUND, "\\leaf")
    assert reader.calls == 1


@WIN
async def test_exception_other_than_an_os_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, Reader(TypeError("a provider bug")))
    with pytest.raises(TypeError, match="a provider bug"):
        await LocalFileSystem("C:\\base").absolute_path("\\leaf")


# --- derived operations paired with native failures (checkpoint section 3) -----------------------


async def exists_propagates(mp: pytest.MonkeyPatch) -> bool:
    native = FileNotFoundError(errno.ENOENT, "gone")
    reader = install(mp, Reader(native))
    result = await LocalFileSystem("C:\\base").exists("\\leaf")
    return expect_div009(result, native, FsErrorCode.NOT_FOUND, "\\leaf") and reader.calls == 1


@WIN
async def test_exists_propagates_the_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await exists_propagates(monkeypatch)


async def test_exists_native_absence_is_still_false(tmp_path: Path) -> None:
    assert await LocalFileSystem(str(tmp_path)).exists(str(tmp_path / "absent")) == Ok(False)


async def resolve_propagates(mp: pytest.MonkeyPatch) -> bool:
    """The first read fails; a forbidden fallback would read again and succeed."""
    native = FileNotFoundError(errno.ENOENT, "gone")
    reader = install(mp, Reader(native, PROC))
    result = await LocalFileSystem("C:\\base").resolve("\\leaf")
    return expect_div009(result, native, FsErrorCode.NOT_FOUND, "\\leaf") and reader.calls == 1


@WIN
async def test_resolve_propagates_without_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await resolve_propagates(monkeypatch)


async def test_resolve_native_absence_still_falls_back(tmp_path: Path) -> None:
    target = str(tmp_path / "absent")
    result = await LocalFileSystem(str(tmp_path)).resolve(target)
    assert isinstance(result, Ok)
    assert result.value.target_key == os.path.normpath(target)


# --- rename_file: the Owner amendment's rows R1-R10 (checkpoint section 4) ------------------------


async def rename_row(
    mp: pytest.MonkeyPatch,
    source: str,
    destination: str,
    signal: RunSignal | None,
    *script: str | BaseException,
) -> tuple[Any, int]:
    reader = install(mp, Reader(*script))
    result = await LocalFileSystem("C:\\base").rename_file(source, destination, signal)
    return result, reader.calls


async def r1_source_failure_under_abort(mp: pytest.MonkeyPatch) -> bool:
    native = FileNotFoundError(errno.ENOENT, "gone")
    result, calls = await rename_row(mp, "\\src", "E:\\abs\\dst", aborted(), native)
    return expect_div009(result, native, FsErrorCode.NOT_FOUND, "\\src") and calls == 1


@WIN
async def test_r1_source_failure_under_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await r1_source_failure_under_abort(monkeypatch)


@WIN
@pytest.mark.parametrize("signal_state", ["none", "aborted"])
async def test_r2_r4_single_reading_argument(
    signal_state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    signal = aborted() if signal_state == "aborted" else None
    for source, destination, failing in [
        ("\\src", "E:\\abs\\dst", "\\src"),
        ("E:\\abs\\src", "\\dst", "\\dst"),
    ]:
        native = FileNotFoundError(errno.ENOENT, "gone")
        result, calls = await rename_row(monkeypatch, source, destination, signal, native)
        assert expect_div009(result, native, FsErrorCode.NOT_FOUND, failing)
        assert calls == 1


@WIN
async def test_r5_first_read_fails_destination_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    native = FileNotFoundError(errno.ENOENT, "gone")
    result, calls = await rename_row(monkeypatch, "\\src", "\\dst", None, native)
    assert expect_div009(result, native, FsErrorCode.NOT_FOUND, "\\src")
    assert calls == 1


@WIN
async def test_r6_second_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    native = PermissionError(errno.EACCES, "denied")
    result, calls = await rename_row(monkeypatch, "\\src", "\\dst", None, PROC, native)
    assert expect_div009(result, native, FsErrorCode.PERMISSION_DENIED, "\\dst")
    assert calls == 2


@WIN
async def test_r7_both_resolve_then_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = await rename_row(monkeypatch, "\\src", "\\dst", aborted(), PROC, PROC)
    assert isinstance(result, Err)
    assert (result.error.code, result.error.path) == (FsErrorCode.ABORTED, "E:\\dst")
    assert calls == 2


@WIN
async def test_r8_complete_roots_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = await rename_row(monkeypatch, "E:\\abs\\src", "\\\\srv\\share\\dst", aborted())
    assert isinstance(result, Err)
    assert (result.error.code, result.error.path) == (
        FsErrorCode.ABORTED,
        "\\\\srv\\share\\dst",
    )
    assert calls == 0


def rooted(path: Path) -> str:
    """The rooted (drive-less) spelling of a contained `tmp_path` path."""
    spelled = str(path)[2:]
    assert spelled.startswith("\\") and not spelled.startswith("\\\\"), spelled
    return spelled


@WIN
async def test_r_success_both_resolve_and_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The successful, un-aborted positive: both rooted arguments read the (injected) process
    cwd on `tmp_path`'s drive, and the rename happens inside `tmp_path`."""
    (tmp_path / "a.txt").write_text("x")
    cwd = str(tmp_path)[:2] + "\\proc"
    reader = install(monkeypatch, Reader(cwd, cwd))
    fs = LocalFileSystem("C:\\never-used")
    result = await fs.rename_file(rooted(tmp_path / "a.txt"), rooted(tmp_path / "b.txt"))
    assert result == Ok(None)
    assert (tmp_path / "b.txt").read_text() == "x" and not (tmp_path / "a.txt").exists()
    assert reader.calls == 2


@WIN
async def test_r9_native_rename_error_origin_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = install(monkeypatch, Reader())
    source = str(tmp_path / "absent")
    result = await LocalFileSystem(str(tmp_path)).rename_file(source, str(tmp_path / "dst"))
    assert isinstance(result, Err)
    assert (result.error.code, result.error.path) == (FsErrorCode.NOT_FOUND, source)
    assert reader.calls == 0


async def r10_nul_path_without_reread(mp: pytest.MonkeyPatch) -> bool:
    """A complete-absolute NUL source (no read), a rooted destination (one successful read), then
    the native NUL rejection. A forbidden handler re-read would be the NEXT read, and it fails."""
    native = FileNotFoundError(errno.ENOENT, "a forbidden re-read")
    result, calls = await rename_row(mp, "E:\\abs\\n\x00ul", "\\dst", None, PROC, native)
    return (
        isinstance(result, Err)
        and result.error.code == FsErrorCode.UNKNOWN
        and result.error.path == "E:\\abs\\n\x00ul"
        and calls == 1
    )


@WIN
async def test_r10_nul_path_without_reread(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await r10_nul_path_without_reread(monkeypatch)


@WIN
@pytest.mark.parametrize("name", ["write_file", "append_file"])
async def test_nul_after_parent_effect_keeps_it(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L12-D006 x DIV-009: the parent created before the NUL rejection stays; the error path makes
    no further read (the next one would fail)."""
    parent = tmp_path / "newparent"
    cwd = str(tmp_path)[:2] + "\\proc"
    reader = install(monkeypatch, Reader(cwd, FileNotFoundError(errno.ENOENT, "re-read")))
    target = rooted(parent) + "\\le\x00af"
    result = await CONSUMERS[name](LocalFileSystem("C:\\never-used"), target, None)
    assert isinstance(result, Err) and result.error.code == FsErrorCode.UNKNOWN
    assert parent.is_dir()
    assert reader.calls == 1


# --- ctx.shell.exec and ctx.subprocess.spawn (spec section 22.2-22.3) -----------------------------


async def shell_failure(mp: pytest.MonkeyPatch) -> bool:
    native = FileNotFoundError(errno.ENOENT, "gone")
    reader = install(mp, Reader(native))

    async def no_shell(_: Any) -> Any:
        raise AssertionError("shell discovery after a failed process-cwd read")

    mp.setattr(LocalShell, "_resolve_shell", no_shell)
    result = await LocalShell(cwd="C:\\base").exec("echo x", cwd="\\leaf")
    return (
        isinstance(result, Err)
        and result.error.code == ShellErrorCode.SPAWN_ERROR
        and result.error.message == str(native)
        and result.error.cause is native
        and reader.calls == 1
    )


@WIN
async def test_shell_exec_spawn_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await shell_failure(monkeypatch)


@WIN
async def test_shell_pre_abort_and_timeout_keep_their_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = install(monkeypatch, Reader())
    shell = LocalShell(cwd="C:\\base")
    first = await shell.exec("echo x", cwd="\\leaf", signal=aborted())
    second = await shell.exec("echo x", cwd="\\leaf", timeout=-1)
    assert isinstance(first, Err) and first.error.code == ShellErrorCode.ABORTED
    assert isinstance(second, Err) and second.error.code == ShellErrorCode.TIMEOUT
    assert reader.calls == 0


def spawn_spy(mp: pytest.MonkeyPatch) -> list[Any]:
    started: list[Any] = []

    async def forbidden(*args: Any, **_: Any) -> Any:
        started.append(args)
        raise AssertionError("a native process creation after a failed process-cwd read")

    mp.setattr(subprocess_module.asyncio, "create_subprocess_exec", forbidden)
    return started


async def subprocess_failure(mp: pytest.MonkeyPatch) -> bool:
    native = FileNotFoundError(errno.ENOENT, "gone")
    reader = install(mp, Reader(native))
    started = spawn_spy(mp)
    try:
        result = await LocalSubprocess(cwd="C:\\base").spawn(
            [sys.executable, "-c", "pass"], SpawnOptions(cwd="\\leaf")
        )
    except AssertionError:
        return False
    return (
        isinstance(result, Err)
        and result.error.code == SubprocessErrorCode.SPAWN_ERROR
        and result.error.message == str(native)
        and result.error.cause is native
        and not result.error.nul_input
        and started == []
        and reader.calls == 1
    )


@WIN
async def test_subprocess_spawn_error_without_a_process(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await subprocess_failure(monkeypatch)


@WIN
async def test_subprocess_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-abort first (no read); the section 21 NUL checks follow resolution."""
    started = spawn_spy(monkeypatch)
    reader = install(monkeypatch, Reader())
    sp = LocalSubprocess(cwd="C:\\base")
    pre = await sp.spawn([sys.executable], SpawnOptions(cwd="\\leaf", signal=aborted()))
    assert isinstance(pre, Err) and pre.error.code == SubprocessErrorCode.ABORTED
    assert reader.calls == 0
    native = FileNotFoundError(errno.ENOENT, "gone")
    reader = install(monkeypatch, Reader(native))
    nul_after = await sp.spawn(["a\x00b"], SpawnOptions(cwd="\\leaf"))
    assert isinstance(nul_after, Err) and nul_after.error.code == SubprocessErrorCode.SPAWN_ERROR
    assert not nul_after.error.nul_input and reader.calls == 1
    reader = install(monkeypatch, Reader())
    nul_only = await sp.spawn(["a\x00b"], SpawnOptions(cwd="E:\\abs"))
    assert isinstance(nul_only, Err) and nul_only.error.nul_input
    assert reader.calls == 0 and started == []


# --- negative controls K1-K12 (checkpoint section 6): green baseline, then the witness FAILS ----


def first_reading_row_holds(mp: pytest.MonkeyPatch) -> Callable[[], Coroutine[Any, Any, bool]]:
    async def witness() -> bool:
        native = FileNotFoundError(errno.ENOENT, "gone")
        reader = install(mp, Reader(native))
        try:
            result = await LocalFileSystem("C:\\base").absolute_path("\\leaf")
        except BaseException:
            return False
        return expect_div009(result, native, FsErrorCode.NOT_FOUND, "\\leaf") and reader.calls == 1

    return witness


def no_read_row_holds(mp: pytest.MonkeyPatch) -> Callable[[], Coroutine[Any, Any, bool]]:
    async def witness() -> bool:
        reader = install(mp, Reader())
        try:
            result = await LocalFileSystem("C:\\base").absolute_path("\\\\srv\\share\\x")
        except BaseException:
            return False
        return isinstance(result, Ok) and reader.calls == 0

    return witness


def type_error_propagates(mp: pytest.MonkeyPatch) -> Callable[[], Coroutine[Any, Any, bool]]:
    async def witness() -> bool:
        install(mp, Reader(TypeError("bug")))
        try:
            await LocalFileSystem("C:\\base").absolute_path("\\leaf")
        except TypeError:
            return True
        except BaseException:
            return False
        return False

    return witness


def wrap_resolution(mp: pytest.MonkeyPatch, wrong: Callable[[Any, str, str], str]) -> None:
    real = filesystem._resolve_local_path
    mp.setattr(filesystem, "_resolve_local_path", lambda cwd, path: wrong(real, cwd, path))


def k1_leak(mp: pytest.MonkeyPatch) -> None:
    def wrong(real: Any, cwd: str, path: str) -> str:
        try:
            return str(real(cwd, path))
        except ProcessCwdReadError as exc:
            raise exc.native from None

    wrap_resolution(mp, wrong)


def k2_fixed_code(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(
        filesystem,
        "to_pi_fs_error",
        lambda exc, path=None: FsError(FsErrorCode.UNKNOWN, str(exc), path, exc),
    )


def k3_no_path(mp: pytest.MonkeyPatch) -> None:
    def wrong(real: Any, cwd: str, path: str) -> str:
        try:
            return str(real(cwd, path))
        except ProcessCwdReadError as exc:
            raise ProcessCwdReadError(exc.native, None) from exc.native  # type: ignore[arg-type]

    wrap_resolution(mp, wrong)


def k4_provider_as_path(mp: pytest.MonkeyPatch) -> None:
    def wrong(real: Any, cwd: str, path: str) -> str:
        try:
            return str(real(cwd, path))
        except ProcessCwdReadError as exc:
            raise ProcessCwdReadError(exc.native, cwd) from exc.native

    wrap_resolution(mp, wrong)


def k5_eager_read(mp: pytest.MonkeyPatch) -> None:
    def wrong(real: Any, cwd: str, path: str) -> str:
        filesystem._process_cwd_reader(path)()
        return str(real(cwd, path))

    wrap_resolution(mp, wrong)


def k6_catch_all(mp: pytest.MonkeyPatch) -> None:
    def reader(argument: str) -> Callable[[], str]:
        def read() -> str:
            try:
                return filesystem._native_process_cwd_view()
            except Exception as exc:  # WRONG: every exception
                native = OSError(str(exc))
                setattr(native, filesystem._DIV009_MARK, True)
                raise ProcessCwdReadError(native, argument) from exc

        return read

    mp.setattr(filesystem, "_process_cwd_reader", reader)


def k7_fabricated_retry(mp: pytest.MonkeyPatch) -> None:
    def reader(argument: str) -> Callable[[], str]:
        def read() -> str:
            try:
                return filesystem._native_process_cwd_view()
            except OSError:
                return "C:\\"  # WRONG: a fabricated cwd

        return read

    mp.setattr(filesystem, "_process_cwd_reader", reader)


def k8_spawn_anyway(mp: pytest.MonkeyPatch) -> None:
    real = subprocess_module.resolve_local_path

    def wrong(cwd: str, path: str) -> str:
        try:
            return real(cwd, path)
        except ProcessCwdReadError:
            return cwd  # WRONG: falls back to the provider cwd and spawns

    mp.setattr(subprocess_module, "resolve_local_path", wrong)


def k9_old_rename_order(mp: pytest.MonkeyPatch) -> None:
    async def wrong(
        self: LocalFileSystem, source: str, destination: str, signal: RunSignal | None = None
    ) -> Any:
        dst = filesystem.resolve_local_path(self.cwd, destination)
        if signal is not None and signal.aborted:
            return Err(filesystem._aborted(dst))
        src = filesystem.resolve_local_path(self.cwd, source)
        await filesystem.asyncio.to_thread(os.replace, src, dst)
        return Ok(None)

    mp.setattr(
        LocalFileSystem,
        "rename_file",
        filesystem._typed_cwd_failure(filesystem._contain_nul(wrong)),
    )


def k10_exists_false(mp: pytest.MonkeyPatch) -> None:
    async def wrong(self: LocalFileSystem, path: str, signal: RunSignal | None = None) -> Any:
        info = await self.file_info(path)
        if isinstance(info, Ok):
            return Ok(True)
        if info.error.code == FsErrorCode.NOT_FOUND:  # WRONG: ignores the failure's origin
            return Ok(False)
        return info

    mp.setattr(LocalFileSystem, "exists", wrong)


def k11_resolve_fallback(mp: pytest.MonkeyPatch) -> None:
    async def wrong(self: LocalFileSystem, path: str, signal: RunSignal | None = None) -> Any:
        canonical = await self.canonical_path(path)
        if isinstance(canonical, Ok):
            return canonical
        if canonical.error.code in (FsErrorCode.NOT_FOUND, FsErrorCode.NOT_SUPPORTED):
            return await self.absolute_path(path)  # WRONG: reads again after the failure
        return canonical

    mp.setattr(LocalFileSystem, "resolve", wrong)


def k12_nul_reread(mp: pytest.MonkeyPatch) -> None:
    raw = LocalFileSystem.rename_file.__wrapped__.__wrapped__  # type: ignore[attr-defined]

    async def wrong(self: LocalFileSystem, *args: Any, **kwargs: Any) -> Any:
        try:
            return await raw(self, *args, **kwargs)
        except ValueError as exc:  # WRONG: the pre-DIV-009 handler resolves again
            source = filesystem.resolve_local_path(self.cwd, args[0])
            filesystem.resolve_local_path(self.cwd, args[1])
            return Err(filesystem._nul_failure(exc, source))

    mp.setattr(LocalFileSystem, "rename_file", filesystem._typed_cwd_failure(wrong))


def witness_of(
    fn: Callable[[pytest.MonkeyPatch], Coroutine[Any, Any, bool]],
) -> Callable[[pytest.MonkeyPatch], Callable[[], Coroutine[Any, Any, bool]]]:
    return lambda mp: lambda: fn(mp)


CONTROLS: list[tuple[str, Callable[[pytest.MonkeyPatch], None], Any]] = [
    ("K1-leak", k1_leak, first_reading_row_holds),
    ("K2-fixed-code", k2_fixed_code, first_reading_row_holds),
    ("K3-no-path", k3_no_path, first_reading_row_holds),
    ("K4-provider-path", k4_provider_as_path, first_reading_row_holds),
    ("K5-eager-read", k5_eager_read, no_read_row_holds),
    ("K6-catch-all", k6_catch_all, type_error_propagates),
    ("K7-fabricated-retry", k7_fabricated_retry, first_reading_row_holds),
    ("K8-spawn-anyway", k8_spawn_anyway, witness_of(subprocess_failure)),
    ("K9-old-rename-order", k9_old_rename_order, witness_of(r1_source_failure_under_abort)),
    ("K10-exists-false", k10_exists_false, witness_of(exists_propagates)),
    ("K11-resolve-fallback", k11_resolve_fallback, witness_of(resolve_propagates)),
    ("K12-nul-reread", k12_nul_reread, witness_of(r10_nul_path_without_reread)),
]


@WIN
@pytest.mark.parametrize(("name", "wrong", "witness"), CONTROLS, ids=[c[0] for c in CONTROLS])
async def test_control_is_killed_by_its_witness(
    name: str, wrong: Callable[[pytest.MonkeyPatch], None], witness: Any
) -> None:
    with pytest.MonkeyPatch.context() as baseline:
        assert await witness(baseline)(), f"{name}: the witness is not green on the baseline"
    with pytest.MonkeyPatch.context() as mutated:
        wrong(mutated)
        assert not await witness(mutated)(), f"{name} survived its witness"


async def test_nul_wrapper_reraises_a_value_error_before_resolution() -> None:
    """The NUL error path classifies only a rejection of a path the operation already resolved; a
    `ValueError` raised before any resolution is not a NUL case and propagates unchanged."""

    class Owner:
        cwd = "C:\\base"

    async def early(self: Owner, path: str, signal: RunSignal | None = None) -> Any:
        raise ValueError("raised before resolution")

    with pytest.raises(ValueError, match="before resolution"):
        await filesystem._contain_nul(early)(Owner(), "a\x00b")


def test_shell_module_converts_only_the_read_error() -> None:
    """The shell and subprocess seams import the one conversion type, nothing broader."""
    assert shell_module.ProcessCwdReadError is ProcessCwdReadError
    assert subprocess_module.ProcessCwdReadError is ProcessCwdReadError
    assert not issubclass(ProcessCwdReadError, OSError)
