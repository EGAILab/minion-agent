"""`EXEC-004`: `ctx.shell`, spec/execution.md section 5."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from minion_agent.execution.errors import ShellErrorCode, SubprocessError, SubprocessErrorCode
from minion_agent.execution.result import Err, Ok, Result
from minion_agent.execution.shell import LocalShell
from minion_agent.execution.subprocess import (
    ExitStatus,
    LocalSubprocess,
    Process,
    SpawnOptions,
)
from minion_agent.runtime.signal import RunAbortController

PY = sys.executable


def _py_shell_command(code: str) -> str:
    """A `bash -c` command line that runs a tiny Python snippet, portable across the bash the
    local shell discovers (Git Bash on Windows, `/bin/bash`/`sh` on POSIX)."""
    return f'"{PY}" -c "{code}"'


async def test_exec_success_returns_stdout_and_exit_code() -> None:
    shell = LocalShell()
    result = await shell.exec(_py_shell_command("print('hi')"))
    assert isinstance(result, Ok)
    assert result.value.stdout.strip() == "hi"
    assert result.value.exit_code == 0


async def test_exec_nonzero_exit_is_still_ok() -> None:
    """A non-zero exit is NOT a `Result` failure at this seam (spec section 5.6)."""
    shell = LocalShell()
    result = await shell.exec(_py_shell_command("import sys; sys.exit(7)"))
    assert isinstance(result, Ok)
    assert result.value.exit_code == 7


async def test_exec_pre_aborted_signal() -> None:
    shell = LocalShell()
    controller = RunAbortController()
    controller.abort()
    result = await shell.exec("echo hi", signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.ABORTED


async def test_exec_invalid_timeout_rejected_before_spawn() -> None:
    shell = LocalShell()
    result = await shell.exec("echo hi", timeout=0)
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.TIMEOUT


async def test_exec_exact_timeout_boundary_accepted() -> None:
    """`SHELL TIMEOUT EXACT BOUNDARY` witness: `2147483.647` is the largest accepted value --
    validated but never actually waited out in a test."""
    shell = LocalShell()
    result = await shell.exec(_py_shell_command("print(1)"), timeout=2147483.647)
    assert isinstance(result, Ok)


async def test_exec_exact_timeout_boundary_rejected() -> None:
    shell = LocalShell()
    result = await shell.exec("echo hi", timeout=2147483.648)
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.TIMEOUT


async def test_exec_timeout_kills_the_command() -> None:
    shell = LocalShell()
    result = await shell.exec(_py_shell_command("import time; time.sleep(30)"), timeout=0.1)
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.TIMEOUT


async def test_exec_missing_cwd_is_spawn_error(tmp_path: Path) -> None:
    shell = LocalShell()
    missing = str(tmp_path / "does-not-exist")
    result = await shell.exec("echo hi", cwd=missing)
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.SPAWN_ERROR


async def test_exec_configured_nonexistent_shell_and_nonexistent_cwd_is_shell_unavailable(
    tmp_path: Path,
) -> None:
    """`SHELL PRE-SPAWN ORDER, COMBINED INVALIDITY` witness (`L12-R010`): shell discovery fails
    BEFORE the cwd existence check -- a candidate checking cwd first would return `spawn_error`
    instead, which this witness would catch."""
    shell = LocalShell(shell_path=str(tmp_path / "no-such-shell"))
    missing_cwd = str(tmp_path / "no-such-cwd")
    result = await shell.exec("echo hi", cwd=missing_cwd)
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.SHELL_UNAVAILABLE


async def test_exec_custom_shell_path_used_when_it_exists() -> None:
    """A configured custom shell path, if it exists, is used directly (no PATH/well-known-path
    search)."""
    shell = LocalShell(shell_path=_discover_a_real_bash())
    result = await shell.exec(_py_shell_command("print('via-custom-shell')"))
    assert isinstance(result, Ok)
    assert result.value.stdout.strip() == "via-custom-shell"


def _discover_a_real_bash() -> str:
    if os.name == "nt":
        for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(env_var)
            if base:
                candidate = os.path.join(base, "Git", "bin", "bash.exe")
                if os.path.exists(candidate):
                    return candidate
        found = shutil.which("bash")
        if found:
            return found
        pytest.skip("no real bash available to exercise the custom-shell-path case")
    if os.path.exists("/bin/bash"):
        return "/bin/bash"
    found = shutil.which("bash")
    if found:
        return found
    return "sh"


async def test_exec_callback_error_wins_over_timeout() -> None:
    """`SHELL FAILURE PRECEDENCE` witness: `callback_error` takes precedence over `timeout` even
    when both conditions are true for the same call."""
    shell = LocalShell()

    def raising_callback(_chunk: str) -> None:
        raise ValueError("callback boom")

    result = await shell.exec(
        _py_shell_command("print('x', flush=True); import time; time.sleep(30)"),
        timeout=0.2,
        on_stdout=raising_callback,
    )
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.CALLBACK_ERROR


async def test_exec_streams_stdout_via_callback_in_addition_to_accumulating() -> None:
    """`L12-PY-R005`: callback payloads are DECODED TEXT (`str`), matching pinned Pi's own
    `onStdout?: (chunk: string) => void` exactly -- not raw bytes."""
    shell = LocalShell()
    seen: list[str] = []
    result = await shell.exec(_py_shell_command("print('streamed')"), on_stdout=seen.append)
    assert isinstance(result, Ok)
    assert all(isinstance(chunk, str) for chunk in seen)
    assert "".join(seen).strip() == "streamed"
    assert result.value.stdout.strip() == "streamed"


async def test_exec_streams_stderr_via_callback() -> None:
    shell = LocalShell()
    seen: list[str] = []
    result = await shell.exec(
        _py_shell_command("import sys; print('err', file=sys.stderr)"), on_stderr=seen.append
    )
    assert isinstance(result, Ok)
    assert all(isinstance(chunk, str) for chunk in seen)
    assert "".join(seen).strip() == "err"
    assert result.value.stderr.strip() == "err"


class _ManualReadableStream:
    """A hand-fed stand-in for `ReadableStream` giving the test EXACT control over chunk
    boundaries -- real OS pipes (especially through an intermediate `bash -c` hop) do not
    reliably preserve a deliberate split even across a sleep. `read_started` is set the moment
    a `read_chunk()` call begins (before it awaits `feed`), so the test can synchronize on the
    PUMP LOOP actually having started its next read, rather than guessing at scheduling ticks."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.read_started = asyncio.Event()

    def feed(self, chunk: bytes | None) -> None:
        self._queue.put_nowait(chunk)

    async def read_chunk(self) -> Ok[bytes | None]:
        self.read_started.set()
        chunk = await self._queue.get()
        return Ok(chunk)


class _FakeProcess:
    """A minimal `Process` duck-type wrapping two `_ManualReadableStream`s."""

    def __init__(self, stdout: _ManualReadableStream, stderr: _ManualReadableStream) -> None:
        self.stdout = stdout
        self.stderr = stderr

    async def wait(self) -> Ok[ExitStatus]:
        return Ok(ExitStatus(exit_code=0))

    async def terminate(self) -> None:
        pass


async def test_exec_streams_stdout_correctly_reassembles_a_multibyte_char_split_across_chunks() -> (
    None
):
    """`L12-PY-R005` witness: a single UTF-8 character split across two separate `read_chunk()`
    calls must still reassemble correctly through the callback -- a naive per-chunk
    `bytes.decode()` would raise or emit a replacement character for the first (incomplete)
    half. Drives `_run_to_completion` directly (see `_FakeProcess`/`_ManualReadableStream`) for
    a deterministic, platform-independent split, rather than trusting real OS/pipe timing to
    land a 4-byte write exactly on a chunk boundary."""
    emoji = "\U0001f600"  # 4 UTF-8 bytes: F0 9F 98 80 -- split after the first 2.
    emoji_bytes = emoji.encode("utf-8")
    stdout_stream = _ManualReadableStream()
    stderr_stream = _ManualReadableStream()
    stderr_stream.feed(None)  # immediate EOF -- this witness only cares about stdout
    process = _FakeProcess(stdout_stream, stderr_stream)

    shell = LocalShell()
    seen: list[str] = []
    run_task = asyncio.ensure_future(
        shell._run_to_completion(process, None, seen.append, None)  # type: ignore[arg-type]
    )

    await asyncio.wait_for(stdout_stream.read_started.wait(), timeout=2.0)  # 1st read_chunk()
    stdout_stream.read_started.clear()
    stdout_stream.feed(emoji_bytes[:2])  # incomplete sequence -- decoder emits nothing yet

    await asyncio.wait_for(stdout_stream.read_started.wait(), timeout=2.0)  # 2nd read_chunk()
    stdout_stream.read_started.clear()
    stdout_stream.feed(emoji_bytes[2:])  # completes the sequence -- decoder emits the char

    await asyncio.wait_for(stdout_stream.read_started.wait(), timeout=2.0)  # 3rd read_chunk()
    stdout_stream.feed(None)  # EOF

    result = await run_task

    assert isinstance(result, Ok)
    assert all(isinstance(chunk, str) for chunk in seen)
    assert seen == [emoji]  # the incomplete first half produced NO callback call on its own
    assert result.value.stdout == emoji


async def test_exec_flushes_a_dangling_incomplete_sequence_at_eof_as_replacement_char() -> None:
    """The other half of the `L12-PY-R005` incremental-decoder witness: a stream that reaches
    EOF WHILE an incomplete multi-byte sequence is still buffered in the decoder must still
    flush it through the callback as a replacement character (`errors="replace"`, matching
    Node's own non-throwing decode semantics, `L12-PY-R002`) -- not silently drop it."""
    stdout_stream = _ManualReadableStream()
    stderr_stream = _ManualReadableStream()
    stderr_stream.feed(None)
    process = _FakeProcess(stdout_stream, stderr_stream)

    shell = LocalShell()
    seen: list[str] = []
    run_task = asyncio.ensure_future(
        shell._run_to_completion(process, None, seen.append, None)  # type: ignore[arg-type]
    )

    await asyncio.wait_for(stdout_stream.read_started.wait(), timeout=2.0)
    stdout_stream.read_started.clear()
    stdout_stream.feed(b"\xf0\x9f")  # first half of the same emoji -- deliberately never completed
    await asyncio.wait_for(stdout_stream.read_started.wait(), timeout=2.0)
    stdout_stream.feed(None)  # EOF while the sequence is still dangling

    result = await run_task

    assert isinstance(result, Ok)
    assert seen == ["�"]
    assert result.value.stdout == "�"


async def test_exec_callback_error_on_trailing_flush_is_reported() -> None:
    """Closes `pump()`'s OTHER callback-error branch: a callback that raises specifically on
    the trailing (at-EOF) flush -- not on an ordinary per-chunk call, already covered by
    `test_exec_callback_error_wins_over_timeout` -- must still be captured as `callback_error`
    and terminate the process, exactly like the ordinary per-chunk case."""
    stdout_stream = _ManualReadableStream()
    stderr_stream = _ManualReadableStream()
    stderr_stream.feed(None)
    process = _FakeProcess(stdout_stream, stderr_stream)

    def raising_callback(_chunk: str) -> None:
        raise ValueError("callback boom on trailing flush")

    shell = LocalShell()
    run_task = asyncio.ensure_future(
        shell._run_to_completion(  # type: ignore[arg-type]
            process, None, raising_callback, None
        )
    )

    await asyncio.wait_for(stdout_stream.read_started.wait(), timeout=2.0)
    stdout_stream.read_started.clear()
    stdout_stream.feed(b"\xf0\x9f")  # dangling incomplete sequence, never completed
    await asyncio.wait_for(stdout_stream.read_started.wait(), timeout=2.0)
    stdout_stream.feed(None)  # EOF -- the trailing flush fires the raising callback

    result = await run_task

    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.CALLBACK_ERROR


async def test_exec_streams_multiple_chunks_before_exit() -> None:
    """Sanity check that ordinary multi-chunk streaming still produces the full accumulated
    output when the process exits normally (both streams reach EOF, settling via that path --
    see `test_completion_idle_timer_resets_on_data` below for the EXACT reset-constant witness,
    tested directly against `_Completion`'s own state machine rather than through a real
    process, since ordinary EOF-based settlement races ahead of the idle timer whenever no
    detached descendant is still holding a pipe open -- exactly as spec section 5.6 describes)."""
    shell = LocalShell()
    result = await shell.exec(
        _py_shell_command(
            "import time; print('a', flush=True); time.sleep(0.05); print('b', flush=True)"
        )
    )
    assert isinstance(result, Ok)
    assert result.value.stdout.split() == ["a", "b"]


async def test_completion_idle_timer_resets_on_data() -> None:
    """`SHELL IDLE-GRACE, EXACT RESET CONSTANT` witness, tested directly against `_Completion`'s
    own state machine for deterministic timing: once armed (on exit), the 100ms timer is reset
    by each `on_data()` call, so it does NOT fire at the original 100ms mark once new data has
    arrived, but DOES fire ~100ms after the LAST reset."""
    from minion_agent.execution.shell import _Completion

    completion = _Completion()
    completion.on_exit()  # arms the first 100ms window
    await asyncio.sleep(0.05)
    assert not completion._settled.is_set()
    completion.on_data()  # resets to a FRESH 100ms window from now
    await asyncio.sleep(0.07)  # 120ms since exit, but only 70ms since the reset
    assert not completion._settled.is_set()
    await asyncio.sleep(0.05)  # ~120ms since the reset -- now past the fresh 100ms window
    assert completion._settled.is_set()


async def test_completion_settles_immediately_when_both_streams_end_before_the_timer_fires() -> (
    None
):
    """The other settlement path: both streams ending on their own (no detached descendant
    holding stdio open) settles WITHOUT waiting for the idle timer at all."""
    from minion_agent.execution.shell import _Completion

    completion = _Completion()
    completion.on_exit()
    completion.on_stdout_end()
    completion.on_stderr_end()
    assert completion._settled.is_set()


async def test_cleanup_kills_active_commands_settlement_is_success() -> None:
    """`CTX.SHELL CLEANUP KILLS ACTIVE COMMANDS; SETTLEMENT PRESERVES A REAL EXIT CODE` witness:
    a cleanup-killed in-flight command settles as SUCCESS (not aborted, not timeout)."""
    shell = LocalShell()
    exec_task = asyncio.ensure_future(shell.exec(_py_shell_command("import time; time.sleep(30)")))
    await asyncio.sleep(0.2)  # let the command actually start and register as active
    await shell.cleanup()
    result = await exec_task
    assert isinstance(result, Ok)
    assert isinstance(result.value.exit_code, int)  # ShellResult's own exit_code is never None


async def test_cleanup_with_no_active_commands_is_a_true_no_op() -> None:
    shell = LocalShell()
    await shell.cleanup()  # must not raise


async def test_local_provider_parity_is_per_seam_smoke() -> None:
    """`LOCAL-PROVIDER PARITY IS PER-SEAM, NOT BLANKET` witness's own PYTHON-side observable
    correlate: `ctx.shell` behaves per its `DIRECT_PI_PARITY` disposition (genuinely spawns a
    real shell and reports a real exit code), distinct from `ctx.subprocess`'s own
    `MINION_EXTENSION` primitives exercised separately in `test_subprocess.py`."""
    shell = LocalShell()
    result = await shell.exec(_py_shell_command("import sys; sys.exit(3)"))
    assert isinstance(result, Ok)
    assert result.value.exit_code == 3


async def test_completion_wait_settled_cancels_the_idle_timer() -> None:
    from minion_agent.execution.shell import _Completion

    completion = _Completion()
    completion.on_exit()
    completion.on_stdout_end()
    completion.on_stderr_end()
    await completion.wait_settled()  # must return immediately; must not raise
    assert completion._settled.is_set()


async def test_exec_auto_discovers_a_shell_without_a_configured_path() -> None:
    """Exercises the real, unpatched shell-discovery branch (no `shell_path` configured) --
    distinct from every other test in this module, which all pass an explicit `shell_path`."""
    shell = LocalShell()
    result = await shell.exec(_py_shell_command("print('auto-discovered')"))
    assert isinstance(result, Ok)
    assert result.value.stdout.strip() == "auto-discovered"


async def test_exec_no_bash_found_on_this_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """The "no bash shell found" error branch -- simulated by hiding every discovery path this
    platform's own `_resolve_shell` would otherwise take."""
    shell = LocalShell()
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    result = await shell.exec("echo hi")
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.SHELL_UNAVAILABLE


@pytest.mark.skipif(os.name != "nt", reason="exercises the Windows-only PATH fallback branch")
async def test_exec_windows_discovers_bash_on_path_when_program_files_lacks_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows discovery step 3: `Program Files\\Git\\bin\\bash.exe` absent, but `bash` is on
    `PATH` -- distinct from `test_exec_no_bash_found_on_this_platform`, which hides both."""
    shell = LocalShell()
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    monkeypatch.setattr(shutil, "which", lambda _name: "C:\\fake\\bash.exe")
    result = await shell._resolve_shell()
    assert isinstance(result, Ok)
    assert result.value == "C:\\fake\\bash.exe"


async def test_exec_spawn_failure_after_cwd_check_maps_to_spawn_error() -> None:
    """The defensive `spawn_result` `Err` branch in `exec()`: reachable if the underlying
    `ctx.subprocess` seam fails to spawn even after this seam's own pre-spawn checks passed --
    exercised via an injected failing subprocess seam."""

    class _FailingSubprocess(LocalSubprocess):
        async def spawn(
            self, argv: Sequence[str], options: SpawnOptions | None = None
        ) -> Result[Process, SubprocessError]:
            return Err(SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "simulated spawn failure"))

    shell = LocalShell(subprocess_seam=_FailingSubprocess())
    result = await shell.exec(_py_shell_command("print('unreachable')"))
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.SPAWN_ERROR


async def test_exec_spawn_abort_race_preserves_aborted_not_spawn_error() -> None:
    """`L12-PY-R006` witness: when the underlying `ctx.subprocess` seam's own `spawn()` fails
    with `SubprocessErrorCode.ABORTED` -- the narrow race where a signal fires between this
    shell seam's own pre-check and the actual spawn call -- that must be preserved as
    `ShellErrorCode.ABORTED`, not unconditionally collapsed into `SPAWN_ERROR` alongside every
    other spawn failure reason."""

    class _AbortingSubprocess(LocalSubprocess):
        async def spawn(
            self, argv: Sequence[str], options: SpawnOptions | None = None
        ) -> Result[Process, SubprocessError]:
            return Err(SubprocessError(SubprocessErrorCode.ABORTED, "aborted"))

    shell = LocalShell(subprocess_seam=_AbortingSubprocess())
    result = await shell.exec(_py_shell_command("print('unreachable')"))
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.ABORTED


async def test_exec_signal_aborted_mid_flight_classifies_aborted() -> None:
    """The `isinstance(wait_result, Err)` -> `ABORTED` classification branch: a signal that
    fires AFTER the process has started (not pre-aborted) still kills it and classifies the
    whole `exec()` call `aborted`."""
    shell = LocalShell()
    controller = RunAbortController()
    exec_task = asyncio.ensure_future(
        shell.exec(_py_shell_command("import time; time.sleep(30)"), signal=controller.signal)
    )
    await asyncio.sleep(0.2)
    controller.abort()
    result = await exec_task
    assert isinstance(result, Err)
    assert result.error.code == ShellErrorCode.ABORTED
