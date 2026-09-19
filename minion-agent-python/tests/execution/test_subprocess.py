"""`EXEC-005`: `ctx.subprocess`, spec/execution.md section 6."""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from minion_agent.execution.errors import SubprocessErrorCode
from minion_agent.execution.result import Err, Ok
from minion_agent.execution.subprocess import (
    LocalSubprocess,
    SpawnOptions,
    StdioMode,
    _wait_and_settle_helper,
)
from minion_agent.runtime.signal import RunAbortController

PY = sys.executable


async def test_spawn_and_wait_success() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "import sys; sys.exit(0)"])
    assert isinstance(result, Ok)
    process = result.value
    wait_result = await process.wait()
    assert isinstance(wait_result, Ok)  # settled at all
    assert wait_result.value.exit_code == 0


async def test_spawn_nonzero_exit_code_preserved() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "import sys; sys.exit(7)"])
    process = result.value  # type: ignore[union-attr]
    wait_result = await process.wait()
    assert isinstance(wait_result, Ok)
    assert wait_result.value.exit_code == 7


async def test_spawn_argv_direct_no_shell_interpretation() -> None:
    """`argv`-direct only -- glob/variable-expansion characters in an argument are passed
    through LITERALLY, never shell-interpreted."""
    sp = LocalSubprocess()
    result = await sp.spawn(
        [PY, "-c", "import sys; print(sys.argv[1])", "$HOME *.txt"],
        SpawnOptions(stdout=StdioMode.PIPED),
    )
    process = result.value  # type: ignore[union-attr]
    assert process.stdout is not None
    chunks = b""
    while True:
        chunk_result = await process.stdout.read_chunk()
        assert isinstance(chunk_result, Ok)
        if chunk_result.value is None:
            break
        chunks += chunk_result.value
    assert chunks.strip() == b"$HOME *.txt"
    await process.wait()


async def test_spawn_binary_not_found_is_spawn_error() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn(["definitely-not-a-real-binary-xyz"])
    assert isinstance(result, Err)
    assert result.error.code == SubprocessErrorCode.SPAWN_ERROR


async def test_spawn_missing_cwd_is_spawn_error(tmp_path: Path) -> None:
    sp = LocalSubprocess()
    missing = str(tmp_path / "does-not-exist")
    result = await sp.spawn([PY, "-c", "pass"], SpawnOptions(cwd=missing))
    assert isinstance(result, Err)
    assert result.error.code == SubprocessErrorCode.SPAWN_ERROR


async def test_spawn_pre_aborted_signal_short_circuits() -> None:
    sp = LocalSubprocess()
    controller = RunAbortController()
    controller.abort()
    result = await sp.spawn([PY, "-c", "pass"], SpawnOptions(signal=controller.signal))
    assert isinstance(result, Err)
    assert result.error.code == SubprocessErrorCode.ABORTED


async def test_spawn_signal_aborted_after_start_triggers_termination() -> None:
    """`PROCESS WAIT, ONE SIGNAL` witness: aborting the ORIGINAL spawn signal kills the process
    and `wait()` classifies `Err(aborted)`."""
    sp = LocalSubprocess()
    controller = RunAbortController()
    result = await sp.spawn(
        [PY, "-c", "import time; time.sleep(30)"], SpawnOptions(signal=controller.signal)
    )
    process = result.value  # type: ignore[union-attr]
    await asyncio.sleep(0.05)  # give the watcher task a moment to start polling
    controller.abort()
    wait_result = await process.wait()
    assert isinstance(wait_result, Err)
    assert wait_result.error.code == SubprocessErrorCode.ABORTED


async def test_watch_signal_task_completes_its_own_kill_and_return_before_wait_is_called() -> None:
    """Deterministically closes `_watch_signal`'s own terminal `return` line -- distinct from
    `test_spawn_signal_aborted_after_start_triggers_termination` above, which races the watcher
    task against `wait()`'s own cancellation of it (so it does not reliably reach its own
    `return` before being cancelled). Here the watcher is given enough time to finish killing
    the process and return ON ITS OWN before `wait()` is ever called, removing that race."""
    sp = LocalSubprocess()
    controller = RunAbortController()
    result = await sp.spawn(
        [PY, "-c", "import time; time.sleep(30)"], SpawnOptions(signal=controller.signal)
    )
    process = result.value  # type: ignore[union-attr]
    controller.abort()
    await asyncio.sleep(0.3)  # let the watcher task kill the process AND reach its own return
    wait_result = await process.wait()
    assert isinstance(wait_result, Err)
    assert wait_result.error.code == SubprocessErrorCode.ABORTED


def test_wait_and_settle_helper_kills_after_timeout_and_waits_again() -> None:
    """`L12-PY-R007` coverage-closing: `_wait_and_settle_helper`'s own timeout-fallback branch
    -- if the `taskkill` helper process is somehow still running past the bounded wait, kill it
    directly and wait once more, best-effort."""

    class _FakeHelper:
        def __init__(self) -> None:
            self.wait_calls = 0
            self.killed = False

        def wait(self, timeout: float) -> None:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="taskkill", timeout=timeout)

        def kill(self) -> None:
            self.killed = True

    helper = _FakeHelper()
    _wait_and_settle_helper(helper)  # type: ignore[arg-type]
    assert helper.killed
    assert helper.wait_calls == 2


async def test_wait_first_call_after_natural_exit_and_later_signal_abort_is_ok() -> None:
    """`L12-PY-R004` witness: the process EXITS NATURALLY (never killed by anything), and only
    AFTER that does its spawn signal get aborted -- an unrelated abort with no causal
    connection to this process's own termination. The FIRST call to `wait()` (uncached, so the
    classification logic is genuinely exercised, not just the cache) must still settle `Ok`
    with the real exit code -- reactively re-checking `signal.aborted` at `wait()` time would
    wrongly report `Err(aborted)` here, since the signal happens to be aborted by then even
    though it never caused anything."""
    sp = LocalSubprocess()
    controller = RunAbortController()
    result = await sp.spawn(
        [PY, "-c", "import sys; sys.exit(5)"], SpawnOptions(signal=controller.signal)
    )
    process = result.value  # type: ignore[union-attr]
    await asyncio.sleep(0.2)  # let the process exit naturally, well before any abort
    controller.abort()  # fires AFTER natural exit -- must not retroactively poison the result
    wait_result = await process.wait()  # first (uncached) call
    assert isinstance(wait_result, Ok)
    assert wait_result.value.exit_code == 5


async def test_wait_first_call_after_explicit_terminate_and_later_signal_abort_is_ok() -> None:
    """Case B of the same witness: `terminate()` is called explicitly while the spawn signal is
    still un-aborted (so the kill cause is recorded as `"explicit"`, not `"signal"`). The SAME
    spawn signal is then aborted afterward, unrelated to the termination that already happened.
    `wait()`'s FIRST (uncached) call must still settle `Ok`, not flip to `Err(aborted)` just
    because the signal happens to be aborted by the time `wait()` runs -- whichever cause was
    recorded FIRST wins."""
    sp = LocalSubprocess()
    controller = RunAbortController()
    result = await sp.spawn(
        [PY, "-c", "import time; time.sleep(30)"], SpawnOptions(signal=controller.signal)
    )
    process = result.value  # type: ignore[union-attr]
    await process.terminate()  # explicit kill; signal never fired -- kill_cause = "explicit"
    controller.abort()  # fires AFTER the explicit terminate -- must not flip the recorded cause
    wait_result = await process.wait()  # first (uncached) call
    assert isinstance(wait_result, Ok)


async def test_wait_takes_no_signal_argument() -> None:
    """`wait()` accepts no arguments at all -- there is no separate wait-time signal."""
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "pass"])
    process = result.value  # type: ignore[union-attr]
    assert list(inspect.signature(process.wait).parameters) == []
    await process.wait()  # `L12-PY-R007`: settle/close, don't leak this process to the GC.


async def test_terminate_settles_wait_as_success() -> None:
    """`PROCESS TERMINATE() SETTLEMENT PRESERVES A REAL EXIT CODE` witness, case A: an explicit
    `terminate()` with no spawn signal involved always settles `Ok`, never `Err`.

    The exit_code VALUE inside that `Ok` is platform-dependent, and correctly so (see
    `subprocess.py`'s own comment on this): POSIX reports a signal-terminated child with a
    negative returncode, mapped to `None` (no numeric code); Windows' `taskkill`/
    `TerminateProcess` reports a real, non-negative code (commonly `1`) with no equivalent
    "absent" case at all -- there, the reported code IS the real code, preserved as such."""
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "import time; time.sleep(30)"])
    process = result.value  # type: ignore[union-attr]
    await process.terminate()
    wait_result = await process.wait()
    assert isinstance(wait_result, Ok)
    if os.name == "nt":
        assert isinstance(wait_result.value.exit_code, int)
    else:
        assert wait_result.value.exit_code is None


async def test_terminate_after_natural_exit_preserves_the_real_code() -> None:
    """Case B of the same witness: a process that already exited with a real numeric code before
    `terminate()` is called keeps that real code -- `terminate()` on an already-finished process
    is a no-op, and `wait()` must not overwrite the genuine exit status."""
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "import sys; sys.exit(42)"])
    process = result.value  # type: ignore[union-attr]
    first_wait = await process.wait()
    await process.terminate()  # no-op: already exited
    second_wait = await process.wait()
    assert first_wait == second_wait
    assert isinstance(second_wait, Ok)
    assert second_wait.value.exit_code == 42


async def test_terminate_is_idempotent() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "import time; time.sleep(30)"])
    process = result.value  # type: ignore[union-attr]
    await process.terminate()
    await process.terminate()  # must not raise
    await process.wait()


async def test_wait_is_idempotent_and_cached() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "import sys; sys.exit(3)"])
    process = result.value  # type: ignore[union-attr]
    first = await process.wait()
    second = await process.wait()
    assert first == second


async def test_wait_is_safe_when_called_concurrently() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "import sys; sys.exit(5)"])
    process = result.value  # type: ignore[union-attr]
    results = await asyncio.gather(process.wait(), process.wait(), process.wait())
    assert all(r == results[0] for r in results)
    assert results[0].value.exit_code == 5


async def test_async_context_manager_terminates_on_exit() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "import time; time.sleep(30)"])
    process = result.value  # type: ignore[union-attr]
    async with process:
        pass
    wait_result = await process.wait()
    assert isinstance(wait_result, Ok)


async def test_stdio_null_mode_populates_no_stream() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn(
        [PY, "-c", "pass"],
        SpawnOptions(stdin=StdioMode.NULL, stdout=StdioMode.NULL, stderr=StdioMode.NULL),
    )
    process = result.value  # type: ignore[union-attr]
    assert process.stdin is None
    assert process.stdout is None
    assert process.stderr is None
    await process.wait()


async def test_stdio_inherit_mode_populates_no_stream_handle() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn(
        [PY, "-c", "pass"],
        SpawnOptions(stdin=StdioMode.INHERIT, stdout=StdioMode.INHERIT, stderr=StdioMode.INHERIT),
    )
    process = result.value  # type: ignore[union-attr]
    assert process.stdout is None
    assert process.stderr is None
    await process.wait()


async def test_writable_stream_write_and_close() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn(
        [PY, "-c", "import sys; data = sys.stdin.buffer.read(); sys.stdout.buffer.write(data)"],
        SpawnOptions(stdin=StdioMode.PIPED, stdout=StdioMode.PIPED),
    )
    process = result.value  # type: ignore[union-attr]
    assert process.stdin is not None
    write_result = await process.stdin.write(b"hello")
    assert write_result == Ok(None)
    await process.stdin.close()
    await process.stdin.close()  # idempotent, must not raise
    assert process.stdout is not None
    chunks = b""
    while True:
        chunk_result = await process.stdout.read_chunk()
        assert isinstance(chunk_result, Ok)
        if chunk_result.value is None:
            break
        chunks += chunk_result.value
    assert chunks == b"hello"
    await process.wait()


async def test_read_chunk_returns_none_at_eof() -> None:
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "print('x')"], SpawnOptions(stdout=StdioMode.PIPED))
    process = result.value  # type: ignore[union-attr]
    assert process.stdout is not None
    saw_data = False
    while True:
        chunk_result = await process.stdout.read_chunk()
        assert isinstance(chunk_result, Ok)
        if chunk_result.value is None:
            break
        saw_data = True
    assert saw_data
    await process.wait()


async def test_pipe_failure_does_not_affect_wait() -> None:
    """`PIPE FAILURE INDEPENDENCE` witness: a pipe read after the stream has already been closed
    at the OS level is independent of the process's own `wait()` result."""
    sp = LocalSubprocess()
    result = await sp.spawn(
        [PY, "-c", "import sys; sys.exit(0)"], SpawnOptions(stdout=StdioMode.PIPED)
    )
    process = result.value  # type: ignore[union-attr]
    wait_result = await process.wait()
    assert wait_result.value.exit_code == 0  # type: ignore[union-attr]
    # Draining after exit is still well-defined (EOF), proving wait() didn't consume/break stdout.
    assert process.stdout is not None
    drained = await process.stdout.read_chunk()
    assert isinstance(drained, Ok)


async def test_cwd_defaults_to_the_providers_own_cwd(tmp_path: Path) -> None:
    """`SUBPROCESS CWD/ENVIRONMENT DEFAULTS` witness."""
    sp = LocalSubprocess(cwd=str(tmp_path))
    result = await sp.spawn(
        [PY, "-c", "import os; print(os.getcwd())"], SpawnOptions(stdout=StdioMode.PIPED)
    )
    process = result.value  # type: ignore[union-attr]
    assert process.stdout is not None
    output = b""
    while True:
        chunk = await process.stdout.read_chunk()
        if chunk.value is None:  # type: ignore[union-attr]
            break
        output += chunk.value  # type: ignore[union-attr]
    await process.wait()
    assert os.path.samefile(output.decode().strip(), str(tmp_path))


async def test_inherit_env_true_overlays_base_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUBPROCESS_TEST_BASE_VAR", "base-value")
    sp = LocalSubprocess()
    code = (
        "import os; print(os.environ.get('SUBPROCESS_TEST_BASE_VAR', ''), os.environ.get('X', ''))"
    )
    result = await sp.spawn(
        [PY, "-c", code],
        SpawnOptions(env={"X": "1"}, inherit_env=True, stdout=StdioMode.PIPED),
    )
    process = result.value  # type: ignore[union-attr]
    assert process.stdout is not None
    output = b""
    while True:
        chunk = await process.stdout.read_chunk()
        if chunk.value is None:  # type: ignore[union-attr]
            break
        output += chunk.value  # type: ignore[union-attr]
    await process.wait()
    assert output.decode().strip() == "base-value 1"


async def test_inherit_env_false_is_exactly_the_supplied_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUBPROCESS_TEST_BASE_VAR", "base-value")
    sp = LocalSubprocess()
    code = (
        "import os; print(os.environ.get('SUBPROCESS_TEST_BASE_VAR', 'ABSENT'), "
        "os.environ.get('X', ''))"
    )
    result = await sp.spawn(
        [PY, "-c", code],
        SpawnOptions(env={"X": "1"}, inherit_env=False, stdout=StdioMode.PIPED),
    )
    process = result.value  # type: ignore[union-attr]
    assert process.stdout is not None
    output = b""
    while True:
        chunk = await process.stdout.read_chunk()
        if chunk.value is None:  # type: ignore[union-attr]
            break
        output += chunk.value  # type: ignore[union-attr]
    await process.wait()
    assert output.decode().strip() == "ABSENT 1"


async def test_writable_stream_write_failure_maps_to_pipe_error() -> None:
    """A genuine, portable OSError trigger: writing to stdin after the underlying writer has
    already been closed."""
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "pass"], SpawnOptions(stdin=StdioMode.PIPED))
    process = result.value  # type: ignore[union-attr]
    assert process.stdin is not None
    await process.wait()
    await process.stdin.close()
    write_result = await process.stdin.write(b"too late")
    assert isinstance(write_result, Err)
    assert write_result.error.code == SubprocessErrorCode.PIPE_ERROR


async def test_readable_stream_read_chunk_failure_maps_to_pipe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sp = LocalSubprocess()
    result = await sp.spawn([PY, "-c", "print('x')"], SpawnOptions(stdout=StdioMode.PIPED))
    process = result.value  # type: ignore[union-attr]
    assert process.stdout is not None

    async def _raise(_n: int) -> bytes:
        raise OSError("simulated pipe failure")

    monkeypatch.setattr(process.stdout, "_reader", type("_R", (), {"read": staticmethod(_raise)})())
    read_result = await process.stdout.read_chunk()
    assert isinstance(read_result, Err)
    assert read_result.error.code == SubprocessErrorCode.PIPE_ERROR
    await process.wait()
