"""`EXEC-005`: `ctx.subprocess`, spec/execution.md section 6."""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from minion_agent.execution import subprocess as subprocess_module
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


async def test_signal_kill_attempt_with_no_effect_does_not_classify_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L12-PY-R004` witness (refined, second review): the signal fires while `_watch_signal`'s
    poll loop still observes `returncode is None`, so it DOES attempt a kill -- but the process
    has, by the time the attempt actually runs, already exited naturally (or via a concurrent
    explicit action) on its own, making the kill attempt a genuine no-op. `wait()` must settle
    `Ok` with the process's own REAL exit status, not `Err(aborted)`, merely because a kill was
    ATTEMPTED -- only an attempt that `_kill_process_tree` itself confirms found a live target
    may be recorded as causal. Monkeypatches `_kill_process_tree` to report `False` (no live
    target) unconditionally, and gives the child a long enough natural lifetime that the abort
    is guaranteed to fire while `_watch_signal`'s loop still observes `returncode is None` (so
    it deterministically enters the kill-attempt branch, lines 314-316, rather than depending on
    real OS-level exit-detection timing -- the review's own exact race is not independently
    reproducible against genuinely live, non-deterministic process timing)."""

    async def fake_kill_process_tree(pid: int) -> bool:
        return False

    monkeypatch.setattr(subprocess_module, "_kill_process_tree", fake_kill_process_tree)

    sp = LocalSubprocess()
    controller = RunAbortController()
    result = await sp.spawn(
        [PY, "-c", "import time; time.sleep(0.5); import sys; sys.exit(0)"],
        SpawnOptions(signal=controller.signal),
    )
    process = result.value  # type: ignore[union-attr]
    await asyncio.sleep(0.05)  # child is still running (sleeping); watcher sees returncode None
    controller.abort()  # _watch_signal's poll loop attempts a kill; the mock reports no effect
    await asyncio.sleep(0.05)  # give the watcher a chance to observe the abort and attempt it
    wait_result = await process.wait()  # settles once the child's own 0.5s sleep finishes
    assert isinstance(wait_result, Ok), (
        f"a no-op kill attempt must not classify Err(aborted): {wait_result!r}"
    )
    assert wait_result.value.exit_code == 0


async def test_wait_awaits_a_pending_kill_attempt_before_reading_kill_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L12-PY-R004` witness (refined a third time, targeted closure review): an IMMEDIATE mock
    result (as in the test above) is not a sufficient negative control for this race, per the
    review -- it never exercises `wait()` actually observing the process's own exit WHILE a kill
    attempt is still in flight. This test controls that exact interleaving deterministically with
    `asyncio.Event`s instead of relying on sleep-based timing luck:

    1. `_kill_process_tree` is monkeypatched to signal `entered` the moment it's called, then
       BLOCK on `release` before returning `False` (no live target found).
    2. The real child process is left running; once the kill attempt is confirmed pending
       (`entered` is set), `process.wait()` is started concurrently and given time to observe
       the child exit on its own (`self._proc.wait()` resolves) -- but the kill attempt is still
       blocked on `release`, so `wait()` must NOT have settled yet.
    3. Only then is `release` set, letting the pending kill attempt resolve to `False`.

    An earlier revision's `wait()` cancelled the watcher task as soon as it observed the
    process's own exit (step 2), which would have thrown away the still-pending correction step
    entirely and left the eagerly-set `"signal"` flag uncorrected -- misclassifying this genuinely
    no-effect kill as `Err(aborted)`. The current `wait()` awaits the watcher instead of
    cancelling it, so the correction always gets to run before `_kill_cause` is read."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fake_kill_process_tree(pid: int) -> bool:
        entered.set()
        await release.wait()
        return False

    monkeypatch.setattr(subprocess_module, "_kill_process_tree", fake_kill_process_tree)

    sp = LocalSubprocess()
    controller = RunAbortController()
    result = await sp.spawn(
        [PY, "-c", "import time; time.sleep(0.15); import sys; sys.exit(0)"],
        SpawnOptions(signal=controller.signal),
    )
    process = result.value  # type: ignore[union-attr]
    await asyncio.sleep(0.02)  # watcher is polling; child is still running
    controller.abort()
    await asyncio.wait_for(entered.wait(), timeout=2.0)  # kill attempt is now pending/blocked

    wait_task = asyncio.ensure_future(process.wait())
    await asyncio.sleep(0.3)  # the real child has exited naturally by now (its own 0.15s sleep)
    assert not wait_task.done(), (
        "wait() must not settle while the watcher's kill-attempt correction is still pending"
    )

    release.set()  # let the pending (no-effect) kill attempt resolve
    wait_result = await asyncio.wait_for(wait_task, timeout=2.0)
    assert isinstance(wait_result, Ok), (
        f"a pending no-effect kill attempt must not classify Err(aborted): {wait_result!r}"
    )
    assert wait_result.value.exit_code == 0


async def test_wait_awaits_a_pending_kill_attempt_that_eventually_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L12-PY-R004` witness, delayed-`True` counterpart to the test above (per the review: "Add
    both delayed-`False` and delayed-`True` witnesses"). Here the pending kill attempt eventually
    DOES find and kill a live target -- `_kill_cause` must remain `"signal"` and `wait()` must
    classify `Err(aborted)`, proving the fix's `await` (rather than `cancel()`) of the watcher
    task doesn't merely coincidentally pass the no-effect case but correctly preserves a genuine
    kill's classification too. Wraps the REAL `_kill_process_tree` so `release` gates an actual
    OS-level kill of the real long-running child, rather than only simulating one."""
    entered = asyncio.Event()
    release = asyncio.Event()
    real_kill_process_tree = subprocess_module._kill_process_tree

    async def delayed_kill_process_tree(pid: int) -> bool:
        entered.set()
        await release.wait()
        return await real_kill_process_tree(pid)

    monkeypatch.setattr(subprocess_module, "_kill_process_tree", delayed_kill_process_tree)

    sp = LocalSubprocess()
    controller = RunAbortController()
    result = await sp.spawn(
        [PY, "-c", "import time; time.sleep(30)"], SpawnOptions(signal=controller.signal)
    )
    process = result.value  # type: ignore[union-attr]
    await asyncio.sleep(0.02)
    controller.abort()
    # kill attempt is pending; the real kill has not been issued yet
    await asyncio.wait_for(entered.wait(), timeout=2.0)

    wait_task = asyncio.ensure_future(process.wait())
    await asyncio.sleep(0.1)
    assert not wait_task.done(), "wait() must not settle while the real kill is still pending"

    release.set()  # let the pending kill attempt actually run now
    wait_result = await asyncio.wait_for(wait_task, timeout=5.0)
    assert isinstance(wait_result, Err), (
        f"a genuine pending kill must classify aborted: {wait_result!r}"
    )
    assert wait_result.error.code == SubprocessErrorCode.ABORTED


@pytest.mark.skipif(os.name != "nt", reason="exercises the taskkill-spawn OSError branch")
async def test_kill_process_tree_reports_no_target_when_taskkill_itself_fails_to_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_kill_process_tree`'s own `except OSError: return False` branch (e.g. `taskkill.exe`
    missing from `PATH`) -- the helper `Popen(...)` call itself raises rather than the child
    process, and this must be reported the same as "no live target found", not propagate."""

    def fake_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        raise OSError("simulated: taskkill not found")

    monkeypatch.setattr(subprocess_module._subprocess, "Popen", fake_popen)

    killed = await subprocess_module._kill_process_tree(os.getpid())

    assert killed is False


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


async def test_read_chunk_returns_buffered_output_after_wait() -> None:
    """`L12-PY-R007` witness (refined, second review): the earlier fix closed `proc.stdout`'s
    own transport as part of `wait()`'s cleanup (piggybacking on the top-level `proc._transport`
    close), which discarded already-buffered-but-not-yet-read stdout data -- `read_chunk()`
    called AFTER `wait()` has settled returned `Err(pipe_error)` instead of the real payload.
    `wait()` must close only `proc._transport` itself; each stdio stream's OWN separate
    transport (confirmed empirically independent on Windows' ProactorEventLoop) is closed by
    `ReadableStream.read_chunk()` on its own natural EOF, so already-buffered output already
    sitting in the pipe remains fully readable after `wait()` returns."""
    sp = LocalSubprocess()
    result = await sp.spawn(
        [PY, "-c", "import sys; sys.stdout.write('buffered-payload'); sys.exit(0)"],
        SpawnOptions(stdout=StdioMode.PIPED),
    )
    process = result.value  # type: ignore[union-attr]
    wait_result = await process.wait()
    assert isinstance(wait_result, Ok)
    assert process.stdout is not None
    chunks = b""
    while True:
        chunk_result = await process.stdout.read_chunk()
        assert isinstance(chunk_result, Ok), (
            f"reading buffered stdout after wait() must not fail: {chunk_result!r}"
        )
        if chunk_result.value is None:
            break
        chunks += chunk_result.value
    assert chunks == b"buffered-payload"


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
