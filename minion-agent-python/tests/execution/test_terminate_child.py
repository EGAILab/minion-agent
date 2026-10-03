"""`EXEC-011` (WP-12.E5): `Process.terminate_child()`, spec/execution.md section 16.

POSIX witnesses run on a POSIX host (the Python gate's Linux container run); Windows witnesses run
on Windows. Each child prints `ready` once its `SIGTERM` disposition is installed, so a request is
never racing the child's own startup."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from minion_agent.execution.result import Err, Ok
from minion_agent.execution.subprocess import LocalSubprocess, Process, SpawnOptions, StdioMode
from minion_agent.runtime.signal import RunAbortController

PY = sys.executable
POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX SIGTERM semantics")
WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Windows TerminateProcess semantics")

_READY = "import sys; sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
_HANG = "import time\nwhile True: time.sleep(0.05)\n"


def _handler(body: str) -> str:
    return (
        "import signal, sys, time, os\ndef h(signum, frame):\n"
        f"{body}\nsignal.signal(signal.SIGTERM, h)\n"
    )


async def _spawn_ready(code: str, signal_controller: RunAbortController | None = None) -> Process:
    options = SpawnOptions(
        stdout=StdioMode.PIPED, signal=signal_controller.signal if signal_controller else None
    )
    result = await LocalSubprocess().spawn([PY, "-c", code], options)
    assert isinstance(result, Ok)
    process = result.value
    assert process.stdout is not None
    first = await process.stdout.read_chunk()
    assert isinstance(first, Ok)
    assert first.value is not None and first.value.startswith(b"ready")
    return process


async def _exit_code(process: Process) -> int | None:
    result = await asyncio.wait_for(process.wait(), 10)
    assert isinstance(result, Ok)
    return result.value.exit_code


# ---- POSIX: witnesses 1-5 ----


@POSIX
async def test_posix_default_disposition_is_signal_termination() -> None:
    process = await _spawn_ready(_READY + _HANG)
    await process.terminate_child()
    assert await _exit_code(process) is None


@POSIX
@pytest.mark.parametrize("code", [0, 7])
async def test_posix_handled_sigterm_reports_the_childs_own_code(code: int) -> None:
    process = await _spawn_ready(_handler(f"    os._exit({code})") + _READY + _HANG)
    await process.terminate_child()
    assert await _exit_code(process) == code


@POSIX
async def test_posix_delayed_exit_is_not_completed_by_the_request() -> None:
    process = await _spawn_ready(_handler("    time.sleep(0.5); os._exit(0)") + _READY + _HANG)
    await process.terminate_child()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(process.wait()), 0.25)
    assert await _exit_code(process) == 0


@POSIX
async def test_posix_ignored_sigterm_then_own_exit() -> None:
    """A truly ignored signal (`SIG_IGN`, no handler), then an exit the child schedules itself."""
    ignore = "import os, signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    process = await _spawn_ready(ignore + _READY + "time.sleep(0.6)\nos._exit(0)\n")
    await process.terminate_child()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(process.wait()), 0.25)
    assert await _exit_code(process) == 0


def _with_descendant(marker: Path) -> str:
    descendant = (
        f"import time, pathlib; time.sleep(0.7); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    return (
        f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
        + _READY
        + _HANG
    )


@POSIX
async def test_posix_direct_child_only_descendant_in_group_survives(tmp_path: Path) -> None:
    """Witness 5: the descendant shares the child's process group (the child leads its own
    session), so its survival shows the signal went to the PID, not the group."""
    child_marker = tmp_path / "after-terminate-child"
    process = await _spawn_ready(_with_descendant(child_marker))
    await process.terminate_child()
    assert await _exit_code(process) is None
    await asyncio.sleep(1.2)
    assert child_marker.exists()

    tree_marker = tmp_path / "after-terminate"
    twin = await _spawn_ready(_with_descendant(tree_marker))
    await twin.terminate()
    await _exit_code(twin)
    await asyncio.sleep(1.2)
    assert not tree_marker.exists()


@POSIX
async def test_posix_single_delivery_on_repeated_calls() -> None:
    """Witness 8: the child counts `SIGTERM`s and reports the count when it exits. The second call
    is made only after the child ACKNOWLEDGES the first (`got`), so a resent signal could not
    coalesce with a still-pending first one (standard POSIX signals are not queued) -- otherwise
    a resending implementation would also count 1."""
    handler = "    global n\n    n += 1\n    sys.stdout.write('got\\n'); sys.stdout.flush()"
    code = _handler(handler) + "n = 0\n" + _READY + "time.sleep(1.0)\nos._exit(n)\n"
    process = await _spawn_ready(code)
    await process.terminate_child()
    assert process.stdout is not None
    ack = await asyncio.wait_for(process.stdout.read_chunk(), 5)
    assert isinstance(ack, Ok)
    assert ack.value is not None and ack.value.startswith(b"got")
    await process.terminate_child()
    assert await _exit_code(process) == 1


@POSIX
async def test_posix_terminate_child_is_not_a_cause_claim() -> None:
    """A later abort of the spawn-time signal still claims `SIGNAL` (section 16.3)."""
    controller = RunAbortController()
    process = await _spawn_ready(_handler("    pass") + _READY + _HANG, controller)
    await process.terminate_child()
    controller.abort()
    result = await asyncio.wait_for(process.wait(), 10)
    assert isinstance(result, Err)


@POSIX
async def test_posix_later_terminate_still_tree_kills() -> None:
    """A later explicit `terminate()` after a request still claims `EXPLICIT` and hard-kills: the
    child ignores the `SIGTERM`, then `SIGKILL` ends it with no exit code (section 16.3)."""
    ignore = "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    process = await _spawn_ready(ignore + _READY + _HANG)
    await process.terminate_child()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(process.wait()), 0.25)
    await process.terminate()
    assert await _exit_code(process) is None


# ---- Windows: witness 7 ----


@WINDOWS
async def test_windows_effective_termination_reports_no_code() -> None:
    process = await _spawn_ready(_READY + _HANG)
    await process.terminate_child()
    assert await _exit_code(process) is None


@WINDOWS
async def test_windows_direct_child_only_descendant_survives(tmp_path: Path) -> None:
    marker = tmp_path / "after-terminate-child"
    process = await _spawn_ready(_with_descendant(marker))
    await process.terminate_child()
    assert await _exit_code(process) is None
    await asyncio.sleep(1.2)
    assert marker.exists()


@WINDOWS
async def test_windows_terminate_still_reports_its_real_code() -> None:
    """`terminate()` is unchanged (`L12-R020`): its Windows outcome keeps the OS code."""
    process = await _spawn_ready(_READY + _HANG)
    await process.terminate()
    assert await _exit_code(process) == 1


# ---- Both platforms: no-op cases (witness 8) ----


async def test_no_op_after_exit_keeps_the_real_code() -> None:
    result = await LocalSubprocess().spawn([PY, "-c", "import sys; sys.exit(3)"])
    assert isinstance(result, Ok)
    process = result.value
    assert await _exit_code(process) == 3
    await process.terminate_child()
    assert await _exit_code(process) == 3


async def test_no_op_after_terminate() -> None:
    process = await _spawn_ready(_READY + _HANG)
    await process.terminate()
    await process.terminate_child()
    code = await _exit_code(process)
    assert code == (1 if os.name == "nt" else None)


async def test_repeated_calls_do_not_raise() -> None:
    process = await _spawn_ready(_READY + _HANG)
    await process.terminate_child()
    await process.terminate_child()
    await _exit_code(process)
    await process.terminate_child()


@WINDOWS
async def test_windows_exited_but_unobserved_is_not_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process exited but asyncio has not observed it yet: `TerminateProcess` fails with
    `ERROR_ACCESS_DENIED`, CPython sets the real code, and the termination is not effective
    (libuv's `UV_ESRCH`). asyncio's observation is held back by reporting `returncode` unset."""
    result = await LocalSubprocess().spawn([PY, "-c", "import sys; sys.exit(4)"])
    assert isinstance(result, Ok)
    process = result.value
    popen = process._proc._transport.get_extra_info("subprocess")  # type: ignore[attr-defined]
    await asyncio.to_thread(popen.wait)  # the OS process has exited
    popen.returncode = None  # as if not yet observed: Popen.terminate must discover the exit itself
    monkeypatch.setattr(type(process._proc), "returncode", property(lambda self: None))
    await process.terminate_child()
    monkeypatch.undo()
    assert process._terminated_by_child_request is False
    assert popen.returncode == 4
    assert await _exit_code(process) == 4


@WINDOWS
async def test_windows_terminate_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    process = await _spawn_ready(_READY + _HANG)
    popen = process._proc._transport.get_extra_info("subprocess")  # type: ignore[attr-defined]

    def fail() -> None:
        raise OSError("refused")

    monkeypatch.setattr(popen, "terminate", fail)
    await process.terminate_child()
    assert process._terminated_by_child_request is False
    await process.terminate()
    await _exit_code(process)


@WINDOWS
async def test_windows_missing_popen_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    process = await _spawn_ready(_READY + _HANG)
    transport = process._proc._transport  # type: ignore[attr-defined]
    monkeypatch.setattr(transport, "get_extra_info", lambda name, default=None: None)
    await process.terminate_child()
    assert process._terminated_by_child_request is False
    monkeypatch.undo()
    await process.terminate()
    await _exit_code(process)
