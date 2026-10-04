"""`L12-D002`: `Process.wait()` settles on the process's own exit, independent of stdio state
(spec/execution.md section 6) -- also when a descendant still holds an inherited pipe.

asyncio's own `Process.wait()` resolves only after every pipe has disconnected; on Windows a
descendant holding stdout therefore kept the certified `wait()` pending until it let go (3 s in the
witness below), while Linux settled at exit."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import sys
import time
from typing import Any

import pytest

from minion_agent.execution.result import Ok
from minion_agent.execution.subprocess import LocalSubprocess, SpawnOptions

PY = sys.executable

_PARENT = (
    "import os, subprocess, sys\n"
    "subprocess.Popen([sys.executable, '-c', {descendant!r}], stdin=subprocess.DEVNULL,"
    " stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno(),"
    " start_new_session=(os.name != 'nt'))\n"
    "sys.stdout.write('parent\\n'); sys.stdout.flush()\n"
    "os._exit(3)\n"
)


def _parent(descendant: str) -> list[str]:
    return [PY, "-c", _PARENT.format(descendant=descendant)]


async def _read_all(process: object) -> bytes:
    data = b""
    stream = process.stdout  # type: ignore[attr-defined]
    while True:
        chunk = await stream.read_chunk()
        assert isinstance(chunk, Ok)
        if chunk.value is None:
            return data
        data += chunk.value


async def test_wait_settles_at_exit_while_a_descendant_holds_the_pipe() -> None:
    result = await LocalSubprocess().spawn(_parent("import time; time.sleep(3)"), SpawnOptions())
    assert isinstance(result, Ok)
    process = result.value
    started = time.monotonic()
    status = await asyncio.wait_for(process.wait(), 10)
    elapsed = time.monotonic() - started
    assert isinstance(status, Ok) and status.value.exit_code == 3
    assert elapsed < 1.5, f"wait() waited {elapsed:.2f}s for the descendant's pipe"
    await process.terminate()


async def test_reading_continues_after_wait_until_the_descendant_closes() -> None:
    """The read side is untouched by `wait()`: output the descendant writes AFTER the parent's
    exit is still read, then EOF when it lets go."""
    descendant = (
        "import sys, time; time.sleep(0.5); sys.stdout.write('late\\n'); sys.stdout.flush()"
    )
    result = await LocalSubprocess().spawn(_parent(descendant), SpawnOptions())
    assert isinstance(result, Ok)
    process = result.value
    status = await asyncio.wait_for(process.wait(), 10)
    assert isinstance(status, Ok) and status.value.exit_code == 3
    output = await asyncio.wait_for(_read_all(process), 10)
    assert output.replace(b"\r\n", b"\n") == b"parent\nlate\n"


async def test_exit_observed_before_the_hook_is_installed() -> None:
    """A process that has already exited when `Process` is built still settles."""
    from minion_agent.execution import subprocess as module

    proc = await asyncio.create_subprocess_exec(PY, "-c", "pass")
    await proc.wait()
    process = module.Process(proc, None, None, None, None)
    status = await asyncio.wait_for(process.wait(), 5)
    assert isinstance(status, Ok) and status.value.exit_code == 0


async def test_a_cancelled_waiter_does_not_poison_later_waits() -> None:
    """`L12-D002-I001` (Codex's reproducer): cancelling a task suspended in `wait()` cancels that
    caller only; a later `wait()` still returns the real exit code."""
    from minion_agent.execution.subprocess import StdioMode

    result = await LocalSubprocess().spawn(
        [PY, "-c", "import sys; print('ready', flush=True); sys.stdin.buffer.read(1)"],
        SpawnOptions(stdin=StdioMode.PIPED),
    )
    assert isinstance(result, Ok)
    process = result.value
    assert process.stdout is not None and process.stdin is not None
    ready = await process.stdout.read_chunk()
    assert isinstance(ready, Ok) and ready.value is not None
    waiter = asyncio.create_task(process.wait())
    await asyncio.sleep(0)
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter
    await process.stdin.write(b"x")
    await process.stdin.close()
    status = await asyncio.wait_for(process.wait(), 10)
    assert isinstance(status, Ok) and status.value.exit_code == 0
    await _read_all(process)


async def test_wait_then_terminate_disposes_pipes_a_descendant_still_holds() -> None:
    """`L12-D002-I002` (Codex's reproducer): after the parent's exit, `wait()` then `terminate()`
    is a guaranteed-safe disposal even while a descendant still holds stdout/stderr -- the next
    test forces collection after this test's loop has closed, with destructor warnings as
    errors (`L12-PY-R007`)."""
    result = await LocalSubprocess().spawn(
        [
            PY,
            "-c",
            "import os, subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'],"
            " stdin=subprocess.DEVNULL, stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno())\n"
            "os._exit(0)\n",
        ],
        SpawnOptions(),
    )
    assert isinstance(result, Ok)
    process = result.value
    assert isinstance(await process.wait(), Ok)
    await process.terminate()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_collect_after_the_disposal_loop_has_closed() -> None:
    gc.collect()


async def test_terminate_after_exit_also_disposes_a_piped_stdin() -> None:
    from minion_agent.execution.subprocess import StdioMode

    result = await LocalSubprocess().spawn([PY, "-c", "pass"], SpawnOptions(stdin=StdioMode.PIPED))
    assert isinstance(result, Ok)
    process = result.value
    assert isinstance(await process.wait(), Ok)
    await process.terminate()
    assert process.stdin is not None


# ---- ReadableStream.close() (EXEC-012, spec section 16.2) ----

_HOLD = "import sys, time; sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep(30)"


async def _spawn(code: str) -> object:
    result = await LocalSubprocess().spawn([PY, "-c", code], SpawnOptions())
    assert isinstance(result, Ok)
    return result.value


async def test_close_settles_a_pending_read_as_eof_and_later_reads_too() -> None:
    process: Any = await _spawn(_HOLD)
    ready = await process.stdout.read_chunk()
    assert isinstance(ready, Ok) and ready.value is not None
    pending = asyncio.create_task(process.stdout.read_chunk())
    await asyncio.sleep(0.05)
    assert not pending.done()
    await process.stdout.close()
    settled = await asyncio.wait_for(pending, 5)
    assert isinstance(settled, Ok) and settled.value is None
    later = await process.stdout.read_chunk()
    assert isinstance(later, Ok) and later.value is None
    await process.stdout.close()  # repeated close is harmless
    await process.terminate()
    await process.wait()


async def test_close_abandons_output_already_buffered() -> None:
    """`close()` abandons output not yet consumed: data already buffered when it is called is
    not delivered -- the next read is EOF."""
    code = "import sys, time; sys.stdout.write('buffered'); sys.stdout.flush(); time.sleep(30)"
    process: Any = await _spawn(code)
    await asyncio.sleep(0.5)  # the output has arrived in the stream's buffer
    await process.stdout.close()
    after = await process.stdout.read_chunk()
    assert isinstance(after, Ok) and after.value is None
    await process.terminate()
    await process.wait()


async def test_close_never_terminates_the_child_and_leaves_the_sibling_open() -> None:
    code = (
        "import sys, time; sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep(0.5);"
        " sys.stderr.write('still here\\n'); sys.stderr.flush(); time.sleep(0.2)"
    )
    process: Any = await _spawn(code)
    await process.stdout.read_chunk()
    await process.stdout.close()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(process.wait()), 0.2)  # still running
    data = b""
    while True:
        chunk = await asyncio.wait_for(process.stderr.read_chunk(), 5)
        assert isinstance(chunk, Ok)
        if chunk.value is None:
            break
        data += chunk.value
    assert data.replace(b"\r\n", b"\n") == b"still here\n"
    status = await process.wait()
    assert isinstance(status, Ok) and status.value.exit_code == 0


async def test_a_background_writer_meets_a_closed_reader_after_close(tmp_path: Any) -> None:
    """The ordinary OS consequence (Pi's `stream.destroy()` at settlement): after `close()`, a
    descendant's next write fails, so the step after it never runs."""
    marker = tmp_path / "survived-write"
    descendant = (
        "import sys, time, pathlib\ntime.sleep(0.6)\n"
        "try:\n    sys.stdout.write('late\\n' * 1000); sys.stdout.flush()\n"
        f"    pathlib.Path({str(marker)!r}).write_text('x')\nexcept OSError:\n    pass\n"
    )
    process: Any = await _spawn(_PARENT.format(descendant=descendant))
    assert isinstance(await process.wait(), Ok)
    await process.stdout.close()
    await process.stderr.close()
    await asyncio.sleep(1.5)
    assert not marker.exists()


async def test_wait_then_close_disposes_pipes_a_descendant_still_holds() -> None:
    """Disposal under the amended model: the process settled (`wait()`) and every owned piped
    handle released (`close()`) -- no `terminate()`; the next test collects after the loop
    closed."""
    process: Any = await _spawn(_PARENT.format(descendant="import time; time.sleep(2)"))
    assert isinstance(await process.wait(), Ok)
    await process.stdout.close()
    await process.stderr.close()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_collect_after_the_close_disposal_loop_has_closed() -> None:
    gc.collect()


async def test_read_error_after_close_is_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read that fails because the caller closed the stream is EOF, never `pipe_error`."""
    from minion_agent.execution.subprocess import ReadableStream

    class _Failing:
        _transport = None

        async def read(self, n: int) -> bytes:
            stream._closed = True
            raise OSError("closed underneath")

    stream = ReadableStream(_Failing())  # type: ignore[arg-type]
    result = await stream.read_chunk()
    assert isinstance(result, Ok) and result.value is None
