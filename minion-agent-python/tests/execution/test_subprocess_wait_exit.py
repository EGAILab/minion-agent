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
