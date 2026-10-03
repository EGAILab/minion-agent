"""`L12-D002`: `Process.wait()` settles on the process's own exit, independent of stdio state
(spec/execution.md section 6) -- also when a descendant still holds an inherited pipe.

asyncio's own `Process.wait()` resolves only after every pipe has disconnected; on Windows a
descendant holding stdout therefore kept the certified `wait()` pending until it let go (3 s in the
witness below), while Linux settled at exit."""

from __future__ import annotations

import asyncio
import sys
import time

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
