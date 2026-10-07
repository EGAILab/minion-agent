"""What `find` and `grep` share (spec/tools.md WP-13.4): engine resolution against the execution
world, Pi's full `TruncationResult`, and the engine run -- spawn through `ctx.subprocess`, read both
pipes to EOF and the exit (Pi's `close` event, no idle grace), release the read ends."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...execution import Err
from ...execution.subprocess import Process, ReadableStream, SpawnOptions, Subprocess
from ...execution.world import ExecutionWorldIdentity
from ...runtime.signal import RunSignal
from ._readline import LineSplitter
from .bash import scalar_command
from .paths import BuiltinToolError
from .search_engines import EngineName, Engines, EngineStore, not_available
from .truncate import DEFAULT_MAX_BYTES, Truncation, utf8_len

MAX_SAFE_INTEGER = 9007199254740991


class AbortWindow:
    """Pi's abort-listener lifetime for a search call (CE-L13-WP134-01): an abort counts only while
    the window is open. `close()` is called at engine completion and latches whether the signal had
    fired by then; after it, the signal is no longer consulted. A window over no signal, or opened
    on an already-aborted signal (a listener registered too late never fires), starts closed."""

    __slots__ = ("observed", "open", "signal")

    def __init__(self, signal: RunSignal | None, *, listening: bool = True) -> None:
        self.signal = signal
        self.open = listening and signal is not None and not signal.aborted
        self.observed = False

    def fired(self) -> bool:
        """An abort inside the open window (as a listener would see it now)."""
        return self.open and self.signal is not None and self.signal.aborted

    def close(self) -> None:
        if self.open:
            self.observed = self.observed or self.fired()
            self.open = False


async def resolve_engine(engines: Engines, subprocess: Subprocess, engine: EngineName) -> list[str]:
    """TOOL-038: a certified store is usable only in the LOCAL execution world."""
    if (
        isinstance(engines, EngineStore)
        and subprocess.execution_world != ExecutionWorldIdentity.local()
    ):
        raise not_available(
            engine, f"{subprocess.execution_world.value} (non-local execution world)"
        )
    return await engines.resolve(engine)


def truncation_result(truncation: Truncation) -> dict[str, Any]:
    """Pi's `truncateHead` `TruncationResult`, verbatim key set (`details.truncation`)."""
    return {
        "content": truncation.content,
        "truncated": truncation.truncated,
        "truncatedBy": truncation.truncated_by,
        "totalLines": truncation.total_lines,
        "totalBytes": truncation.total_bytes,
        "outputLines": truncation.output_lines,
        "outputBytes": utf8_len(truncation.content),
        "lastLinePartial": False,
        "firstLineExceedsLimit": truncation.first_line_exceeds_limit,
        "maxLines": MAX_SAFE_INTEGER,
        "maxBytes": DEFAULT_MAX_BYTES,
    }


@dataclass(slots=True)
class EngineRun:
    """One engine process: stdout lines go to `on_line` as they arrive (which may ask to stop the
    engine by returning True); stderr is accumulated as text."""

    process: Process
    stderr: str = ""
    exit_code: int | None = None
    stopped: bool = False
    _stopping: asyncio.Task[None] | None = None

    async def _pump_stdout(
        self, stream: ReadableStream | None, on_line: Callable[[str], bool]
    ) -> None:
        if stream is None:  # pragma: no cover - stdout is always piped
            return
        splitter = LineSplitter()
        while True:
            chunk = await stream.read_chunk()
            data = None if isinstance(chunk, Err) else chunk.value
            lines = splitter.finish() if data is None else splitter.feed(data)
            for line in lines:
                if on_line(line) and not self.stopped:
                    self.stopped = True
                    self._stopping = asyncio.ensure_future(terminate_quietly(self.process))
            if data is None:
                return

    async def _pump_stderr(self, stream: ReadableStream | None) -> None:
        if stream is None:  # pragma: no cover - stderr is always piped
            return
        parts: list[bytes] = []
        while True:
            chunk = await stream.read_chunk()
            if isinstance(chunk, Err) or chunk.value is None:
                break
            parts.append(chunk.value)
        self.stderr = b"".join(parts).decode("utf-8", "replace")

    async def run(
        self, on_line: Callable[[str], bool], on_complete: Callable[[], None] | None = None
    ) -> None:
        """Exit AND EOF on both pipes -- ENGINE COMPLETION, Pi's child `close` -- then
        `on_complete` (synchronously, before anything else is awaited); only then the cleanup:
        joining a stop request's termination acknowledgement and releasing the read ends
        (spec/execution.md section 16). Pi's `stopChild` is synchronous and its `close` handler
        never waits for a kill to be acknowledged, so that join must not delay completion
        (CE-L13-WP134-01, targeted closure 1)."""
        try:
            await asyncio.gather(
                self._pump_stdout(self.process.stdout, on_line),
                self._pump_stderr(self.process.stderr),
            )
            status = await self.process.wait()
            self.exit_code = None if isinstance(status, Err) else status.value.exit_code
            if on_complete is not None:
                on_complete()
        finally:
            if self._stopping is not None:
                await self._stopping
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None:
                    await stream.close()


async def spawn_engine(subprocess: Subprocess, argv: list[str], failure: str) -> EngineRun:
    """Spawn with Pi's `spawn` defaults: inherited environment, stdin ignored, stdout/stderr piped.
    A spawn failure is Pi's `Failed to run <engine>: <cause>` with the Layer-12 message as cause.
    Every argument takes its scalar form first (each unpaired surrogate becomes U+FFFD), as Node
    does at the OS boundary -- the WP-13.3 argv projection rule (`WP133-AUD-R001`)."""
    spawned = await subprocess.spawn(
        [scalar_command(arg) for arg in argv], SpawnOptions(inherit_env=True)
    )
    if isinstance(spawned, Err):
        raise BuiltinToolError(f"{failure}: {spawned.error.message}")
    return EngineRun(spawned.value)


async def terminate_quietly(process: Process) -> None:
    with contextlib.suppress(Exception):
        await process.terminate()
