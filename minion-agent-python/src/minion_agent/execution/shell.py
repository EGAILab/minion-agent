"""`ctx.shell` (`EXEC-004`, spec/execution.md section 5). `DIRECT_PI_PARITY` for the observable
contract, built on TOP of `ctx.subprocess` (`subprocess.py`) rather than reimplementing process
management -- matching the frozen design's own stated architecture.

The exact six-step pre-spawn order (`L12-R010`) and exact timeout boundary (`L12-R011`) are
precise, previously-litigated details -- do not reorder or approximate them. The idle-grace
completion rule (`L12-R007`, `EXIT_STDIO_GRACE_MS = 100`, reset on every post-exit data event) is
this seam's OWN layered concern, built on `ctx.subprocess.wait()`'s simpler exit-only settlement.
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

from ..runtime.signal import RunSignal
from .errors import ShellError, ShellErrorCode
from .filesystem import resolve_local_path
from .result import Err, Ok, Result
from .subprocess import LocalSubprocess, Process, ReadableStream, SpawnOptions, StdioMode

_MAX_TIMEOUT_MS = 2_147_483_647
_EXIT_STDIO_GRACE_S = 0.1


@dataclass(frozen=True, slots=True)
class ShellResult:
    """`exec()`'s own success payload -- present REGARDLESS of `exit_code`; a non-zero exit is
    NOT a `Result` failure at this seam (spec section 5.6)."""

    stdout: str
    stderr: str
    exit_code: int


class Shell(Protocol):
    """`EXEC-004`."""

    cwd: str

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
        timeout: float | None = None,
        signal: RunSignal | None = None,
        on_stdout: Callable[[bytes], None] | None = None,
        on_stderr: Callable[[bytes], None] | None = None,
    ) -> Result[ShellResult, ShellError]: ...
    async def cleanup(self) -> None: ...


def _validate_timeout(timeout: float | None) -> Result[float | None, ShellError]:
    """`L12-R011`'s exact boundary: rejected when `timeout * 1000 > 2_147_483_647`. The largest
    accepted value is `2147483.647` seconds exactly."""
    if timeout is None:
        return Ok(None)
    if not math.isfinite(timeout) or timeout <= 0:
        message = "Invalid timeout: must be a finite positive number of seconds"
        return Err(ShellError(ShellErrorCode.TIMEOUT, message))
    if timeout * 1000 > _MAX_TIMEOUT_MS:
        message = f"Invalid timeout: maximum is {_MAX_TIMEOUT_MS / 1000} seconds"
        return Err(ShellError(ShellErrorCode.TIMEOUT, message))
    return Ok(timeout)


class _Completion:
    """Mirrors pinned Pi's own `waitForChildProcess` state machine (`nodejs.ts` lines ~278-345)
    exactly: once the directly-spawned process exits, arm a 100ms idle-grace timer; any stdio
    data received before it fires resets it; settle on the timer firing OR both streams ending,
    whichever happens first."""

    def __init__(self) -> None:
        self._exited = False
        self._stdout_ended = False
        self._stderr_ended = False
        self._settled = asyncio.Event()
        self._idle_timer: asyncio.TimerHandle | None = None

    def _maybe_finalize_after_exit(self) -> None:
        if self._exited and self._stdout_ended and self._stderr_ended:
            self._settled.set()

    def _arm_idle_timer(self) -> None:
        loop = asyncio.get_running_loop()
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = loop.call_later(_EXIT_STDIO_GRACE_S, self._settled.set)

    def on_data(self) -> None:
        if self._exited:
            self._arm_idle_timer()

    def on_exit(self) -> None:
        self._exited = True
        self._maybe_finalize_after_exit()
        if not self._settled.is_set():
            self._arm_idle_timer()

    def on_stdout_end(self) -> None:
        self._stdout_ended = True
        self._maybe_finalize_after_exit()

    def on_stderr_end(self) -> None:
        self._stderr_ended = True
        self._maybe_finalize_after_exit()

    async def wait_settled(self) -> None:
        await self._settled.wait()
        if self._idle_timer is not None:
            self._idle_timer.cancel()


class LocalShell:
    """The local `ctx.shell` provider (`EXEC-004`/spec section 8) -- `DIRECT_PI_PARITY`,
    genuinely sourced from pinned Pi's own harness-tier reference implementation."""

    __slots__ = ("_active", "_shell_path", "_subprocess", "cwd")

    def __init__(
        self,
        cwd: str | None = None,
        shell_path: str | None = None,
        subprocess_seam: LocalSubprocess | None = None,
    ) -> None:
        self.cwd = cwd if cwd is not None else os.getcwd()
        self._shell_path = shell_path
        self._subprocess = (
            subprocess_seam if subprocess_seam is not None else LocalSubprocess(self.cwd)
        )
        self._active: set[Process] = set()

    async def _resolve_shell(self) -> Result[str, ShellError]:
        """`L12-R010` step 4. `DIRECT_PI_PARITY` for the observable guarantee; discovery
        mechanics are platform-appropriate, not a literal port."""
        if self._shell_path is not None:
            if await asyncio.to_thread(os.path.exists, self._shell_path):
                return Ok(self._shell_path)
            return Err(
                ShellError(
                    ShellErrorCode.SHELL_UNAVAILABLE,
                    f"Custom shell path not found: {self._shell_path}",
                )
            )
        if os.name == "nt":
            candidates: list[str] = []
            program_files = os.environ.get("PROGRAMFILES")
            if program_files:
                candidates.append(os.path.join(program_files, "Git", "bin", "bash.exe"))
            program_files_x86 = os.environ.get("PROGRAMFILES(X86)")
            if program_files_x86:
                candidates.append(os.path.join(program_files_x86, "Git", "bin", "bash.exe"))
            for candidate in candidates:
                if await asyncio.to_thread(os.path.exists, candidate):
                    return Ok(candidate)
            bash_on_path = shutil.which("bash")
            if bash_on_path:
                return Ok(bash_on_path)
            return Err(
                ShellError(
                    ShellErrorCode.SHELL_UNAVAILABLE,
                    "No bash shell found. Install Git for Windows, add bash to PATH, or "
                    "configure an explicit shell path.",
                )
            )
        # POSIX-only: unreachable on Windows (the os.name == "nt" branch above always returns
        # first).
        if await asyncio.to_thread(os.path.exists, "/bin/bash"):  # pragma: no cover
            return Ok("/bin/bash")
        bash_on_path = shutil.which("bash")  # pragma: no cover
        if bash_on_path:  # pragma: no cover
            return Ok(bash_on_path)
        return Ok("sh")  # pragma: no cover -- POSIX never fails purely for lacking bash

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
        timeout: float | None = None,
        signal: RunSignal | None = None,
        on_stdout: Callable[[bytes], None] | None = None,
        on_stderr: Callable[[bytes], None] | None = None,
    ) -> Result[ShellResult, ShellError]:
        # Step 1: pre-aborted signal.
        if signal is not None and signal.aborted:
            return Err(ShellError(ShellErrorCode.ABORTED, "aborted"))
        # Step 2: timeout validation.
        timeout_result = _validate_timeout(timeout)
        if isinstance(timeout_result, Err):
            return timeout_result
        timeout_s = timeout_result.value
        # Step 3: lexical cwd resolution.
        resolved_cwd = resolve_local_path(self.cwd, cwd) if cwd is not None else self.cwd
        # Step 4: shell discovery -- BEFORE the cwd existence check (L12-R010).
        shell_result = await self._resolve_shell()
        if isinstance(shell_result, Err):
            return shell_result
        shell_path = shell_result.value
        # Step 5: cwd existence check.
        if not await asyncio.to_thread(os.path.isdir, resolved_cwd):
            return Err(
                ShellError(
                    ShellErrorCode.SPAWN_ERROR, f"Working directory does not exist: {resolved_cwd}"
                )
            )
        # Step 6: spawn.
        spawn_result = await self._subprocess.spawn(
            [shell_path, "-c", command],
            SpawnOptions(
                cwd=resolved_cwd,
                env=env,
                inherit_env=inherit_env,
                signal=signal,
                stdin=StdioMode.NULL,
                stdout=StdioMode.PIPED,
                stderr=StdioMode.PIPED,
            ),
        )
        if isinstance(spawn_result, Err):
            spawn_error = spawn_result.error
            return Err(
                ShellError(ShellErrorCode.SPAWN_ERROR, spawn_error.message, spawn_error.cause)
            )
        process = spawn_result.value
        self._active.add(process)
        try:
            return await self._run_to_completion(process, timeout_s, on_stdout, on_stderr)
        finally:
            self._active.discard(process)

    async def _run_to_completion(
        self,
        process: Process,
        timeout_s: float | None,
        on_stdout: Callable[[bytes], None] | None,
        on_stderr: Callable[[bytes], None] | None,
    ) -> Result[ShellResult, ShellError]:
        completion = _Completion()
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        callback_error: Exception | None = None
        timed_out = False

        async def pump(
            stream: ReadableStream | None,
            chunks: list[bytes],
            on_chunk: Callable[[bytes], None] | None,
            mark_end: Callable[[], None],
        ) -> None:
            nonlocal callback_error
            if stream is None:  # pragma: no cover -- exec() always spawns piped stdout/stderr
                mark_end()
                return
            while True:
                result = await stream.read_chunk()
                if isinstance(result, Err) or result.value is None:
                    mark_end()
                    return
                chunk = result.value
                chunks.append(chunk)
                completion.on_data()
                if on_chunk is not None:
                    try:
                        on_chunk(chunk)
                    except Exception as exc:
                        if callback_error is None:
                            callback_error = exc
                        await process.terminate()

        async def watch_exit() -> None:
            await process.wait()
            completion.on_exit()

        async def watch_timeout() -> None:
            nonlocal timed_out
            if timeout_s is None:
                return
            await asyncio.sleep(timeout_s)
            timed_out = True
            await process.terminate()

        tasks = [
            asyncio.ensure_future(
                pump(process.stdout, stdout_chunks, on_stdout, completion.on_stdout_end)
            ),
            asyncio.ensure_future(
                pump(process.stderr, stderr_chunks, on_stderr, completion.on_stderr_end)
            ),
            asyncio.ensure_future(watch_exit()),
            asyncio.ensure_future(watch_timeout()),
        ]

        await completion.wait_settled()
        for task in tasks:
            if not task.done():
                task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks, return_exceptions=True)

        wait_result = await process.wait()

        if callback_error is not None:
            return Err(
                ShellError(ShellErrorCode.CALLBACK_ERROR, str(callback_error), callback_error)
            )
        if timed_out:
            return Err(ShellError(ShellErrorCode.TIMEOUT, f"timeout:{timeout_s}"))
        if isinstance(wait_result, Err):
            return Err(ShellError(ShellErrorCode.ABORTED, "aborted"))

        exit_code = wait_result.value.exit_code
        return Ok(
            ShellResult(
                stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
                stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
                exit_code=exit_code if exit_code is not None else 0,
            )
        )

    async def cleanup(self) -> None:
        """`L12-R017`: kills the process tree of every command THIS provider currently has in
        flight and clears its own tracking; best-effort, MUST NOT raise. A cleanup-killed
        command's own `exec()` call settles through this SAME `_run_to_completion` path (its
        `watch_exit()` observes the kill via `process.wait()`, which now preserves a real exit
        code per `L12-R020` -- `terminate()` here never sets an abort signal or fires the
        timeout, so neither of those classifications can apply)."""
        active = list(self._active)
        self._active.clear()
        for process in active:
            await process.terminate()
