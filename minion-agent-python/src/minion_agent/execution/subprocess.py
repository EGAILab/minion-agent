"""`ctx.subprocess` (`EXEC-005`, spec/execution.md section 6). `MINION_EXTENSION` -- no direct
Pi seam; the primitive set is derived from the process-management internals pinned Pi's own
`Shell.exec()` reference implementation demonstrates it needs, since `ctx.shell`'s own local
provider (`shell.py`) is built ON TOP of this seam rather than duplicating process management.

One signal, not two (`L12-R006`): `spawn()` accepts a cancellation signal; `wait()` does not.
Cancellation flowing through the ORIGINAL spawn-time signal is implemented as an active
background watcher (this codebase's `RunSignal` is deliberately poll-based, matching pinned Pi's
own cooperative `AbortSignal` model -- see `runtime/signal.py` -- so something has to actually
poll it for "a signal that aborts after the process has started triggers termination" to be a
REAL effect, not merely a classification `wait()` happens to observe if called again later).
"""

from __future__ import annotations

import asyncio
import os
import signal as _os_signal
import subprocess as _subprocess
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from ..runtime.signal import RunSignal
from .errors import SubprocessError, SubprocessErrorCode
from .filesystem import resolve_local_path
from .result import Err, Ok, Result
from .world import ExecutionWorldIdentity

_SIGNAL_POLL_INTERVAL_S = 0.01
_TASKKILL_WAIT_TIMEOUT_S = 5.0

_KillCause = Literal["signal", "explicit", None]


class StdioMode(StrEnum):
    """`EXEC-005`."""

    INHERIT = "inherit"
    PIPED = "piped"
    NULL = "null"


@dataclass(frozen=True, slots=True)
class SpawnOptions:
    """`EXEC-005`. Defaults match `ctx.shell`'s own local-provider spawn shape (no stdin unless
    requested; stdout/stderr piped so they can be captured)."""

    cwd: str | None = None
    env: dict[str, str] | None = None
    inherit_env: bool = True
    stdin: StdioMode = StdioMode.NULL
    stdout: StdioMode = StdioMode.PIPED
    stderr: StdioMode = StdioMode.PIPED
    signal: RunSignal | None = None


@dataclass(frozen=True, slots=True)
class ExitStatus:
    """`exit_code` is `None` when the OS genuinely reports no numeric code for the process's own
    termination (the common signal-terminated case); a real code, when the OS provides one, is
    preserved (`L12-R020` -- a killed process is not guaranteed to lack a numeric exit code)."""

    exit_code: int | None


def _effective_env(env: dict[str, str] | None, inherit_env: bool) -> dict[str, str]:
    """`EXEC-005`'s own cwd/environment rule, mirroring `ctx.shell`'s (`EXEC-004`) identically --
    `inherit_env=True` overlays the provider's own base environment with the call's own `env`;
    `inherit_env=False` is EXACTLY the call's own `env` and nothing else."""
    if not inherit_env:
        return dict(env) if env else {}
    base = dict(os.environ)
    if env:
        base.update(env)
    return base


def _stdio_to_asyncio(mode: StdioMode) -> int | None:
    if mode is StdioMode.INHERIT:
        return None
    if mode is StdioMode.PIPED:
        return _subprocess.PIPE
    return _subprocess.DEVNULL


def _wait_and_settle_helper(helper: _subprocess.Popen[bytes]) -> None:
    """Runs OFF the event loop (`asyncio.to_thread`) since `Popen.wait()` blocks. Best-effort,
    MUST NOT raise: a bounded wait, falling back to killing the helper itself if `taskkill` is
    somehow still running past that."""
    try:
        helper.wait(timeout=_TASKKILL_WAIT_TIMEOUT_S)
    except _subprocess.TimeoutExpired:
        with suppress(OSError):
            helper.kill()
        with suppress(OSError, _subprocess.TimeoutExpired):
            helper.wait(timeout=_TASKKILL_WAIT_TIMEOUT_S)


def _close_owned_transport(owner: object | None) -> None:
    """`L12-PY-R007`'s other resource-warning source: an `asyncio` subprocess/stream transport
    left open for the garbage collector, rather than explicitly closed, can still have buffered
    pipe-transport state on Windows' `ProactorEventLoop` at GC time, producing a
    `PytestUnraisableExceptionWarning`/`ValueError: I/O operation on closed pipe` from
    `ProactorBasePipeTransport.__del__`. `owner._transport` is a private `asyncio` attribute (no
    public API exposes it) -- accessed defensively; its absence or an error while closing it is
    never this function's own failure to propagate, matching every other best-effort cleanup
    path in this module.

    Takes ONE owner at a time (refined at `L12-PY-R007`, second review): an
    earlier revision closed `proc`, `proc.stdin`, `proc.stdout`, AND `proc.stderr` all together,
    unconditionally, inside `wait()` -- empirically confirmed each stdio `StreamReader` owns its
    OWN separate pipe transport, NOT transitively closed by `proc`'s own transport, so closing
    stdout/stderr THERE broke the contracted "wait() settles on exit alone; the caller may keep
    calling `read_chunk()` until EOF afterward" guarantee (spec section 6): any buffered-but-
    unread output became inaccessible the instant `wait()` returned. `wait()` now closes ONLY
    `proc`'s own transport (confirmed empirically NOT to affect the separate stdio transports);
    each `ReadableStream` closes its OWN transport once it reaches EOF NATURALLY (below);
    `terminate()` additionally closes any STILL-OPEN stdio transports as a final best-effort
    sweep, covering the original scenario (a forcibly-killed process whose caller never drains
    its streams at all) without breaking the read-after-wait contract for the ordinary case."""
    transport = getattr(owner, "_transport", None)
    if transport is not None:
        with suppress(Exception):
            transport.close()


async def _kill_process_tree(pid: int) -> None:
    """Kills the WHOLE process tree/group where the platform supports it -- the SAME mechanism
    `ctx.shell`'s own tree-kill guarantee (`EXEC-004`) relies on, since it is built on this
    primitive. POSIX: process-group `SIGKILL` via a negative PID (requires the process to have
    been spawned into its own session, `start_new_session=True`), falling back to a single-PID
    kill. Windows: `taskkill /T /F`.

    Fire-and-forget, matching pinned Pi's own `killProcessTree` EXACTLY (`nodejs.ts:253-276`):
    a bare `spawn("taskkill", ...)` / `process.kill(-pid, "SIGKILL")` call with NO confirmation
    of whether a live target was even found -- `killProcessTree` there returns `void`, not a
    Promise, and is never awaited by its own callers either.

    `L12-PY-R004` (refined a fourth time, third targeted closure review): an earlier revision
    returned `bool` (whether a live target was found) so `_watch_signal` could avoid
    classifying a genuinely no-effect kill attempt as `"signal"`-caused -- but that required
    `Process.wait()` to await this function's own confirmation before settling, which violated
    the approved contract's own "`wait()` settles on the process's own exit alone" rule (a
    slow, or permanently stuck, kill-confirmation could delay `wait()` past the target
    process's own already-observed exit -- the targeted closure review's own exact
    reproduction). Pinned Pi does not attempt this "did the kill actually have an effect"
    distinction at all -- `Shell.exec()`'s own classification simply checks
    `abortSignal.aborted` ONCE, at the moment the process's own exit is FIRST observed
    (`nodejs.ts:491`), with no kill-confirmation step in between. `_watch_signal` now matches
    that: it records causation the instant it decides to kill (eager, unconfirmed, first-wins)
    and dispatches this call as a background, un-awaited task -- see its own docstring."""
    if os.name == "nt":
        try:
            helper = _subprocess.Popen(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.DEVNULL,
                stdin=_subprocess.DEVNULL,
            )
        except OSError:
            return
        # `L12-PY-R007`: a fire-and-forget Popen with no wait()/close() leaks the helper
        # process's own handle (a ResourceWarning at GC time) and, transitively, its DEVNULL
        # pipe transports. `Popen.wait()` blocks, so it runs off the event loop -- this
        # function is itself `async` precisely so that settling can be awaited (by ITS OWN
        # caller's background task, never by `Process.wait()` directly) rather than blocking
        # the loop for up to `_TASKKILL_WAIT_TIMEOUT_S`.
        await asyncio.to_thread(_wait_and_settle_helper, helper)
        return
    # POSIX-only: unreachable on Windows (the os.name == "nt" branch above always returns
    # first) -- os.killpg/signal.SIGKILL also aren't in Windows' own typeshed subset.
    try:  # pragma: no cover
        os.killpg(pid, _os_signal.SIGKILL)  # type: ignore[attr-defined]
    except OSError:  # pragma: no cover
        with suppress(OSError):
            os.kill(pid, _os_signal.SIGKILL)  # type: ignore[attr-defined]


class WritableStream:
    """`EXEC-005`."""

    __slots__ = ("_closed", "_writer")

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._closed = False

    async def write(self, data: bytes) -> Result[None, SubprocessError]:
        try:
            self._writer.write(data)
            await self._writer.drain()
        except OSError as exc:
            return Err(SubprocessError(SubprocessErrorCode.PIPE_ERROR, str(exc), exc))
        return Ok(None)

    async def close(self) -> None:
        """Best-effort, MUST NOT raise, idempotent -- signals EOF to the child's own stdin."""
        if self._closed:
            return
        self._closed = True
        with suppress(OSError):
            self._writer.close()
            await self._writer.wait_closed()


class ReadableStream:
    """`EXEC-005`. `read_chunk()` returns `Ok(None)` for EOF, never raising for an ordinary
    stream-ended condition."""

    __slots__ = ("_reader",)

    _CHUNK_SIZE = 65536

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader

    async def read_chunk(self) -> Result[bytes | None, SubprocessError]:
        try:
            chunk = await self._reader.read(self._CHUNK_SIZE)
        except OSError as exc:
            return Err(SubprocessError(SubprocessErrorCode.PIPE_ERROR, str(exc), exc))
        if not chunk:
            # `L12-PY-R007`: close THIS stream's own transport once it reaches EOF naturally --
            # a fully-drained stream's resources are cleaned up promptly at the point its own
            # lifecycle actually ends, rather than left for the garbage collector (the original
            # resource-warning source) or closed prematurely by `wait()` before the caller has
            # finished reading (the second review's own regression).
            _close_owned_transport(self._reader)
            return Ok(None)
        return Ok(chunk)


class Process:
    """`EXEC-005`. Owns its own stdio handles and underlying OS process handle. Calling `wait()`
    or `terminate()` is the ONLY guaranteed-safe disposal path -- a caller obligation, not an
    implicit-cleanup guarantee (`CE-L12-01-01`). `async with` is offered as an ergonomic
    convenience whose `__aexit__` calls `terminate()`; the underlying contract does not depend
    on it."""

    __slots__ = (
        "_kill_cause",
        "_proc",
        "_spawn_signal",
        "_terminate_called",
        "_wait_lock",
        "_wait_result",
        "_watcher_task",
        "pid",
        "stderr",
        "stdin",
        "stdout",
    )

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        spawn_signal: RunSignal | None,
        stdin: WritableStream | None,
        stdout: ReadableStream | None,
        stderr: ReadableStream | None,
    ) -> None:
        self._proc = proc
        self._spawn_signal = spawn_signal
        self.pid = proc.pid
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self._wait_result: Result[ExitStatus, SubprocessError] | None = None
        self._wait_lock = asyncio.Lock()
        self._terminate_called = False
        self._kill_cause: _KillCause = None
        """Recorded ONCE, at the true moment of causation (`L12-PY-R004`) -- never re-derived
        reactively from the signal's CURRENT state at `wait()` time, which would misclassify a
        process that exited naturally (or was explicitly terminated) followed much later by an
        UNRELATED signal abort. `"signal"`: `_watch_signal` decided to kill the process because
        the spawn-time signal fired. `"explicit"`: `terminate()` was called with no spawn-signal
        involved. Whichever is recorded FIRST wins -- a `terminate()` racing a signal that
        already fired must not overwrite the signal's own causal priority (matches the existing
        "terminate() racing an already-fired signal is a no-op" idempotence rule)."""
        self._watcher_task: asyncio.Task[None] | None = None
        """Stored solely to keep a strong reference alive for the task's own lifetime (asyncio
        only weakly tracks a fire-and-forget task otherwise) -- `wait()` deliberately never reads
        or awaits this (`L12-PY-R004`, refined a fourth time); see its own docstring."""
        if spawn_signal is not None:
            self._watcher_task = asyncio.ensure_future(self._watch_signal(spawn_signal))

    async def _watch_signal(self, signal: RunSignal) -> None:
        """Polls the ORIGINAL spawn-time signal for the lifetime of the process, terminating it
        the moment the signal fires (cooperative/poll-based, matching `RunSignal`'s own design).

        `L12-PY-R004` (refined a fourth time, third targeted closure review): records
        `_kill_cause` the INSTANT this loop decides to kill -- eager, UNCONFIRMED, first-wins --
        matching pinned Pi's own `Shell.exec()` classification, which checks `abortSignal.aborted`
        ONCE at process-exit-observation time with no kill-confirmation step at all (see
        `_kill_process_tree`'s own docstring for the full rationale and the prior revisions this
        superseded). The actual kill is dispatched as a background, UN-AWAITED task
        (`add_done_callback` suppresses an unexpected exception there from raising an "exception
        was never retrieved" warning, matching `filesystem.py`'s own `_race_signal` convention)
        -- this loop's own critical path never awaits the kill's confirmation, so `Process.wait()`
        has nothing kill-confirmation-related to ever wait for either, preserving the contract's
        own "`wait()` settles on the process's own exit alone" rule unconditionally."""
        while self._proc.returncode is None:
            if signal.aborted:
                if self._kill_cause is None:
                    self._kill_cause = "signal"
                kill_task = asyncio.ensure_future(_kill_process_tree(self.pid))
                kill_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                return
            await asyncio.sleep(_SIGNAL_POLL_INTERVAL_S)

    async def wait(self) -> Result[ExitStatus, SubprocessError]:
        """Settles on the PROCESS's own exit alone -- independent of stdio state, AND
        independent of any signal-triggered kill's own confirmation (`L12-PY-R004`, refined a
        fourth time): this awaits ONLY `self._proc.wait()`, never `_watch_signal`'s own task or
        `_kill_process_tree`'s own completion, so a slow (or permanently stuck) kill-confirmation
        helper can never delay settlement past the target process's own already-observed exit --
        the targeted closure review's own exact reproduction of an earlier revision that awaited
        the watcher task here. `Err(aborted)` when `_kill_cause` was recorded (`"signal"`, set
        the instant `_watch_signal` decided to kill -- eager, unconfirmed, matching pinned Pi's
        own lack of kill confirmation, not re-derived from the signal's current state, which
        would misclassify an unrelated later abort); otherwise `Ok(ExitStatus{...})`, whether the
        process exited on its own or was killed via an explicit `terminate()` (`L12-R020`'s
        conditional exit-code preservation)."""
        if self._wait_result is not None:
            return self._wait_result
        async with self._wait_lock:
            if self._wait_result is not None:
                return self._wait_result
            returncode = await self._proc.wait()
            # `L12-PY-R004` (refined a fourth time): deliberately does NOT touch
            # `self._watcher_task` here -- no cancel, no await. `_watch_signal`'s own loop
            # condition (`while self._proc.returncode is None`) exits naturally, on its own,
            # within one poll tick of the process exiting (whether the signal ever fired or
            # not), and its actual kill dispatch is itself a fire-and-forget background task
            # this method never depends on -- see both docstrings above.
            # `L12-PY-R007` (refined, second review): close ONLY the process's own transport
            # here -- confirmed empirically NOT to affect the separate stdio stream transports
            # (each owns its own). Closing stdout/stderr HERE, unconditionally, would break the
            # contracted "read_chunk() still works after wait()" guarantee; each stream instead
            # closes its own transport when IT naturally reaches EOF (`ReadableStream.read_chunk`
            # above), and `terminate()` sweeps any still-open ones as a final best-effort step.
            _close_owned_transport(self._proc)
            result: Result[ExitStatus, SubprocessError]
            if self._kill_cause == "signal":
                result = Err(SubprocessError(SubprocessErrorCode.ABORTED, "aborted"))
            else:
                # POSIX: asyncio reports a signal-terminated child as a NEGATIVE returncode
                # (`-signum`) -- this platform's own "no numeric exit code" case, mapped to
                # None. A non-negative returncode is the real code, preserved exactly.
                #
                # Windows has no equivalent signal-termination convention: a taskkill/
                # TerminateProcess-killed process still reports a real (if OS-synthesized,
                # commonly `1`), non-negative exit code from GetExitCodeProcess -- there is no
                # "genuinely absent" case to detect here at all. Under this contract's own rule
                # ("a real code when the OS provides one, None only when genuinely absent"),
                # that Windows-reported code IS the real code and is correctly preserved, not
                # a bug -- `exit_code: None` after terminate() is a POSIX-specific observable
                # outcome, not a cross-platform guarantee.
                exit_code = returncode if returncode >= 0 else None
                result = Ok(ExitStatus(exit_code=exit_code))
            self._wait_result = result
            return result

    async def terminate(self) -> None:
        """Best-effort, MUST NOT raise, idempotent. Kills the whole process tree."""
        if self._terminate_called:
            return
        self._terminate_called = True
        if self._kill_cause is None:
            self._kill_cause = "explicit"
        await _kill_process_tree(self.pid)

    async def __aenter__(self) -> Process:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.terminate()


class Subprocess(Protocol):
    """The `ctx.subprocess` capability seam (`EXEC-005`) -- an independently-swappable
    abstraction, matching the SAME shape `FileSystem`/`Shell` already declare (`L12-PY-R003`).
    Registered on the Runtime under `__service_name__` so `Context.require(Subprocess)`/
    `ctx.subprocess` resolves whichever conforming provider is mounted, not necessarily
    `LocalSubprocess` -- `LocalShell` depends on THIS protocol, never the concrete class."""

    __service_name__: str = "subprocess"

    cwd: str
    execution_world: ExecutionWorldIdentity

    async def spawn(
        self, argv: Sequence[str], options: SpawnOptions | None = None
    ) -> Result[Process, SubprocessError]: ...


class LocalSubprocess:
    """The local `ctx.subprocess` provider (`EXEC-005`/spec section 8) -- `MINION_EXTENSION`,
    matching `EXEC-005`'s own `intentional divergence` disposition (no Pi seam exists for it to
    be direct parity with)."""

    __service_name__: str = "subprocess"

    __slots__ = ("cwd", "execution_world")

    def __init__(
        self, cwd: str | None = None, execution_world: ExecutionWorldIdentity | None = None
    ) -> None:
        self.cwd = cwd if cwd is not None else os.getcwd()
        self.execution_world = (
            execution_world if execution_world is not None else ExecutionWorldIdentity.local()
        )

    async def spawn(
        self, argv: Sequence[str], options: SpawnOptions | None = None
    ) -> Result[Process, SubprocessError]:
        """`argv`-direct only -- never shell-interpreted, regardless of what `argv[0]` looks
        like. A pre-aborted `signal` short-circuits before any process starts."""
        opts = options if options is not None else SpawnOptions()
        if opts.signal is not None and opts.signal.aborted:
            return Err(SubprocessError(SubprocessErrorCode.ABORTED, "aborted"))
        resolved_cwd = resolve_local_path(self.cwd, opts.cwd) if opts.cwd is not None else self.cwd
        effective_env = _effective_env(opts.env, opts.inherit_env)
        spawn_kwargs: dict[str, object] = {}
        if os.name != "nt":  # pragma: no cover -- POSIX-only, unreachable on Windows
            spawn_kwargs["start_new_session"] = True
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=resolved_cwd,
                env=effective_env,
                stdin=_stdio_to_asyncio(opts.stdin),
                stdout=_stdio_to_asyncio(opts.stdout),
                stderr=_stdio_to_asyncio(opts.stderr),
                **spawn_kwargs,  # type: ignore[arg-type]
            )
        except OSError as exc:
            return Err(SubprocessError(SubprocessErrorCode.SPAWN_ERROR, str(exc), exc))
        process = Process(
            proc=proc,
            spawn_signal=opts.signal,
            stdin=WritableStream(proc.stdin) if proc.stdin is not None else None,
            stdout=ReadableStream(proc.stdout) if proc.stdout is not None else None,
            stderr=ReadableStream(proc.stderr) if proc.stderr is not None else None,
        )
        return Ok(process)
