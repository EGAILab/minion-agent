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
from .environment import EnvSnapshot, Platform, host_platform, local_baseline
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
    # `WP12E4-C002` = B: the baseline is `base_env()`'s own source, read now -- the live native
    # environment on Windows, `os.environ` on POSIX (spec/execution.md section 15.4).
    base = dict(local_baseline())
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


async def _issue_kill(pid: int) -> _subprocess.Popen[bytes] | None:
    """Kills the WHOLE process tree/group where the platform supports it -- the SAME mechanism
    `ctx.shell`'s own tree-kill guarantee (`EXEC-004`) relies on, since it is built on this
    primitive. POSIX: process-group `SIGKILL` via a negative PID (requires the process to have
    been spawned into its own session, `start_new_session=True`), falling back to a single-PID
    kill. Windows: `taskkill /T /F`. Fire-and-forget, matching pinned Pi's own `killProcessTree`
    (`nodejs.ts:253-276`), which is `void` and never awaited by its own callers either.

    Returns the Windows `taskkill` helper's own `Popen` handle for a SEPARATE, later,
    best-effort confirmation/cleanup step (`_confirm_kill`, avoiding a leaked process handle --
    `L12-PY-R007`); `None` on POSIX or when spawning the helper itself failed. This return value
    is NEVER consulted for causal classification -- `L12-PY-R004`'s independently-agreed
    convergence checkpoint (`CE-L12-PY-01-01`, `minion-agent-docs#121` @
    `2db656c01126bfb775d1fe453241e191ed78b2f0`) settled the shared cause-classification model
    as a deterministic first-claim state machine: the claim happens at the moment `_watch_signal`
    OBSERVES the signal fired while the process was still believed running, independent of
    whether the subsequent kill attempt here later succeeds, fails to find a live target, or
    fails to even be issued to the OS at all -- exactly matching pinned Rust's own certified
    `subprocess.rs` (`monitor_child`'s CAS happens before `kill_process_tree` is ever called).
    See `_watch_signal`'s own docstring for the classification side of this."""
    if os.name == "nt":
        try:
            return _subprocess.Popen(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.DEVNULL,
                stdin=_subprocess.DEVNULL,
            )
        except OSError:
            return None
    # POSIX-only: unreachable on Windows (the os.name == "nt" branch above always returns
    # first) -- os.killpg/signal.SIGKILL also aren't in Windows' own typeshed subset.
    try:  # pragma: no cover
        os.killpg(pid, _os_signal.SIGKILL)  # type: ignore[attr-defined]
    except OSError:  # pragma: no cover
        with suppress(OSError):
            os.kill(pid, _os_signal.SIGKILL)  # type: ignore[attr-defined]
    return None  # pragma: no cover


async def _confirm_kill(helper: _subprocess.Popen[bytes] | None) -> None:
    """Best-effort confirmation/cleanup of the Windows `taskkill` helper `_issue_kill` spawned
    -- fully decoupled from causal classification, which `_issue_kill`'s own return already
    settled. `L12-PY-R007`: a fire-and-forget `Popen` with no `wait()`/`close()` leaks the
    helper process's own handle (a `ResourceWarning` at GC time) and, transitively, its
    `DEVNULL` pipe transports; `Popen.wait()` blocks, so it runs off the event loop
    (`asyncio.to_thread`)."""
    if helper is not None:
        await asyncio.to_thread(_wait_and_settle_helper, helper)


async def _issue_and_confirm_kill(pid: int) -> None:
    """Combines `_issue_kill` and `_confirm_kill` for the fire-and-forget background dispatch
    from `_watch_signal`, whose own critical path no longer needs either step's outcome --
    `L12-PY-R004`'s classification already happened at observation time (see `_watch_signal`'s
    own docstring). `terminate()` still awaits the two steps separately, since ITS caller
    explicitly awaits confirmation/cleanup before returning."""
    await _confirm_kill(await _issue_kill(pid))


def _posix_terminate_child(pid: int) -> None:  # pragma: no cover
    """`EXEC-011` (WP-12.E5), POSIX: one `SIGTERM` to the direct child, only while the child still
    owns its PID (`WP12E5-I001`).

    asyncio's child watcher reaps the child, then publishes the exit to the transport through a
    queued callback, so `Process.returncode is None` does not prove the PID is still ours: a
    released (and possibly reused) PID must never be signalled. Order matters:

    1. `pidfd_open(pid)` first, where available (Linux), pinning the process the PID names now.
    2. `waitid(P_PID, pid, WEXITED | WNOHANG | WNOWAIT)` -- it never reaps. `ChildProcessError`:
       already reaped, the PID is not ours, do nothing. A result: exited (a zombie), nothing to
       request. `None`: still running and unreaped -- and since reaping is irreversible, the
       pidfd opened in step 1 names this same child.
    3. Signal through the pidfd (`pidfd_send_signal`), which is safe even if the child is reaped
       afterwards (`ESRCH`).

    Without pidfd support (Linux before 5.3, other POSIX systems) `os.kill` follows step 2 in the
    same synchronous block; asyncio then reaps on a watcher THREAD, so a microsecond window
    between step 2 and the signal remains there -- disclosed, and outside the certified hosts
    (Linux with pidfd; Windows). `Popen.send_signal` is avoided: its `poll()` can reap the child
    behind asyncio's own watcher."""
    pidfd: int | None = None
    with suppress(AttributeError, OSError):
        pidfd = os.pidfd_open(pid)  # type: ignore[attr-defined]
    try:
        try:
            exited = os.waitid(  # type: ignore[attr-defined]
                os.P_PID,  # type: ignore[attr-defined]
                pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,  # type: ignore[attr-defined]
            )
        except ChildProcessError:
            return
        if exited is not None:
            return
        with suppress(OSError):
            if pidfd is not None:
                _os_signal.pidfd_send_signal(pidfd, _os_signal.SIGTERM)  # type: ignore[attr-defined]
            else:
                os.kill(pid, _os_signal.SIGTERM)
    finally:
        if pidfd is not None:
            os.close(pidfd)


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
        "_terminate_child_called",
        "_terminated_by_child_request",
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
        self._terminate_child_called = False
        self._terminated_by_child_request = False
        """`EXEC-011` (WP-12.E5), Windows only: an effective `terminate_child()` -- the direct
        `TerminateProcess` found the process still running. `wait()` then reports no exit code,
        as Node does (`exit_signal = SIGTERM`), not the OS-synthesized `1`."""
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
        or awaits this (`L12-PY-R004`, refined a sixth time); see its own docstring."""
        if spawn_signal is not None:
            self._watcher_task = asyncio.ensure_future(self._watch_signal(spawn_signal))

    async def _watch_signal(self, signal: RunSignal) -> None:
        """Polls the ORIGINAL spawn-time signal for the lifetime of the process, terminating it
        the moment the signal fires (cooperative/poll-based, matching `RunSignal`'s own design).

        `L12-PY-R004` (deterministic first-claim model, independently agreed at the
        `CE-L12-PY-01-01` checkpoint -- `minion-agent-docs#121` @
        `2db656c01126bfb775d1fe453241e191ed78b2f0`): records `_kill_cause = "signal"` THE
        INSTANT this loop observes the signal fired while its own most recent liveness check
        (the `while` condition itself) still found the process running -- BEFORE the kill is
        even dispatched, not after it is confirmed issued. The subsequent kill attempt's own
        success, failure to find a live target, or failure to even reach the OS does NOT
        retroactively rewrite this claim -- classification is a function of OBSERVATION alone,
        matching pinned Rust's own certified `subprocess.rs` (`monitor_child`'s CAS happens
        before `kill_process_tree` is ever called). `_issue_kill`+`_confirm_kill` are then
        dispatched together as a single background, UN-AWAITED task (`add_done_callback`
        suppresses an unexpected exception there from raising an "exception was never
        retrieved" warning, matching `filesystem.py`'s own `_race_signal` convention) -- this
        loop's own critical path never awaits either, so `Process.wait()` has nothing
        kill-related to ever wait for, preserving the contract's own "`wait()` settles on the
        process's own exit alone" rule unconditionally."""
        while self._proc.returncode is None:
            if signal.aborted:
                if self._kill_cause is None:
                    self._kill_cause = "signal"
                kill_task = asyncio.ensure_future(_issue_and_confirm_kill(self.pid))
                kill_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                return
            await asyncio.sleep(_SIGNAL_POLL_INTERVAL_S)

    async def wait(self) -> Result[ExitStatus, SubprocessError]:
        """Settles on the PROCESS's own exit alone -- independent of stdio state, AND
        independent of any signal-triggered kill's own confirmation (`L12-PY-R004`): this awaits
        ONLY `self._proc.wait()`, never `_watch_signal`'s own task or `_confirm_kill`'s own
        completion, so a slow (or permanently stuck) kill-confirmation helper can never delay
        settlement past the target process's own already-observed exit -- the targeted closure
        review's own exact reproduction of an earlier revision that awaited the watcher task
        here. `Err(aborted)` when `_kill_cause` was recorded (`"signal"`, set the instant
        `_watch_signal` OBSERVED the signal fired while the process was still running -- see
        `_watch_signal`'s own docstring; not re-derived from the signal's current state, which
        would misclassify an unrelated later abort); otherwise `Ok(ExitStatus{...})`, whether the
        process exited on its own or was killed via an explicit `terminate()` (`L12-R020`'s
        conditional exit-code preservation)."""
        if self._wait_result is not None:
            return self._wait_result
        async with self._wait_lock:
            if self._wait_result is not None:
                return self._wait_result
            returncode = await self._proc.wait()
            # `L12-PY-R004` (refined a sixth time): deliberately does NOT touch
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
                #
                # `EXEC-011` (WP-12.E5): an effective Windows `terminate_child()` is reported
                # without a code, as Node reports it (`status: null, signal: SIGTERM`).
                exit_code = (
                    returncode
                    if returncode >= 0 and not self._terminated_by_child_request
                    else None
                )
                result = Ok(ExitStatus(exit_code=exit_code))
            self._wait_result = result
            return result

    async def terminate(self) -> None:
        """Best-effort, MUST NOT raise, idempotent. Kills the whole process tree. Matches
        `_watch_signal`'s own claim-then-issue ordering (`L12-PY-R004`, deterministic
        first-claim model): `_kill_cause` is recorded THE INSTANT this method is entered
        (first-wins, before the kill is even dispatched), never gated on whether the
        subsequent OS-level kill attempt succeeds. Unlike `_watch_signal`, this method's OWN
        caller explicitly awaits it, so it also awaits confirmation/cleanup before returning."""
        if self._terminate_called:
            return
        self._terminate_called = True
        if self._kill_cause is None:
            self._kill_cause = "explicit"
        helper = await _issue_kill(self.pid)
        await _confirm_kill(helper)

    async def terminate_child(self) -> None:
        """`EXEC-011` (WP-12.E5, `spec/execution.md` section 16). A termination REQUEST to the
        directly spawned process only -- Node's `uv_process_kill(process, SIGTERM)`, which pinned
        Pi's `spawnSync` lookup interruption sends. Best-effort, MUST NOT raise, idempotent.

        POSIX: one `SIGTERM` to the direct child -- never `SIGKILL`, never the process group (the
        child leads its own session, so its descendants share the group and are left alone) --
        and never to a PID the child no longer owns (`_posix_terminate_child`, `WP12E5-I001`).
        Windows: `TerminateProcess(handle, 1)` on the direct process (the transport's
        `Popen.terminate`); no tree, no job.

        A no-op when the process has already exited, when this was already called (Node's
        `spawn_sync` `Kill()` sends once), or after `terminate()`. Not a cause claim: section 6's
        first-claim classification is untouched, so a later signal abort is still `aborted`.
        `wait()` reports the child's own final outcome -- a handled `SIGTERM` exit 0 is `0`."""
        if (
            self._terminate_child_called
            or self._terminate_called
            or self._proc.returncode is not None
        ):
            return
        self._terminate_child_called = True
        if os.name == "nt":
            popen = self._proc._transport.get_extra_info("subprocess")  # type: ignore[attr-defined]
            if popen is None or popen.returncode is not None:
                return
            try:
                popen.terminate()
            except OSError:
                return
            # `Popen.terminate` maps the `ERROR_ACCESS_DENIED` of an already-exited process to its
            # real code (`returncode` set); an effective termination leaves `returncode` unset --
            # libuv's `UV_ESRCH` versus `exit_signal = SIGTERM`.
            self._terminated_by_child_request = popen.returncode is None
            return
        # POSIX-only: unreachable on Windows (the branch above always returns first).
        _posix_terminate_child(self.pid)  # pragma: no cover

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

    # `WP-12.E4` (`EXEC-010`): the execution world's family, provider-declared and constant for
    # the provider's lifetime -- read-only (`WP12E4-I004`).
    @property
    def platform(self) -> Platform: ...

    # `WP-12.E4`: a read-only snapshot of exactly what an `inherit_env=True` spawn would inherit
    # now, before any overlay (spec/execution.md section 15.3).
    def base_env(self) -> EnvSnapshot: ...

    async def spawn(
        self, argv: Sequence[str], options: SpawnOptions | None = None
    ) -> Result[Process, SubprocessError]: ...


class LocalSubprocess:
    """The local `ctx.subprocess` provider (`EXEC-005`/spec section 8) -- `MINION_EXTENSION`,
    matching `EXEC-005`'s own `intentional divergence` disposition (no Pi seam exists for it to
    be direct parity with)."""

    __service_name__: str = "subprocess"

    __slots__ = ("_platform", "cwd", "execution_world")

    def __init__(
        self, cwd: str | None = None, execution_world: ExecutionWorldIdentity | None = None
    ) -> None:
        self.cwd = cwd if cwd is not None else os.getcwd()
        self.execution_world = (
            execution_world if execution_world is not None else ExecutionWorldIdentity.local()
        )
        # `WP-12.E4`: a local provider declares its host's family.
        self._platform = host_platform()

    def base_env(self) -> EnvSnapshot:
        """`WP-12.E4`: read now from the same source `inherit_env=True` uses (C002)."""
        return EnvSnapshot(local_baseline(), self._platform)

    @property
    def platform(self) -> Platform:
        """Read-only: a local provider declares its host's family once (`WP12E4-I004`)."""
        return self._platform

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
