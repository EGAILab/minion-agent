"""The `bash` built-in tool (`TOOL-034`, `TOOL-035`; pinned Pi `core/tools/bash.ts`), spec/tools.md
WP-13.3, over the execution world: `ctx.subprocess` runs the shell, `ctx.fs` answers existence and
holds the full-output file (Pi's `BashOperations`, `MINION_ARCHITECTURAL_MAPPING`, Owner Q3).

Execution order (spec "Execution, in order"): the spawn environment; timeout validation; the abort
check; shell selection; the cwd check; spawn; the run (output intake, timer, settlement);
classification -- the signal first, then the timeout, then the exit code; the result.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from typing import Any

from ...execution import Err, FileSystem, FsErrorCode, Platform
from ...execution.errors import SubprocessErrorCode
from ...execution.subprocess import (
    Process,
    ReadableStream,
    SpawnOptions,
    StdioMode,
    Subprocess,
)
from ...execution.world import ExecutionWorldError, validate
from ...llm import TextBlock
from ...runtime.signal import RunSignal
from ..definition import ToolDefinition, ToolExecutionContext
from ..result import ToolResult
from ._js import number_to_string
from .bash_output import OutputAccumulator
from .bash_shell import ShellConfig, ShellNotFoundError, select_shell
from .environment import compose_spawn_environment
from .paths import BuiltinToolError, cause
from .truncate import DEFAULT_MAX_BYTES, format_size

BASH_DESCRIPTION = (
    "Execute a bash command in the current working directory. Returns stdout and stderr. Output "
    "is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output "
    "is saved to a temp file. Optionally provide a timeout in seconds."
)

BASH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {
            "type": "number",
            "description": "Timeout in seconds (optional, no default timeout)",
        },
    },
    "required": ["command"],
}

MAX_TIMEOUT_MS = 2_147_483_647
EXIT_STDIO_GRACE_S = 0.1
"""Pi's `EXIT_STDIO_GRACE_MS`: settle 100 ms after exit with no further output."""

SESSION_ENVIRONMENT_NAMES = (
    "MINION_SESSION_ID",
    "MINION_SESSION_FILE",
    "MINION_PROVIDER",
    "MINION_MODEL",
    "MINION_REASONING_LEVEL",
)
"""Owner Q1: removed by exact spelling from every spawn, then re-injected from the call's
`ToolExecutionContext` when exposed."""

FULL_OUTPUT_PREFIX = "minion-bash-"
FULL_OUTPUT_SUFFIX = ".log"


class IncompatibleExecutionWorldError(Exception):
    """Activation failure: `ctx.fs` and `ctx.subprocess` address different execution worlds
    (spec/execution.md section 7). Carries the normative `ExecutionWorldError`."""

    def __init__(self, error: ExecutionWorldError) -> None:
        super().__init__("bash requires ctx.fs and ctx.subprocess in the same execution world")
        self.error = error


class _Aborted(Exception):
    """Aborted before a process ran (Pi's thrown `Error("aborted")`, nothing collected)."""


class _AbortedWith(Exception):
    """Aborted after the run: classified by the signal first (Pi's `Error("aborted")`)."""

    def __init__(self, run: _Run) -> None:
        super().__init__("aborted")
        self.run = run


class _TimedOutWith(Exception):
    """Timed out (Pi's `Error("timeout:<seconds>")`)."""

    def __init__(self, run: _Run) -> None:
        super().__init__("timeout")
        self.run = run


def session_environment(context: ToolExecutionContext | None) -> dict[str, str]:
    """Owner Q1 injection: `MINION_SESSION_ID` always; `MINION_SESSION_FILE` when present;
    `MINION_PROVIDER` and `MINION_MODEL` only when both are present; `MINION_REASONING_LEVEL` when
    present and non-empty (`"off"` is present)."""
    if context is None:
        return {}
    injected = {"MINION_SESSION_ID": context.session_id}
    if context.session_file:
        injected["MINION_SESSION_FILE"] = context.session_file
    if context.provider and context.model:
        injected["MINION_PROVIDER"] = context.provider
        injected["MINION_MODEL"] = context.model
    if context.reasoning_level:
        injected["MINION_REASONING_LEVEL"] = context.reasoning_level
    return injected


def resolve_timeout_ms(timeout: float | None) -> float | None:
    """Pi's `resolveTimeoutMs`: `None` for no timeout; validated binary64 milliseconds."""
    if timeout is None:
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        raise BuiltinToolError("Invalid timeout: must be a finite number of seconds")
    timeout_ms = timeout * 1000
    if timeout_ms > MAX_TIMEOUT_MS:
        raise BuiltinToolError(
            f"Invalid timeout: maximum is {number_to_string(MAX_TIMEOUT_MS / 1000)} seconds"
        )
    return timeout_ms


def scheduled_delay_ms(timeout_ms: float) -> int:
    """Node's timer normalization (`WP133-CON-R003`): `Timeout` clamps below 1 ms to 1, and
    `insert` truncates the fraction -- `max(1, trunc(ms))` whole milliseconds."""
    return max(1, math.trunc(timeout_ms))


def scalar_command(command: str) -> str:
    """`WP133-AUD-R001`: the command's scalar form before either transport -- each unpaired
    surrogate becomes U+FFFD, a valid pair (even held as two code points) is kept."""
    return command.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


async def _check_cwd(fs: FileSystem, platform: Platform, cwd: str) -> None:
    """`CE-WP133-01`: POSIX `access(F_OK)` follows (`probe_dir_entry`); Windows' does not
    (`file_info`). `not_supported` is a disclosed prerequisite error, never absence."""
    operation = "file_info" if platform is Platform.WINDOWS else "probe_dir_entry"
    result = await (fs.file_info(cwd) if platform is Platform.WINDOWS else fs.probe_dir_entry(cwd))
    if isinstance(result, Err):
        if result.error.code == FsErrorCode.NOT_SUPPORTED:
            raise BuiltinToolError(f"bash requires a filesystem provider that supports {operation}")
        raise BuiltinToolError(
            f"Working directory does not exist: {cwd}\nCannot execute bash commands."
        )


class _Run:
    """One command's run: output intake into the accumulator and the full-output file, the
    timeout timer, and settlement (spec "Run and settlement")."""

    def __init__(self, fs: FileSystem, process: Process) -> None:
        self.fs = fs
        self.process = process
        self.output = OutputAccumulator()
        self.backlog: list[bytes] = []
        self.full_output_path: str | None = None
        self.file_error: FsErrorCode | None = None
        self.timed_out = False
        self._intake = asyncio.Lock()
        self._data_after_exit = asyncio.Event()
        self._exited = False

    async def _persist(self, chunk: bytes | None) -> None:
        """Pi's temp-file rule: once the stream needs the file, open it, write every earlier raw
        chunk, then every accepted chunk -- the RAW bytes, through `ctx.fs`."""
        if self.file_error is not None:
            return
        if self.full_output_path is None:
            if chunk is not None:
                self.backlog.append(chunk)
            if not self.output.should_use_temp_file:
                return
            created = await self.fs.create_temp_file(FULL_OUTPUT_PREFIX, FULL_OUTPUT_SUFFIX)
            if isinstance(created, Err):
                self.file_error = created.error.code
                return
            self.full_output_path = created.value
            pending, self.backlog = b"".join(self.backlog), []
        elif chunk is not None:
            pending = chunk
        else:
            return
        if pending:
            written = await self.fs.append_file(self.full_output_path, pending)
            if isinstance(written, Err):
                self.file_error = written.error.code

    async def accept(self, chunk: bytes) -> None:
        async with self._intake:
            self.output.append(chunk)
            await self._persist(chunk)
            if self.file_error is not None:
                # Minion-defined: the run cannot keep its full output -- end it (Pi: an unhandled
                # stream error).
                await self.process.terminate()
        if self._exited:
            self._data_after_exit.set()

    async def pump(self, stream: ReadableStream | None) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read_chunk()
            if isinstance(chunk, Err) or chunk.value is None:
                return  # EOF, or a pipe failure ending this stream's intake
            await self.accept(chunk.value)

    async def settle(self) -> int | None:
        """Waits for exit, then for both streams' EOF or 100 ms without further data (re-armed by
        each chunk); never on `wait()` alone. Returns the exit code (`None` if killed)."""
        pumps = asyncio.ensure_future(
            asyncio.gather(self.pump(self.process.stdout), self.pump(self.process.stderr))
        )
        try:
            status = await self.process.wait()
            self._exited = True
            while not pumps.done():
                self._data_after_exit.clear()
                more = asyncio.ensure_future(self._data_after_exit.wait())
                waiters: set[asyncio.Future[Any]] = {pumps, more}
                done, _ = await asyncio.wait(
                    waiters, timeout=EXIT_STDIO_GRACE_S, return_when=asyncio.FIRST_COMPLETED
                )
                more.cancel()
                if not done:
                    break  # 100 ms after exit with no further data
        finally:
            pumps.cancel()  # pending reads are cancelled; later chunks are dropped
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pumps
        async with self._intake:
            self.output.finish()
            await self._persist(None)
        if isinstance(status, Err):
            return None
        return status.value.exit_code


def _format_output(run: _Run, empty_text: str) -> tuple[str, dict[str, Any]]:
    """Pi's `formatOutput`: the content, or `empty_text`, then a truncation notice."""
    truncation = run.output.snapshot()
    text = truncation["content"] or empty_text
    if not truncation["truncated"]:
        return text, {}
    path = run.full_output_path
    end = truncation["totalLines"]
    start = end - truncation["outputLines"] + 1
    if truncation["lastLinePartial"]:
        text += (
            f"\n\n[Showing last {format_size(truncation['outputBytes'])} of line {end} "
            f"(line is {format_size(run.output.last_line_bytes)}). Full output: {path}]"
        )
    elif truncation["truncatedBy"] == "lines":
        text += f"\n\n[Showing lines {start}-{end} of {end}. Full output: {path}]"
    else:
        text += (
            f"\n\n[Showing lines {start}-{end} of {end} "
            f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {path}]"
        )
    return text, {"truncation": truncation, "fullOutputPath": path}


def _with_status(text: str, status: str) -> str:
    """Pi's `appendStatus`."""
    return f"{text}\n\n{status}" if text else status


def create_bash_tool(
    fs: FileSystem,
    subprocess: Subprocess,
    *,
    shell_path: str | None = None,
    expose_session_environment: bool = True,
) -> ToolDefinition:
    """The `bash` tool over one execution world (Owner Q3 factory). Raises
    `IncompatibleExecutionWorldError` when `fs` and `subprocess` address different worlds."""
    world = validate([("fs", fs.execution_world), ("subprocess", subprocess.execution_world)])
    if isinstance(world, Err):
        raise IncompatibleExecutionWorldError(world.error)

    async def run_command(
        command: str, timeout: float | None, signal: RunSignal | None, env: dict[str, str]
    ) -> tuple[_Run, int | None]:
        timeout_ms = resolve_timeout_ms(timeout)
        if signal is not None and signal.aborted:
            raise _Aborted
        try:
            config: ShellConfig = await select_shell(fs, subprocess, shell_path)
        except ShellNotFoundError as error:
            raise BuiltinToolError(str(error)) from error
        await _check_cwd(fs, subprocess.platform, subprocess.cwd)
        projected = scalar_command(command)
        via_stdin = config.transport == "stdin"
        argv = (
            [config.shell, *config.args] if via_stdin else [config.shell, *config.args, projected]
        )
        spawned = await subprocess.spawn(
            argv,
            SpawnOptions(
                env=env,
                inherit_env=False,
                stdin=StdioMode.PIPED if via_stdin else StdioMode.NULL,
                signal=signal,
            ),
        )
        if isinstance(spawned, Err):
            if spawned.error.code == SubprocessErrorCode.ABORTED:
                raise _Aborted
            raise BuiltinToolError(f"Failed to start the shell {config.shell}")
        process = spawned.value
        run = _Run(fs, process)
        background: list[asyncio.Future[Any]] = []
        if via_stdin and process.stdin is not None:
            stdin = process.stdin

            async def feed() -> None:  # never blocks the timer, abort or output monitoring
                await stdin.write(projected.encode("utf-8"))
                await stdin.close()

            background.append(asyncio.ensure_future(feed()))
        timer: asyncio.TimerHandle | None = None
        if timeout_ms is not None:

            def fire() -> None:
                run.timed_out = True
                background.append(asyncio.ensure_future(process.terminate()))

            timer = asyncio.get_running_loop().call_later(
                scheduled_delay_ms(timeout_ms) / 1000, fire
            )
        try:
            exit_code = await run.settle()
        finally:
            if timer is not None:
                timer.cancel()
            for task in background:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if run.file_error is not None:
            raise BuiltinToolError(f"Cannot write the full-output file: {cause(run.file_error)}")
        if signal is not None and signal.aborted:
            raise _AbortedWith(run)
        if run.timed_out:
            raise _TimedOutWith(run)
        return run, exit_code

    async def execute(
        tool_call_id: str,
        arguments: dict[str, Any],
        signal: RunSignal | None,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        command: str = arguments["command"]
        timeout: float | None = arguments.get("timeout")
        injected = session_environment(context) if expose_session_environment else {}
        env = compose_spawn_environment(
            subprocess.base_env(), remove=SESSION_ENVIRONMENT_NAMES, inject=injected
        )
        try:
            run, exit_code = await run_command(command, timeout, signal, env)
        except _Aborted:
            raise BuiltinToolError("Command aborted") from None
        except _AbortedWith as aborted_run:
            text, _ = _format_output(aborted_run.run, "")
            raise BuiltinToolError(_with_status(text, "Command aborted")) from None
        except _TimedOutWith as timed_out_run:
            text, _ = _format_output(timed_out_run.run, "")
            status = f"Command timed out after {number_to_string(timeout)} seconds"  # type: ignore[arg-type]
            raise BuiltinToolError(_with_status(text, status)) from None
        text, details = _format_output(run, "(no output)")
        if exit_code is not None and exit_code != 0:
            raise BuiltinToolError(_with_status(text, f"Command exited with code {exit_code}"))
        return ToolResult(
            tool_call_id=tool_call_id,
            content=(TextBlock(text=text),),
            tool_name="bash",
            details=details,
        )

    return ToolDefinition(
        name="bash",
        label="bash",
        description=BASH_DESCRIPTION,
        parameters=BASH_PARAMETERS,
        execute=execute,
        wants_signal=True,
        wants_context=True,
    )
