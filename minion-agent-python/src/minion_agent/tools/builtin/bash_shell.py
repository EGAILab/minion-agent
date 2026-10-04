"""`bash` shell selection (`TOOL-034`; pinned Pi `utils/shell.ts` `getShellConfig`,
`findBashOnPath`, `getBashShellConfig`, `isLegacyWslBashPath`), spec/tools.md WP-13.3 "Shell
selection" and "The lookup" -- the scoping v4 matrix over the execution world (`ctx.fs` existence,
`ctx.subprocess` platform/baseline/lookup), never the host.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import dataclass
from typing import Any, Literal

from ...execution import Err, FileSystem, Platform
from ...execution.subprocess import Process, ReadableStream, SpawnOptions, Subprocess

LOOKUP_TIMEOUT_S = 5.0
LOOKUP_OUTPUT_BUDGET = 1024 * 1024
"""Node `spawnSync`'s default `maxBuffer`: one budget over stdout and stderr together."""

_JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003"
    "\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)
"""ECMAScript `String.prototype.trim`'s set (WhiteSpace and LineTerminator) -- U+FEFF included,
which is how a lookup's leading BOM disappears (Python's `str.strip()` would keep it)."""

_LINE_SPLIT = re.compile("\r?\n")
# JS /^[a-z]:\\windows\\(?:system32|sysnative)\\bash\.exe$/ without flags: `$` is end of input.
_LEGACY_WSL_BASH = re.compile(r"^[a-z]:\\windows\\(?:system32|sysnative)\\bash\.exe\Z")


@dataclass(frozen=True, slots=True)
class ShellConfig:
    shell: str
    args: tuple[str, ...]
    transport: Literal["argv", "stdin"]


class ShellNotFoundError(Exception):
    """Pi's `getShellConfig` errors, surfaced verbatim as the tool's error text."""


def js_trim(text: str) -> str:
    return text.strip(_JS_WHITESPACE)


def is_legacy_wsl_bash_path(path: str) -> bool:
    """Pi's `isLegacyWslBashPath`: `/` -> `\\`, then ECMAScript `toLowerCase`. Only characters that
    lower-case to ASCII can affect this ASCII-only match (U+212A KELVIN SIGN -> "k" in both), and
    CPython's `str.lower` agrees with ECMAScript on all of them."""
    return _LEGACY_WSL_BASH.match(path.replace("/", "\\").lower()) is not None


def shell_config(path: str) -> ShellConfig:
    """Pi's `getBashShellConfig`: legacy WSL bash reads the command from stdin (`-s`)."""
    if is_legacy_wsl_bash_path(path):
        return ShellConfig(path, ("-s",), "stdin")
    return ShellConfig(path, ("-c",), "argv")


async def _exists(fs: FileSystem, path: str) -> bool:
    """`existsSync(path)` -> `ctx.fs.probe_dir_entry(path)` is Ok (`CE-WP133-01`); any error is
    "not found"."""
    return not isinstance(await fs.probe_dir_entry(path), Err)


async def close_streams(process: Process) -> None:
    """Releases both read ends (`ReadableStream.close()`, spec/execution.md section 16) -- Pi's
    `stream.destroy()` / `CloseStdioPipes()`. Never signals the process."""
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            await stream.close()


async def lookup(subprocess: Subprocess, argv: list[str]) -> str | None:
    """`findBashOnPath`'s `spawnSync` (spec "The lookup"; `WP133-CON-R005`/`R006`, `CE-WP133-02`,
    `DIV-001`). Returns the decoded stdout when the final exit status is 0, else `None`.

    Collection ends at exit plus EOF on both pipes (no idle grace), at the combined 1 MiB budget
    (the crossing chunk kept), or at the 5000 ms timer. An interruption of an UNEXITED lookup
    calls the certified tree `terminate()` and takes whatever `wait()` reports; an exited lookup
    keeps its status and nothing is terminated. The interruption itself never decides the outcome.
    """
    spawned = await subprocess.spawn(argv, SpawnOptions(inherit_env=True))
    if isinstance(spawned, Err):
        return None
    process = spawned.value
    stdout_chunks: list[bytes] = []
    total = 0
    over_budget = asyncio.Event()

    async def pump(stream: ReadableStream | None, keep: list[bytes] | None) -> None:
        nonlocal total
        if stream is None:  # pragma: no cover - both pipes are requested
            return
        while True:
            chunk = await stream.read_chunk()
            if isinstance(chunk, Err) or chunk.value is None:
                return
            if keep is not None:
                keep.append(chunk.value)
            total += len(chunk.value)
            if total > LOOKUP_OUTPUT_BUDGET:
                over_budget.set()
                return

    exited = asyncio.ensure_future(process.wait())
    pumps = asyncio.ensure_future(
        asyncio.gather(pump(process.stdout, stdout_chunks), pump(process.stderr, None))
    )
    budget = asyncio.ensure_future(over_budget.wait())
    finished: set[asyncio.Future[Any]] = {exited, pumps}
    completion = asyncio.ensure_future(asyncio.wait(finished))
    try:
        first: set[asyncio.Future[Any]] = {completion, budget}
        await asyncio.wait(first, timeout=LOOKUP_TIMEOUT_S, return_when="FIRST_COMPLETED")
        if not exited.done():
            # Interrupted while the lookup is still running: the certified hard kill (DIV-001).
            await process.terminate()
        status = await exited
    finally:
        for task in (pumps, budget, completion):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await close_streams(process)  # collection finished or interrupted (section 16.4)
    if isinstance(status, Err) or status.value.exit_code != 0:
        return None
    stdout = b"".join(stdout_chunks).decode("utf-8", "replace")  # `Buffer#toString`: BOM kept
    return stdout or None


async def select_shell(
    fs: FileSystem, subprocess: Subprocess, shell_path: str | None
) -> ShellConfig:
    """The scoping v4 matrix. Raises `ShellNotFoundError` with Pi's text."""
    if shell_path:
        if await _exists(fs, shell_path):
            return shell_config(shell_path)
        raise ShellNotFoundError(f"Custom shell path not found: {shell_path}")
    if subprocess.platform is Platform.WINDOWS:
        environment = subprocess.base_env()
        candidates: list[str] = []
        for name in ("ProgramFiles", "ProgramFiles(x86)"):
            value = environment.get(name)
            if value:
                candidates.append(f"{value}\\Git\\bin\\bash.exe")
        for candidate in candidates:
            if await _exists(fs, candidate):
                return shell_config(candidate)
        found = await lookup(subprocess, ["where", "bash.exe"])
        if found is not None:
            first = _LINE_SPLIT.split(js_trim(found))[0]
            if first and await _exists(fs, first):
                return shell_config(first)
        searched = "\n".join(f"  {candidate}" for candidate in candidates)
        raise ShellNotFoundError(
            "No bash shell found. Options:\n"
            "  1. Install Git for Windows: https://git-scm.com/download/win\n"
            "  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n"
            "  3. Set shellPath in settings.json\n\n"
            f"Searched Git Bash in:\n{searched}"
        )
    if await _exists(fs, "/bin/bash"):
        return shell_config("/bin/bash")
    found = await lookup(subprocess, ["which", "bash"])
    if found is not None:
        first = _LINE_SPLIT.split(js_trim(found))[0]
        if first:
            return shell_config(first)
    return ShellConfig("sh", ("-c",), "argv")
