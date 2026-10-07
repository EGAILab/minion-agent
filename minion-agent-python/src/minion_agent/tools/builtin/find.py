"""The `find` built-in tool (`TOOL-036`; pinned Pi `core/tools/find.ts`, default implementation),
spec/tools.md WP-13.4 "`find`": Pi's wrapper over the pinned `fd 10.4.2` (`TOOL-038`), reached
through the execution world -- `ctx.fs` for the repository walk, `ctx.subprocess` for the engine.

Cross-entry order is the engine's (`ENGINE_DEFINED_UNSPECIFIED`, Owner Q2): nothing here sorts.
`DIV-002` keeps a Windows full-path `**/` component's zero-directory meaning.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from typing import Any

from ...execution import Err, FileSystem, Platform, Subprocess
from ...execution.world import ExecutionWorldError, validate
from ...llm import TextBlock
from ...runtime.signal import RunSignal
from ..definition import ToolDefinition
from ..result import ToolResult
from ._js import number_to_string
from ._node_path import NodePath
from ._search import (
    MAX_SAFE_INTEGER,
    AbortWindow,
    EngineRun,
    resolve_engine,
    spawn_engine,
    terminate_quietly,
    truncation_result,
)
from ._signal import _ABANDONED, _POLL_INTERVAL_S, _discard
from .bash_shell import js_trim
from .paths import BuiltinToolError, aborted, preprocess_path
from .search_engines import Engines
from .truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 1000

FIND_DESCRIPTION = (
    "Search for files by glob pattern. Returns matching file paths relative to the search "
    f"directory. Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} results or "
    f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
)

FIND_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": (
                "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
            ),
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (default: current directory)",
        },
        "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
    },
    "required": ["pattern"],
}

_SEPARATOR_CLASS = "[/\\\\]"  # Pi's String.raw`[/\\]`: either separator


type _Token = tuple[str, str]  # (kind, source text): class | open | comma | close | star | lit


def _pi_windows_rewrite(pattern: str) -> str:
    return pattern.replace("/", _SEPARATOR_CLASS)


def _lex(text: str) -> list[_Token]:
    """Glob text as the pinned fd reads it on Windows (spec/tools.md WP-13.4, "Recursive components
    and Pi-scope constructs"): no backslash escape; a class is "[", an optional "!" or "^", a
    leading "]" as a member, then up to the next "]" (an unclosed "[" is an ordinary character);
    "{" and "*" are syntax; "," and "}" are syntax only inside an open brace group."""
    tokens: list[_Token] = []
    index, depth = 0, 0
    while index < len(text):
        char = text[index]
        if char == "[":
            j = index + 1
            if j < len(text) and text[j] in "!^":
                j += 1
            if j < len(text) and text[j] == "]":
                j += 1
            end = text.find("]", j)
            if end != -1:
                tokens.append(("class", text[index : end + 1]))
                index = end + 1
                continue
        if char == "{":
            kind, depth = "open", depth + 1
        elif char == "}" and depth:
            kind, depth = "close", depth - 1
        elif char == "," and depth:
            kind = "comma"
        else:
            kind = "star" if char == "*" else "lit"
        tokens.append((kind, char))
        index += 1
    return tokens


def _windows_full_path(pattern: str) -> str:
    """The effective Windows full-path pattern (`DIV-002`, CE-L13-WP134-01). It starts from Pi's
    rewrite (every `/` to `[/\\]`) read as the pinned fd reads it, and changes only the recursive
    components: a `**` (exactly two stars) followed by a separator and preceded by a separator or
    beginning a brace alternative; the pattern-initial one is left alone. `SEP ** SEP` becomes
    `{SEP,SEP**SEP}` (adjacent components collapse into one); an alternative-start `** SEP rest`
    becomes `** SEP rest,rest` -- the alternative duplicated without it, since the pinned fd never
    lets an empty alternative match. Every other character is Pi's, so every other construct keeps
    Pi's Windows meaning and the result is the component-local union the contract specifies."""
    tokens = _lex(_pi_windows_rewrite(pattern))
    count = len(tokens)

    def is_sep(k: int) -> bool:
        return 0 <= k < count and tokens[k] == ("class", _SEPARATOR_CLASS)

    def is_double_star(k: int) -> bool:
        return (
            k + 1 < count
            and tokens[k][0] == "star"
            and tokens[k + 1][0] == "star"
            and (k + 2 >= count or tokens[k + 2][0] != "star")
            and (k == 0 or tokens[k - 1][0] != "star")
        )

    def alternative_end(k: int) -> int | None:
        depth = 0
        for j in range(k, count):
            kind = tokens[j][0]
            if kind == "open":
                depth += 1
            elif kind == "close":
                if depth == 0:
                    return j
                depth -= 1
            elif kind == "comma" and depth == 0:
                return j
        return None

    def render(lo: int, hi: int) -> str:
        out: list[str] = []
        i = lo
        while i < hi:
            if (
                i > 0
                and is_double_star(i)
                and is_sep(i + 2)
                and (is_sep(i - 1) or tokens[i - 1][0] in ("open", "comma"))
            ):
                j = i + 3
                while j + 2 <= hi and is_double_star(j) and is_sep(j + 2):
                    j += 3
                if is_sep(i - 1):
                    sep = _SEPARATOR_CLASS
                    out[-1:] = [f"{{{sep},{sep}**{sep}}}"]
                    i = j
                    continue
                end = alternative_end(i)
                if end is not None and end <= hi:
                    rest = render(j, end)
                    out.append(f"**{_SEPARATOR_CLASS}{rest},{rest}")
                    i = end
                    continue
            out.append(tokens[i][1])
            i += 1
        return "".join(out)

    return render(0, count)


async def _exists(fs: FileSystem, platform: Platform, path: str) -> bool:
    """Pi's `pathExists` (`access(F_OK)`), mapped as WP-13.3 maps it (`CE-WP133-01`,
    `WP134-CON-R001`): non-following on Windows, following on POSIX; any error is false."""
    if platform is Platform.WINDOWS:
        return not isinstance(await fs.file_info(path), Err)
    return not isinstance(await fs.probe_dir_entry(path), Err)


async def _race_window[T](work: Coroutine[Any, Any, T], window: AbortWindow) -> T:
    """Pi's find settles "Operation aborted" the moment its listener fires (onAbort -> settle), and
    the listener lives from the call's start until engine completion (CE-L13-WP134-01). An abort
    inside the window answers at once and leaves the work running (it stops the engine); once the
    window has closed the work's own outcome stands."""
    task = asyncio.ensure_future(work)

    async def watch() -> None:
        while window.open and not window.fired():
            await asyncio.sleep(_POLL_INTERVAL_S)

    watcher = asyncio.ensure_future(watch())
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watcher.cancel()
    if task.done() or not window.fired():
        return await task
    _ABANDONED.add(task)
    task.add_done_callback(_discard)
    raise aborted()


def relativize(line: str, search_path: str, node: NodePath) -> str:
    """Pi's `relativizeFindResultPath`."""
    had_trailing = line.endswith(node.sep) or (node.windows and line.endswith("/"))
    relative = node.relative(search_path, line) if node.is_absolute(line) else line
    posix = "/".join(relative.split(node.sep))
    return posix + "/" if had_trailing and not posix.endswith("/") else posix


def create_find_tool(fs: FileSystem, subprocess: Subprocess, engines: Engines) -> ToolDefinition:
    """The `find` tool over one execution world and an engine source: the certified
    `EngineStore`, or an explicit (uncertified) `EngineOverride`."""
    world = validate([("fs", fs.execution_world), ("subprocess", subprocess.execution_world)])
    if isinstance(world, Err):
        raise _IncompatibleWorld(world.error)
    node = NodePath(subprocess.platform)

    async def work(
        pattern: str, path: str | None, limit: Any, window: AbortWindow
    ) -> tuple[str, dict[str, Any]]:
        working = preprocess_path(path or ".")
        resolved = await fs.absolute_path(working)
        search_path = working if isinstance(resolved, Err) else resolved.value
        effective_limit = DEFAULT_LIMIT if limit is None else limit
        fd = await resolve_engine(engines, subprocess, "fd")
        if window.fired():
            raise aborted()
        args = ["--glob", "--color=never", "--hidden"]
        inside_repo = False
        current = search_path
        while True:
            if await _exists(fs, subprocess.platform, node.join(current, ".git")):
                inside_repo = True
                break
            parent = node.dirname(current)
            if parent == current:
                break
            current = parent
        if not inside_repo:
            args.append("--no-require-git")
        args += ["--max-results", number_to_string(effective_limit)]
        pi_pattern = effective = pattern
        if "/" in pattern:
            args.append("--full-path")
            prefixed = pattern
            if not pattern.startswith("/") and not pattern.startswith("**/") and pattern != "**":
                prefixed = f"**/{pattern}"
            pi_pattern = effective = prefixed
            if node.windows:
                pi_pattern = _pi_windows_rewrite(prefixed)
                effective = _windows_full_path(prefixed)

        async def run_fd(glob: str, may_retry: bool) -> tuple[EngineRun, list[str], bool]:
            run = await spawn_engine(
                subprocess, [*fd, *args, "--", glob, search_path], "Failed to run fd"
            )
            lines: list[str] = []
            rejected: list[bool] = []

            def on_line(line: str) -> bool:
                lines.append(line)
                return False

            def complete() -> None:  # engine completion: the outcome is decided here
                rejected.append(may_retry and run.exit_code != 0 and not "\n".join(lines))
                if not rejected[0]:
                    window.close()

            async def watch_abort() -> None:  # Pi's onAbort: stopChild(), only while listening
                while window.open and not window.fired():
                    await asyncio.sleep(_POLL_INTERVAL_S)
                if window.fired():
                    await terminate_quietly(run.process)

            watcher = asyncio.ensure_future(watch_abort())
            try:
                await run.run(on_line, complete)
            finally:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            return run, lines, bool(rejected) and rejected[0]

        run, lines, rejected = await run_fd(effective, effective != pi_pattern)
        if rejected:
            # The corrected pattern was rejected (non-zero exit, no output): the result is Pi's own
            # -- fd's outcome for Pi's rewritten text (CE-L13-WP134-01, rule 5). The abort window
            # stays open across both runs.
            if window.fired():
                raise aborted()
            # TOOL-038 "Verification at use time": every spawn is preceded by its own verification
            # (WP134-IMPL-R004); a failed one is the unavailable-engine error, with no spawn.
            fd = await resolve_engine(engines, subprocess, "fd")
            run, lines, _ = await run_fd(pi_pattern, False)
        if window.observed:  # decided at engine completion, not after the stream release
            raise aborted()
        output = "\n".join(lines)
        if run.exit_code != 0 and not output:
            code = "null" if run.exit_code is None else str(run.exit_code)
            raise BuiltinToolError(js_trim(run.stderr) or f"fd exited with code {code}")
        if not output:
            return "No files found matching pattern", {}
        relativized: list[str] = []
        for raw in lines:
            line = js_trim(raw[:-1] if raw.endswith("\r") else raw)
            if line:
                relativized.append(relativize(line, search_path, node))
        limit_reached = len(relativized) >= effective_limit
        truncation = truncate_head("\n".join(relativized), max_lines=MAX_SAFE_INTEGER)
        text = truncation.content
        details: dict[str, Any] = {}
        notices: list[str] = []
        if limit_reached:
            notices.append(
                f"{number_to_string(effective_limit)} results limit reached. "
                f"Use limit={number_to_string(effective_limit * 2)} for more, or refine pattern"
            )
            details["resultLimitReached"] = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation_result(truncation)
        if notices:
            text += "\n\n[" + ". ".join(notices) + "]"
        return text, details

    async def execute(
        tool_call_id: str, arguments: dict[str, Any], signal: RunSignal | None
    ) -> ToolResult:
        if signal is not None and signal.aborted:
            raise aborted()
        window = AbortWindow(signal)
        text, details = await _race_window(
            work(arguments["pattern"], arguments.get("path"), arguments.get("limit"), window),
            window,
        )
        return ToolResult(
            tool_call_id=tool_call_id,
            content=(TextBlock(text=text),),
            tool_name="find",
            details=details,
        )

    return ToolDefinition(
        name="find",
        label="find",
        description=FIND_DESCRIPTION,
        parameters=FIND_PARAMETERS,
        execute=execute,
        wants_signal=True,
    )


class _IncompatibleWorld(Exception):
    def __init__(self, error: ExecutionWorldError) -> None:
        super().__init__("find requires ctx.fs and ctx.subprocess in the same execution world")
        self.error = error
