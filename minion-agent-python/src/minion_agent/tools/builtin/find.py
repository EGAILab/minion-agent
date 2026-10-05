"""The `find` built-in tool (`TOOL-036`; pinned Pi `core/tools/find.ts`, default implementation),
spec/tools.md WP-13.4 "`find`": Pi's wrapper over the pinned `fd 10.4.2` (`TOOL-038`), reached
through the execution world -- `ctx.fs` for the repository walk, `ctx.subprocess` for the engine.

Cross-entry order is the engine's (`ENGINE_DEFINED_UNSPECIFIED`, Owner Q2): nothing here sorts.
`DIV-002` keeps a Windows full-path `**/` component's zero-directory meaning.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    resolve_engine,
    spawn_engine,
    terminate_quietly,
    truncation_result,
)
from ._signal import race_abort
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


def _windows_full_path(pattern: str) -> str:
    """Pi's Windows rewrite of every `/` to `[/\\]`, except (`DIV-002`) a `/**/` outside any brace
    group becomes an alternation that also matches zero directories, so `**/` keeps its ordinary
    meaning. Inside braces (fd's glob has no nested alternation) Pi's rewrite is kept as is.
    An escaped character and a character class are never part of a `/**/` component, but their
    `/` still takes Pi's rewrite (Pi's `replaceAll` spares neither), so they keep Pi's meaning."""
    out: list[str] = []
    depth = 0
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            escaped = pattern[index + 1]
            out.append(char + (_SEPARATOR_CLASS if escaped == "/" else escaped))
            index += 2
            continue
        if char == "[":
            end = pattern.find("]", index + 2)
            if end != -1:
                out.append(pattern[index : end + 1].replace("/", _SEPARATOR_CLASS))
                index = end + 1
                continue
        if char == "{":
            depth += 1
        elif char == "}" and depth:
            depth -= 1
        if depth == 0 and pattern.startswith("/**/", index):
            out.append(f"{{{_SEPARATOR_CLASS},{_SEPARATOR_CLASS}**{_SEPARATOR_CLASS}}}")
            index += 4
            continue
        out.append(_SEPARATOR_CLASS if char == "/" else char)
        index += 1
    return "".join(out)


async def _exists(fs: FileSystem, platform: Platform, path: str) -> bool:
    """Pi's `pathExists` (`access(F_OK)`), mapped as WP-13.3 maps it (`CE-WP133-01`,
    `WP134-CON-R001`): non-following on Windows, following on POSIX; any error is false."""
    if platform is Platform.WINDOWS:
        return not isinstance(await fs.file_info(path), Err)
    return not isinstance(await fs.probe_dir_entry(path), Err)


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
        pattern: str, path: str | None, limit: Any, signal: RunSignal | None
    ) -> tuple[str, dict[str, Any]]:
        working = preprocess_path(path or ".")
        resolved = await fs.absolute_path(working)
        search_path = working if isinstance(resolved, Err) else resolved.value
        effective_limit = DEFAULT_LIMIT if limit is None else limit
        fd = await resolve_engine(engines, subprocess, "fd")
        if signal is not None and signal.aborted:
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
        effective = pattern
        if "/" in pattern:
            args.append("--full-path")
            if not pattern.startswith("/") and not pattern.startswith("**/") and pattern != "**":
                effective = f"**/{pattern}"
            if node.windows:
                effective = _windows_full_path(effective)
        args += ["--", effective, search_path]

        run = await spawn_engine(subprocess, [*fd, *args], "Failed to run fd")
        lines: list[str] = []

        def on_line(line: str) -> bool:
            lines.append(line)
            return False

        async def watch_abort() -> None:  # Pi's onAbort: stopChild()
            if signal is None:
                return
            while not signal.aborted:
                await asyncio.sleep(0.01)
            await terminate_quietly(run.process)

        watcher = asyncio.ensure_future(watch_abort())
        try:
            await run.run(on_line)
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        if signal is not None and signal.aborted:
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
        text, details = await race_abort(
            work(arguments["pattern"], arguments.get("path"), arguments.get("limit"), signal),
            signal,
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
