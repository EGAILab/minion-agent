"""The `grep` built-in tool (`TOOL-037`; pinned Pi `core/tools/grep.ts`, default implementation),
spec/tools.md WP-13.4 "`grep`": Pi's wrapper over the pinned `ripgrep 15.2.0` (`TOOL-038`) -- the
JSON match stream, Pi's own context reconstruction from a re-read of each file, line and output
truncation. Matching itself is the engine's.

Pi registers its abort listener only after the engine is spawned, so an abort before that point is
never observed (an already-aborted signal does not fire again); an abort after it kills the engine
and the call settles `"Operation aborted"` once the engine has exited.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from typing import Any

from ...execution import Err, FileSystem, Subprocess
from ...execution.filesystem import DirEntryProbeKind
from ...execution.world import ExecutionWorldError, validate
from ...llm import TextBlock
from ...runtime.signal import RunSignal
from ..definition import ToolDefinition
from ..result import ToolResult
from ._js import math_max, math_min, number_to_string
from ._node_path import NodePath
from ._search import (
    MAX_SAFE_INTEGER,
    resolve_engine,
    spawn_engine,
    terminate_quietly,
    truncation_result,
)
from ._utf16 import decode_utf8, from_units, to_units
from .bash_shell import js_trim
from .paths import BuiltinToolError, aborted, preprocess_path
from .search_engines import Engines
from .truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 100
GREP_MAX_LINE_LENGTH = 500
_DIRECTORY_KINDS = (DirEntryProbeKind.DIRECTORY, DirEntryProbeKind.SYMLINK_TO_DIRECTORY)

GREP_DESCRIPTION = (
    "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
    f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} matches or "
    f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Long lines are truncated to "
    f"{GREP_MAX_LINE_LENGTH} chars."
)

GREP_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
        "path": {
            "type": "string",
            "description": "Directory or file to search (default: current directory)",
        },
        "glob": {
            "type": "string",
            "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
        },
        "ignoreCase": {
            "type": "boolean",
            "description": "Case-insensitive search (default: false)",
        },
        "literal": {
            "type": "boolean",
            "description": "Treat pattern as literal string instead of regex (default: false)",
        },
        "context": {
            "type": "number",
            "description": "Number of lines to show before and after each match (default: 0)",
        },
        "limit": {
            "type": "number",
            "description": "Maximum number of matches to return (default: 100)",
        },
    },
    "required": ["pattern"],
}


def truncate_line(line: str) -> tuple[str, bool]:
    """Pi's `truncateLine`: measured in UTF-16 code units; a cut may split a surrogate pair,
    leaving a lone high surrogate (carried by the tool-result domain, `L0506-D003`)."""
    units = to_units(line)
    if len(units) <= GREP_MAX_LINE_LENGTH:
        return line, False
    return from_units(units[:GREP_MAX_LINE_LENGTH]) + "... [truncated]", True


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _field(value: Any, key: str) -> Any:
    """`value?.[key]` over parsed JSON: absent unless `value` is an object."""
    return value.get(key) if isinstance(value, dict) else None


def _js_index(lines: list[str], position: float) -> str:
    """`lines[position] ?? ""` with a JS number index: a fractional or out-of-range index is
    absent."""
    if isinstance(position, float) and not position.is_integer():
        return ""
    index = int(position)
    return lines[index] if 0 <= index < len(lines) else ""


def create_grep_tool(fs: FileSystem, subprocess: Subprocess, engines: Engines) -> ToolDefinition:
    """The `grep` tool over one execution world and an engine source (the certified
    `EngineStore`, or an explicit uncertified `EngineOverride`)."""
    world = validate([("fs", fs.execution_world), ("subprocess", subprocess.execution_world)])
    if isinstance(world, Err):
        raise _IncompatibleWorld(world.error)
    node = NodePath(subprocess.platform)

    async def execute(
        tool_call_id: str, arguments: dict[str, Any], signal: RunSignal | None
    ) -> ToolResult:
        if signal is not None and signal.aborted:
            raise aborted()
        rg = await resolve_engine(engines, subprocess, "rg")
        working = preprocess_path(arguments.get("path") or ".")
        resolved = await fs.absolute_path(working)
        search_path = working if isinstance(resolved, Err) else resolved.value
        probe = await fs.probe_dir_entry(search_path)
        if isinstance(probe, Err):
            raise BuiltinToolError(f"Path not found: {search_path}")
        is_directory = probe.value.kind in _DIRECTORY_KINDS
        context = arguments.get("context")
        context_value = context if (context and context > 0) else 0
        limit = arguments.get("limit")
        effective_limit = math_max(1, DEFAULT_LIMIT if limit is None else limit)

        def format_path(file_path: str) -> str:
            if is_directory:
                relative = node.relative(search_path, file_path)
                if relative and not relative.startswith(".."):
                    return relative.replace("\\", "/")
            return node.basename(file_path)

        args = ["--json", "--line-number", "--color=never", "--hidden"]
        if arguments.get("ignoreCase"):
            args.append("--ignore-case")
        if arguments.get("literal"):
            args.append("--fixed-strings")
        if arguments.get("glob"):
            args += ["--glob", arguments["glob"]]
        args += ["--", arguments["pattern"], search_path]

        run = await spawn_engine(subprocess, [*rg, *args], "Failed to run ripgrep")
        state = {"count": 0, "limit_reached": False, "killed_for_limit": False, "aborted": False}
        matches: list[tuple[str, Any, str | None]] = []

        def on_line(line: str) -> bool:
            if not js_trim(line) or state["count"] >= effective_limit:
                return False
            try:
                event = json.loads(line)
            except ValueError:
                return False
            if not isinstance(event, dict) or event.get("type") != "match":
                return False
            state["count"] += 1
            data = _field(event, "data")
            file_path = _field(_field(data, "path"), "text")
            line_number = _field(data, "line_number")
            line_text = _field(_field(data, "lines"), "text")
            if file_path and _is_number(line_number):
                matches.append((file_path, line_number, line_text))
            if state["count"] >= effective_limit:
                state["limit_reached"] = True
                if not run.stopped:
                    state["killed_for_limit"] = True
                return True
            return False

        registered_aborted = signal is not None and signal.aborted

        async def watch_abort() -> None:  # Pi's onAbort, registered after spawn
            if signal is None or registered_aborted:
                return
            while not signal.aborted:
                await asyncio.sleep(0.01)
            state["aborted"] = True
            await terminate_quietly(run.process)

        watcher = asyncio.ensure_future(watch_abort())
        try:
            await run.run(on_line)
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        if state["aborted"]:
            raise aborted()
        if not state["killed_for_limit"] and run.exit_code not in (0, 1):
            code = "null" if run.exit_code is None else str(run.exit_code)
            raise BuiltinToolError(js_trim(run.stderr) or f"ripgrep exited with code {code}")
        if state["count"] == 0:
            return ToolResult(
                tool_call_id=tool_call_id,
                content=(TextBlock(text="No matches found"),),
                tool_name="grep",
                details={},
            )

        cache: dict[str, list[str]] = {}
        lines_truncated = False
        output: list[str] = []

        async def file_lines(file_path: str) -> list[str]:
            if file_path not in cache:
                read = await fs.read_binary_file(file_path)
                if isinstance(read, Err):
                    cache[file_path] = []
                else:
                    text = decode_utf8(read.value).replace("\r\n", "\n").replace("\r", "\n")
                    cache[file_path] = text.split("\n")
            return cache[file_path]

        for file_path, line_number, line_text in matches:
            relative = format_path(file_path)
            if context_value == 0 and line_text is not None:
                sanitized = line_text.replace("\r\n", "\n").replace("\r", "")
                if sanitized.endswith("\n"):
                    sanitized = sanitized[:-1]
                shown, cut = truncate_line(sanitized)
                lines_truncated = lines_truncated or cut
                output.append(f"{relative}:{number_to_string(line_number)}: {shown}")
                continue
            lines = await file_lines(file_path)
            if not lines:
                output.append(f"{relative}:{number_to_string(line_number)}: (unable to read file)")
                continue
            start = math_max(1, line_number - context_value) if context_value > 0 else line_number
            end = (
                math_min(len(lines), line_number + context_value)
                if context_value > 0
                else line_number
            )
            current = start
            while current <= end:
                text = _js_index(lines, current - 1).replace("\r", "")
                shown, cut = truncate_line(text)
                lines_truncated = lines_truncated or cut
                label = number_to_string(current)
                output.append(
                    f"{relative}:{label}: {shown}"
                    if current == line_number
                    else f"{relative}-{label}- {shown}"
                )
                current += 1
                if math.isinf(current):  # pragma: no cover - an infinite window is clamped by end
                    break

        truncation = truncate_head("\n".join(output), max_lines=MAX_SAFE_INTEGER)
        text_out = truncation.content
        details: dict[str, Any] = {}
        notices: list[str] = []
        if state["limit_reached"]:
            notices.append(
                f"{number_to_string(effective_limit)} matches limit reached. "
                f"Use limit={number_to_string(effective_limit * 2)} for more, or refine pattern"
            )
            details["matchLimitReached"] = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation_result(truncation)
        if lines_truncated:
            notices.append(
                f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. "
                "Use read tool to see full lines"
            )
            details["linesTruncated"] = True
        if notices:
            text_out += "\n\n[" + ". ".join(notices) + "]"
        return ToolResult(
            tool_call_id=tool_call_id,
            content=(TextBlock(text=text_out),),
            tool_name="grep",
            details=details,
        )

    return ToolDefinition(
        name="grep",
        label="grep",
        description=GREP_DESCRIPTION,
        parameters=GREP_PARAMETERS,
        execute=execute,
        wants_signal=True,
    )


class _IncompatibleWorld(Exception):
    def __init__(self, error: ExecutionWorldError) -> None:
        super().__init__("grep requires ctx.fs and ctx.subprocess in the same execution world")
        self.error = error
