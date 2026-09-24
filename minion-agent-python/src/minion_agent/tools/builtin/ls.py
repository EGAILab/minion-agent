"""The `ls` built-in tool (`TOOL-028`; pinned Pi `core/tools/ls.ts`), over `ctx.fs`.

spec/tools.md `TOOL-028` "Algorithm": one `probe_dir_entry` for the directory check, raw
`list_dir_raw` enumeration, a stable sort with the pinned ICU collation (`R006-C`), then a lazy
per-entry loop that checks the cap BEFORE each probe and silently skips entries whose probe fails.
"""

from __future__ import annotations

from typing import Any

from ...execution import Err, FileSystem, FsErrorCode
from ...execution.filesystem import DirEntryProbeKind
from ...llm import TextBlock
from ...runtime.signal import RunSignal
from ..definition import ToolDefinition
from ..result import ToolResult
from ._js import number_to_string, to_number
from ._signal import race_abort
from .collation import pinned_collation
from .paths import BuiltinToolError, aborted, cause, preprocess_path
from .truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 500
_MAX_SAFE_INTEGER = 9007199254740991
_DIRECTORY_KINDS = (DirEntryProbeKind.DIRECTORY, DirEntryProbeKind.SYMLINK_TO_DIRECTORY)

LS_DESCRIPTION = (
    "List directory contents. Returns entries sorted alphabetically, with '/' suffix for "
    f"directories. Includes dotfiles. Output is truncated to {DEFAULT_LIMIT} entries or "
    f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
)

LS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list (default: current directory)"},
        "limit": {
            "type": "number",
            "description": "Maximum number of entries to return (default: 500)",
        },
    },
}

type _Output = tuple[str, dict[str, Any]]


async def _list(fs: FileSystem, path: str, limit: float | int, signal: RunSignal | None) -> _Output:
    working = preprocess_path(path)
    resolved = await fs.absolute_path(working)
    directory = working if isinstance(resolved, Err) else resolved.value
    probe = await fs.probe_dir_entry(working)
    if isinstance(probe, Err):
        if probe.error.code == FsErrorCode.NOT_SUPPORTED:
            raise BuiltinToolError(f"Cannot access {directory}: {cause(FsErrorCode.NOT_SUPPORTED)}")
        raise BuiltinToolError(f"Path not found: {directory}")
    if probe.value.kind not in _DIRECTORY_KINDS:
        raise BuiltinToolError(f"Not a directory: {directory}")
    listing = await fs.list_dir_raw(working, signal)
    if isinstance(listing, Err):
        if listing.error.code == FsErrorCode.ABORTED:
            raise aborted()
        raise BuiltinToolError(f"Cannot read directory: {cause(listing.error.code)}")
    names = pinned_collation().sort(listing.value)
    effective_limit = to_number(limit)
    results: list[str] = []
    entry_limit_reached = False
    for name in names:
        if len(results) >= effective_limit:
            entry_limit_reached = True
            break
        joined = await fs.join_path([directory, name])
        if isinstance(joined, Err):
            continue
        entry = await fs.probe_dir_entry(joined.value)
        if isinstance(entry, Err):
            continue
        results.append(name + "/" if entry.value.kind in _DIRECTORY_KINDS else name)
    if not results:
        return "(empty directory)", {}
    truncation = truncate_head("\n".join(results), max_lines=_MAX_SAFE_INTEGER)
    output = truncation.content
    details: dict[str, Any] = {}
    notices: list[str] = []
    if entry_limit_reached:
        notices.append(
            f"{number_to_string(effective_limit)} entries limit reached. "
            f"Use limit={number_to_string(effective_limit * 2)} for more"
        )
        details["entry_limit_reached"] = limit
    if truncation.truncated:
        notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        details["truncation"] = truncation.details()
    if notices:
        output += "\n\n[" + ". ".join(notices) + "]"
    return output, details


def create_ls_tool(fs: FileSystem) -> ToolDefinition:
    """The `ls` tool bound to one `ctx.fs` provider."""

    async def execute(
        tool_call_id: str, arguments: dict[str, Any], signal: RunSignal | None
    ) -> ToolResult:
        if signal is not None and signal.aborted:
            raise aborted()
        path = arguments.get("path") or "."
        limit = arguments.get("limit")
        text, details = await race_abort(
            _list(fs, path, DEFAULT_LIMIT if limit is None else limit, signal), signal
        )
        return ToolResult(
            tool_call_id=tool_call_id,
            content=(TextBlock(text=text),),
            tool_name="ls",
            details=details,
        )

    return ToolDefinition(
        name="ls",
        label="ls",
        description=LS_DESCRIPTION,
        parameters=LS_PARAMETERS,
        execute=execute,
        wants_signal=True,
    )
