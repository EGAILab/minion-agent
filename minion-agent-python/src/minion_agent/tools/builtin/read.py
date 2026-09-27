"""The `read` built-in tool (`TOOL-025`; pinned Pi `core/tools/read.ts`), over `ctx.fs`.

Flow (spec/tools.md `TOOL-025`): path pipeline (`TOOL-026`) -> `check_readable` (`EXEC-008`, Pi's
`access`) -> `read_binary_file` -> image sniff -> image processing, or UTF-8 text with offset/limit,
then head truncation. Every filesystem access is a read-only `ctx.fs` operation.
"""

from __future__ import annotations

import asyncio
import base64
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...execution import Err, FileSystem, FsError, FsErrorCode
from ...llm import ImageBlock, TextBlock
from ...runtime.signal import RunSignal
from ..definition import ToolDefinition
from ..result import ToolResult
from ._js import js_slice, math_max, math_min, number_to_string, to_number
from ._signal import race_abort
from .image import process_image
from .mime import detect_supported_image_mime_type
from .paths import BuiltinToolError, aborted, cause, preprocess_path
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head, utf8_len

NON_VISION_IMAGE_NOTE = (
    "[Current model does not support images. The image will be omitted from this request.]"
)

READ_DESCRIPTION = (
    "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
    "Images are sent as attachments. For text files, output is truncated to "
    f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use "
    "offset/limit for large files. When you need the full file, continue with offset until "
    "complete."
)

READ_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read (relative or absolute)",
        },
        "offset": {
            "type": "number",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}

# Node's own TypeError when read.ts:299 measures `allLines[startLine]` at a fractional index
# (spec/tools.md TOOL-025, IMPL-C010).
_UNDEFINED_LINE_ERROR = (
    'The "string" argument must be of type string or an instance of Buffer or ArrayBuffer. '
    "Received undefined"
)

type _Output = tuple[tuple[TextBlock | ImageBlock, ...], dict[str, Any]]

ModelSupportsImages = Callable[[], bool | None]
"""Whether the model the current request goes to accepts images; `None` when unknown. Pi reads
`ctx.model` at execution time and adds its non-vision note only for a known model without image
input (spec/tools.md TOOL-025, IMPL-C001)."""


@dataclass(frozen=True, slots=True)
class ReadToolOptions:
    auto_resize_images: bool = True
    """Pi's `autoResizeImages` (default `true`)."""
    model_supports_images: ModelSupportsImages | None = None


_ACCESS_SITE = "Cannot access"
_READ_SITE = "Cannot read"


def _fallback_read_site(code: FsErrorCode) -> str:
    """The site of a failed content read on a provider WITHOUT `EXEC-008` (`L13-WP131-C012`, R-G3;
    spec/execution.md §12.5 FALLBACK): the provider-capability fallback, an intentional
    approximation and never Pi-equivalent. Without Pi's access stage, a read failing for a reason
    `access(R_OK)` would have caught is reported at the access site; `is_directory` and
    `not_supported` are genuine read failures."""
    if code in (FsErrorCode.IS_DIRECTORY, FsErrorCode.NOT_SUPPORTED):
        return _READ_SITE
    return _ACCESS_SITE


def _fs_failure(site: str, absolute: str, error: FsError) -> BuiltinToolError:
    if error.code == FsErrorCode.ABORTED:
        return aborted()
    return BuiltinToolError(f"{site} {absolute}: {cause(error.code)}")


class _Read:
    def __init__(self, fs: FileSystem, options: ReadToolOptions) -> None:
        self._fs = fs
        self._options = options

    async def _absolute(self, working: str) -> str:
        resolved = await self._fs.absolute_path(working)
        return working if isinstance(resolved, Err) else resolved.value

    async def run(
        self, path: str, offset: float | None, limit: float | None, signal: RunSignal | None
    ) -> _Output:
        """Pi's abort checkpoints (`read.ts:246`, `:249`) stop the work with `"Operation aborted"`;
        the caller may already have been answered by the race by then. No `ctx.fs` call is
        given the signal: Pi passes none to `access`/`readFile` (`L13-WP131-I001`).

        The operation that failed owns the error site, whatever its code (`L13-WP131-C012`, G1):
        `check_readable` is Pi's `access` stage (`read.ts:248`), so its failure is `Cannot
        access`; the one content read is Pi's sniff + `readFile`, so its failure is `Cannot read`.
        A provider without `EXEC-008` answers `not_supported` and is read in the disclosed
        FALLBACK mode (spec/execution.md §12.5)."""
        working = preprocess_path(path)
        if signal is not None and signal.aborted:
            raise aborted()
        access = await self._fs.check_readable(working)
        fallback = False
        if isinstance(access, Err):
            if access.error.code != FsErrorCode.NOT_SUPPORTED:
                raise _fs_failure(_ACCESS_SITE, await self._absolute(working), access.error)
            fallback = True
        # read.ts:249 -- one checkpoint once the access stage has completed, in both modes.
        if signal is not None and signal.aborted:
            raise aborted()
        read = await self._fs.read_binary_file(working)
        if isinstance(read, Err):
            site = _fallback_read_site(read.error.code) if fallback else _READ_SITE
            raise _fs_failure(site, await self._absolute(working), read.error)
        data = read.value
        mime_type = detect_supported_image_mime_type(data)
        if mime_type is not None:
            return await self._image(data, mime_type)
        return _text(data, path, offset, limit)

    async def _image(self, data: bytes, mime_type: str) -> _Output:
        processed = await asyncio.to_thread(
            process_image, data, mime_type, auto_resize=self._options.auto_resize_images
        )
        supports = self._options.model_supports_images
        note = NON_VISION_IMAGE_NOTE if supports is not None and supports() is False else None
        if not processed.ok:
            text = f"Read image file [{mime_type}]\n{processed.message}"
            if note:
                text += f"\n{note}"
            return (TextBlock(text=text),), {}
        text = f"Read image file [{processed.mime_type}]"
        if processed.hints:
            text += "\n" + "\n".join(processed.hints)
        if note:
            text += f"\n{note}"
        image = ImageBlock(mime_type=processed.mime_type, data=base64.b64decode(processed.data))
        return (TextBlock(text=text), image), {}


def _text(data: bytes, path: str, offset: float | None, limit: float | None) -> _Output:
    # Node's Buffer.toString("utf-8"): U+FFFD per maximal invalid subpart, BOM kept, no newline
    # translation (IMPL-C005).
    all_lines = data.decode("utf-8", "replace").split("\n")
    total_file_lines = len(all_lines)
    # `offset ? Math.max(0, offset - 1) : 0` -- a JS-falsy offset (absent, 0, NaN) starts at line 1.
    start_line = 0.0
    if offset is not None and offset != 0 and not math.isnan(offset):
        start_line = math_max(0.0, offset - 1)
    start_display = start_line + 1
    if start_line >= total_file_lines:
        raise BuiltinToolError(
            f"Offset {number_to_string(offset if offset is not None else 0)} is beyond end of file "
            f"({total_file_lines} lines total)"
        )
    user_limited: float | None = None
    if limit is not None:
        end_line = math_min(start_line + limit, float(total_file_lines))
        selected = "\n".join(js_slice(all_lines, start_line, end_line))
        user_limited = end_line - start_line
    else:
        selected = "\n".join(js_slice(all_lines, start_line))
    truncation = truncate_head(selected)
    details: dict[str, Any] = {}
    if truncation.first_line_exceeds_limit:
        first_line = all_lines[int(start_line)] if start_line.is_integer() else None
        if first_line is None:
            raise BuiltinToolError(_UNDEFINED_LINE_ERROR)
        size = format_size(utf8_len(first_line))
        display = number_to_string(start_display)
        text = (
            f"[Line {display} is {size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. Use bash: "
            f"sed -n '{display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
        )
        details = {"truncation": truncation.details()}
    elif truncation.truncated:
        end_display = start_display + truncation.output_lines - 1
        next_offset = end_display + 1
        shown = (
            f"lines {number_to_string(start_display)}-{number_to_string(end_display)} of "
            f"{total_file_lines}"
        )
        if truncation.truncated_by == "lines":
            notice = f"[Showing {shown}. Use offset={number_to_string(next_offset)} to continue.]"
        else:
            notice = (
                f"[Showing {shown} ({format_size(DEFAULT_MAX_BYTES)} limit). Use "
                f"offset={number_to_string(next_offset)} to continue.]"
            )
        text = f"{truncation.content}\n\n{notice}"
        details = {"truncation": truncation.details()}
    elif user_limited is not None and start_line + user_limited < total_file_lines:
        remaining = total_file_lines - (start_line + user_limited)
        next_offset = start_line + user_limited + 1
        text = (
            f"{truncation.content}\n\n[{number_to_string(remaining)} more lines in file. Use "
            f"offset={number_to_string(next_offset)} to continue.]"
        )
    else:
        text = truncation.content
    return (TextBlock(text=text),), details


def create_read_tool(fs: FileSystem, options: ReadToolOptions | None = None) -> ToolDefinition:
    """The `read` tool bound to one `ctx.fs` provider."""
    reader = _Read(fs, options or ReadToolOptions())

    async def execute(
        tool_call_id: str, arguments: dict[str, Any], signal: RunSignal | None
    ) -> ToolResult:
        if signal is not None and signal.aborted:
            raise aborted()
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        output = await race_abort(
            reader.run(
                arguments["path"],
                None if offset is None else to_number(offset),
                None if limit is None else to_number(limit),
                signal,
            ),
            signal,
        )
        content, details = output
        return ToolResult(
            tool_call_id=tool_call_id, content=content, tool_name="read", details=details
        )

    return ToolDefinition(
        name="read",
        label="read",
        description=READ_DESCRIPTION,
        parameters=READ_PARAMETERS,
        execute=execute,
        wants_signal=True,
    )
