"""The `write` built-in tool (`TOOL-029`; pinned Pi `core/tools/write.ts`), over `ctx.fs`.

spec/tools.md WP-13.2 "`write`": `TOOL-026` steps 1-4, then inside the mutation queue: abort check,
`absolute_path` (in the lock, `L13-WP132-R001`), `create_dir(lexical parent, recursive)`, abort
check, `write_file`, abort check. The success count is `content`'s UTF-16 length -- Pi's
`content.length`, reproduced verbatim although the message says "bytes".
"""

from __future__ import annotations

import os
from typing import Any

from ...execution import Err, FileSystem
from ...llm import TextBlock
from ...runtime.signal import RunSignal
from ..definition import ToolDefinition
from ..result import ToolResult
from ._utf16 import encode_utf8, utf16_length
from .mutation_queue import with_mutation_queue
from .paths import BuiltinToolError, aborted, cause, preprocess_path

WRITE_DESCRIPTION = (
    "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
    "Automatically creates parent directories."
)

WRITE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to write (relative or absolute)",
        },
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


def check_abort(signal: RunSignal | None) -> None:
    """Pi's `throwIfAborted` checkpoint. There is no abort listener: an abort is observed only
    here, after the in-flight operation settled, so the queue lock is never released early
    (`TOOL-033`)."""
    if signal is not None and signal.aborted:
        raise aborted()


def create_write_tool(fs: FileSystem) -> ToolDefinition:
    """The `write` tool bound to one `ctx.fs` provider."""

    async def execute(
        tool_call_id: str, arguments: dict[str, Any], signal: RunSignal | None
    ) -> ToolResult:
        path: str = arguments["path"]
        content: str = arguments["content"]
        p = preprocess_path(path)

        async def work() -> str:
            check_abort(signal)
            absolute = await fs.absolute_path(p)
            if isinstance(absolute, Err):
                raise BuiltinToolError(f"Cannot resolve {path}: {cause(absolute.error.code)}")
            created = await fs.create_dir(os.path.dirname(absolute.value), recursive=True)
            if isinstance(created, Err):
                raise BuiltinToolError(
                    f"Cannot create parent directory of {path}: {cause(created.error.code)}"
                )
            check_abort(signal)
            written = await fs.write_file(p, encode_utf8(content))
            if isinstance(written, Err):
                raise BuiltinToolError(f"Cannot write {path}: {cause(written.error.code)}")
            check_abort(signal)
            return f"Successfully wrote {utf16_length(content)} bytes to {path}"

        text = await with_mutation_queue(fs, p, path, work)
        return ToolResult(
            tool_call_id=tool_call_id,
            content=(TextBlock(text=text),),
            tool_name="write",
            details={},
        )

    return ToolDefinition(
        name="write",
        label="write",
        description=WRITE_DESCRIPTION,
        parameters=WRITE_PARAMETERS,
        execute=execute,
        wants_signal=True,
    )
