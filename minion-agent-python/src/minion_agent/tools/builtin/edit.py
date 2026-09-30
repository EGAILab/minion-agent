"""The `edit` built-in tool (`TOOL-030`; pinned Pi `core/tools/edit.ts`), over `ctx.fs`.

spec/tools.md WP-13.2 "`edit`": `validateEditInput`, `TOOL-026` steps 1-4, then inside the
mutation queue: abort check; the access stage (`EXEC-009` `check_read_write`, or the disclosed
FALLBACK through `check_readable` on a provider without it); abort check; read; abort check;
match (`edit_diff.apply_edits`); abort check; write; abort check. Result details are
`{diff, patch, firstChangedLine}`.
"""

from __future__ import annotations

import json
from typing import Any

from ...execution import Err, FileSystem, FsErrorCode
from ...llm import TextBlock
from ...runtime.signal import RunSignal
from ..definition import ToolDefinition
from ..result import ToolResult
from . import edit_diff
from ._utf16 import decode_utf8, encode_utf8, from_units, to_units
from .mutation_queue import with_mutation_queue
from .paths import BuiltinToolError, cause, preprocess_path
from .write import check_abort

EDIT_DESCRIPTION = (
    "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
    "non-overlapping region of the original file. If two changes affect the same block or nearby "
    "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
    "unchanged regions just to connect distant changes."
)

EDIT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to edit (relative or absolute)",
        },
        "edits": {
            "type": "array",
            "description": (
                "One or more targeted replacements. Each edit is matched against the original "
                "file, not incrementally. Do not include overlapping or nested edits. If two "
                "changes touch the same block or nearby lines, merge them into one edit instead."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "oldText": {
                        "type": "string",
                        "description": (
                            "Exact text for one targeted replacement. It must be unique in the "
                            "original file and must not overlap with any other edits[].oldText in "
                            "the same call."
                        ),
                    },
                    "newText": {
                        "type": "string",
                        "description": "Replacement text for this targeted edit.",
                    },
                },
                "required": ["oldText", "newText"],
            },
        },
    },
    "required": ["path", "edits"],
}

EMPTY_EDITS = "Edit tool input is invalid. edits must contain at least one replacement."


def _is_single_edit(value: Any) -> bool:
    """`isSingleEditInput`: a non-null, non-array object whose `oldText` and `newText` are
    strings."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("oldText"), str)
        and isinstance(value.get("newText"), str)
    )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"not JSON: {name}")  # JSON.parse has no NaN/Infinity literals


def _json_parse(text: str) -> Any:
    """`JSON.parse`: Python's `json` also accepts the `NaN`/`Infinity`/`-Infinity` literals, which
    JSON does not. (An out-of-range number such as `1e999` parses to infinity in both.)"""
    return json.loads(text, parse_constant=_reject_constant)


def prepare_edit_arguments(arguments: Any) -> Any:
    """`prepareEditArguments` (`edit.ts:116-147`). Layer 06 hands it a copy of the object-valued
    tool-call arguments, so step 1's non-object branch is unreachable through the pipeline
    (`L13-WP132-R005`); called directly, the callback still returns a non-object unchanged."""
    if not isinstance(arguments, dict):
        return arguments
    args = arguments
    edits = args.get("edits")
    if isinstance(edits, str):
        try:
            parsed = _json_parse(edits)
        except ValueError:
            pass
        else:
            if isinstance(parsed, list):
                args["edits"] = parsed
            elif _is_single_edit(parsed):
                args["edits"] = [parsed]
    elif _is_single_edit(edits):
        args["edits"] = [edits]
    if not (isinstance(args.get("oldText"), str) and isinstance(args.get("newText"), str)):
        return args
    merged = list(args["edits"]) if isinstance(args.get("edits"), list) else []
    merged.append({"oldText": args["oldText"], "newText": args["newText"]})
    rest = {key: value for key, value in args.items() if key not in ("oldText", "newText")}
    return {**rest, "edits": merged}


async def _access(fs: FileSystem, p: str, path: str, signal: RunSignal | None) -> None:
    """Pi's single access stage. EXEC-009 when the provider has it; otherwise the disclosed
    FALLBACK (spec/execution.md section 13.5): `check_readable`, or no access stage at all when
    that is unsupported too. On failure the signal is checked BEFORE the error is reported."""
    answer = await fs.check_read_write(p)
    if isinstance(answer, Err) and answer.error.code == FsErrorCode.NOT_SUPPORTED:
        answer = await fs.check_readable(p)
        if isinstance(answer, Err) and answer.error.code == FsErrorCode.NOT_SUPPORTED:
            return
    if isinstance(answer, Err):
        check_abort(signal)
        raise BuiltinToolError(f"Could not edit file: {path}. {cause(answer.error.code)}.")


def create_edit_tool(fs: FileSystem) -> ToolDefinition:
    """The `edit` tool bound to one `ctx.fs` provider."""

    async def execute(
        tool_call_id: str, arguments: dict[str, Any], signal: RunSignal | None
    ) -> ToolResult:
        edits = arguments.get("edits")
        if not isinstance(edits, list) or not edits:
            raise BuiltinToolError(EMPTY_EDITS)
        path: str = arguments["path"]
        p = preprocess_path(path)

        async def work() -> tuple[str, dict[str, Any]]:
            check_abort(signal)
            await _access(fs, p, path, signal)
            check_abort(signal)
            data = await fs.read_binary_file(p)
            if isinstance(data, Err):
                raise BuiltinToolError(f"Cannot read {path}: {cause(data.error.code)}")
            raw = to_units(decode_utf8(data.value))
            check_abort(signal)
            bom, content = edit_diff.split_bom(raw)
            ending = edit_diff.detect_line_ending(content)
            normalized = edit_diff.normalize_to_lf(content)
            pairs = [(to_units(edit["oldText"]), to_units(edit["newText"])) for edit in edits]
            base, new = edit_diff.apply_edits(normalized, pairs, path)
            check_abort(signal)
            final = from_units(bom + edit_diff.restore_line_endings(new, ending))
            written = await fs.write_file(p, encode_utf8(final))
            if isinstance(written, Err):
                raise BuiltinToolError(f"Cannot write {path}: {cause(written.error.code)}")
            check_abort(signal)
            diff, first_changed = edit_diff.generate_diff_string(base, new)
            patch = edit_diff.generate_unified_patch(to_units(path), base, new)
            details = {
                "diff": from_units(diff),
                "patch": from_units(patch),
                "firstChangedLine": first_changed,
            }
            return f"Successfully replaced {len(edits)} block(s) in {path}.", details

        text, details = await with_mutation_queue(fs, p, path, work)
        return ToolResult(
            tool_call_id=tool_call_id,
            content=(TextBlock(text=text),),
            tool_name="edit",
            details=details,
        )

    return ToolDefinition(
        name="edit",
        label="edit",
        description=EDIT_DESCRIPTION,
        parameters=EDIT_PARAMETERS,
        execute=execute,
        prepare_arguments=prepare_edit_arguments,
        wants_signal=True,
    )
