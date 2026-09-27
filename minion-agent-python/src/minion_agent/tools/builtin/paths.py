"""The shared `read`/`ls` path-argument pipeline (`TOOL-026`) and the closed `R010-B` error
vocabulary (`TOOL-039`), spec/tools.md "WP-13.1".

Steps 1-4 happen here; step 5 hands the result to a READ-ONLY `ctx.fs` operation, whose own
`resolve_local_path` does tilde expansion, absolute normalization and cwd-relative resolution.
"""

from __future__ import annotations

import os
import re

from ...execution import FsErrorCode, file_url_to_path

OPERATION_ABORTED = "Operation aborted"

CAUSE_PHRASES: dict[FsErrorCode, str] = {
    FsErrorCode.NOT_FOUND: "no such file or directory",
    FsErrorCode.PERMISSION_DENIED: "permission denied",
    FsErrorCode.NOT_DIRECTORY: "not a directory",
    FsErrorCode.IS_DIRECTORY: "is a directory",
    FsErrorCode.INVALID: "invalid path",
    FsErrorCode.NOT_SUPPORTED: "not supported by this provider",
    FsErrorCode.UNKNOWN: "unknown filesystem error",
}
"""`R010-B`'s closed `FsErrorCode -> cause phrase` table. `aborted` is not in it: cancellation
always surfaces as `OPERATION_ABORTED`, Pi's own stable template."""


class BuiltinToolError(Exception):
    """A generated tool error. Layer 06 surfaces `str(error)` as the error result's text, with
    `details: {}` -- the same path pinned Pi's thrown `Error(message)` takes."""


def aborted() -> BuiltinToolError:
    return BuiltinToolError(OPERATION_ABORTED)


def cause(code: FsErrorCode) -> str:
    return CAUSE_PHRASES[code]


# The CLOSED Unicode space set Pi normalizes (utils/paths.ts UNICODE_SPACES); nothing is trimmed.
_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")

# Pi's normalizeWindowsShellPath regex, /^\/(?:mnt\/|cygdrive\/)?([a-z])(?:\/(.*))?$/i, with its JS
# meaning kept: `i` without the `u` flag folds ASCII letters only (re.ASCII -- Python's own
# IGNORECASE would also match U+017F and U+212A), `.` stops at every JS line terminator, and `$` is
# the end of input (`\Z` -- Python's `$` also matches before a trailing "\n").
_WINDOWS_SHELL_PATH = re.compile(
    "^/(?:mnt/|cygdrive/)?([a-z])(?:/([^\n\r\u2028\u2029]*))?\\Z", re.IGNORECASE | re.ASCII
)


def _normalize_windows_shell_path(path: str) -> str:
    if not path.startswith("/") or path.startswith("//") or "\\" in path:
        return path
    match = _WINDOWS_SHELL_PATH.match(path)
    if match is None:
        return path
    suffix = (match.group(2) or "").replace("/", "\\")
    return f"{match.group(1).upper()}:\\{suffix}"


def preprocess_path(path: str) -> str:
    """`TOOL-026` steps 1-4. Returns the string step 5 passes to `ctx.fs`. A malformed `file://`
    URL is rejected HERE, before any `ctx.fs` access (`R002-A`), as `"Cannot access <path>: invalid
    path"`, where `<path>` is the step-4 input string (spec/tools.md, IMPL-C003)."""
    working = _UNICODE_SPACES.sub(" ", path)
    if working.startswith("@"):
        working = working[1:]
    if os.name == "nt":
        working = _normalize_windows_shell_path(working)
    if working.startswith("file://"):
        try:
            return file_url_to_path(working)
        except (ValueError, OSError) as exc:
            raise BuiltinToolError(
                f"Cannot access {working}: {cause(FsErrorCode.INVALID)}"
            ) from exc
    return working
