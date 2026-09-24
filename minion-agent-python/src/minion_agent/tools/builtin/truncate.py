"""Head truncation shared by the Layer 13 tools (pinned Pi `core/tools/truncate.ts`).

Two independent ceilings, whichever is hit first: `DEFAULT_MAX_LINES` lines and `DEFAULT_MAX_BYTES`
UTF-8 bytes. Only complete lines are kept; a first line that alone exceeds the byte ceiling yields
empty content with `first_line_exceeds_limit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ._js import to_fixed

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024


def utf8_len(text: str) -> int:
    """`Buffer.byteLength(text, "utf-8")`. Text decoded from bytes never carries a lone surrogate,
    so this equals the UTF-8 encoding's length."""
    return len(text.encode("utf-8", "surrogatepass"))


def format_size(size: int) -> str:
    """Pi's `formatSize`: `B` below 1 KiB, one-decimal `KB` below 1 MiB, one-decimal `MB` above.
    `1048575` renders `"1024.0KB"` exactly as Pi does."""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{to_fixed(size / 1024, 1)}KB"
    return f"{to_fixed(size / (1024 * 1024), 1)}MB"


@dataclass(frozen=True, slots=True)
class Truncation:
    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    total_bytes: int
    output_lines: int
    first_line_exceeds_limit: bool

    def details(self) -> dict[str, Any]:
        """`details.truncation`, in the shape spec/tools.md `TOOL-025` fixes."""
        return {
            "truncated": self.truncated,
            "truncated_by": self.truncated_by,
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
            "first_line_exceeds_limit": self.first_line_exceeds_limit,
        }


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def truncate_head(
    content: str, *, max_lines: float = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> Truncation:
    total_bytes = utf8_len(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return Truncation(content, False, None, total_lines, total_bytes, total_lines, False)
    if utf8_len(lines[0]) > max_bytes:
        return Truncation("", True, "bytes", total_lines, total_bytes, 0, True)
    kept: list[str] = []
    kept_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    for index, line in enumerate(lines):
        if index >= max_lines:
            break
        line_bytes = utf8_len(line) + (1 if index > 0 else 0)
        if kept_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        kept.append(line)
        kept_bytes += line_bytes
    if len(kept) >= max_lines and kept_bytes <= max_bytes:
        truncated_by = "lines"
    return Truncation(
        "\n".join(kept), True, truncated_by, total_lines, total_bytes, len(kept), False
    )
