"""`bash` output accumulation and tail truncation (`TOOL-035`; pinned Pi
`core/tools/output-accumulator.ts` and `truncate.ts`'s `truncateTail`), spec/tools.md WP-13.3
"Output".

The accumulator decodes the MERGED chunk sequence of both pipes with one streaming decoder, keeps
the counters and a bounded rolling tail, and answers when the full-output file is needed. It owns no
I/O: the tool persists the raw bytes through `ctx.fs` (the file's `MINION_ARCHITECTURAL_MAPPING`).
"""

from __future__ import annotations

import codecs
from typing import Any, Literal

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, utf8_len

_BOM = chr(0xFEFF)


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _bytes_from_end(text: str, max_bytes: int) -> str:
    """Pi's `truncateStringToBytesFromEnd`: the last `max_bytes` bytes, starting at a UTF-8
    character boundary."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    start = len(data) - max_bytes
    while start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    return data[start:].decode("utf-8")


def truncate_tail(
    content: str, *, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> dict[str, Any]:
    """Pi's `truncateTail`, returning its `TruncationResult` with Pi's own (closed) key set."""
    total_bytes = utf8_len(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _result(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes, False,
            max_lines, max_bytes,
        )  # fmt: skip
    kept: list[str] = []
    kept_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False
    for line in reversed(lines):
        if len(kept) >= max_lines:
            break
        line_bytes = utf8_len(line) + (1 if kept else 0)
        if kept_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not kept:
                partial = _bytes_from_end(line, max_bytes)
                kept.insert(0, partial)
                kept_bytes = utf8_len(partial)
                last_line_partial = True
            break
        kept.insert(0, line)
        kept_bytes += line_bytes
    if len(kept) >= max_lines and kept_bytes <= max_bytes:
        truncated_by = "lines"
    output = "\n".join(kept)
    return _result(
        output, True, truncated_by, total_lines, total_bytes, len(kept), utf8_len(output),
        last_line_partial, max_lines, max_bytes,
    )  # fmt: skip


def _result(
    content: str,
    truncated: bool,
    truncated_by: str | None,
    total_lines: int,
    total_bytes: int,
    output_lines: int,
    output_bytes: int,
    last_line_partial: bool,
    max_lines: int,
    max_bytes: int,
) -> dict[str, Any]:
    return {
        "content": content,
        "truncated": truncated,
        "truncatedBy": truncated_by,
        "totalLines": total_lines,
        "totalBytes": total_bytes,
        "outputLines": output_lines,
        "outputBytes": output_bytes,
        "lastLinePartial": last_line_partial,
        "firstLineExceedsLimit": False,
        "maxLines": max_lines,
        "maxBytes": max_bytes,
    }


class OutputAccumulator:
    """Pi's `OutputAccumulator`, minus its own temp-file stream (the tool owns that, over `ctx.fs`).

    - Decoding: ONE streaming decoder over the merged chunks -- WHATWG UTF-8 with replacement,
      stripping exactly one leading BOM of the whole stream, even when split across chunks
      (`WP133-AUD-R002`). CPython's incremental `utf-8` decoder with `errors="replace"` replaces
      each maximal invalid subpart; the BOM is stripped only when the stream's FIRST decoded
      character is U+FEFF, which only a complete `EF BB BF` at the start produces. An incomplete
      prefix at EOF (`EF`, `EF BB`) therefore flushes as U+FFFD, as Pi's `TextDecoder` does
      (`WP133-I001`; CPython's `utf-8-sig` would silently drop it).
    - Rolling tail: trimmed to its last `2 * max_bytes` bytes once over `4 * max_bytes`, with the
      line-boundary flag; a snapshot drops a partial first line only when the tail holds a newline
      (`WP133-CON-R002`).
    """

    def __init__(self, *, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES):
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self._max_rolling_bytes = max(max_bytes * 2, 1)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._at_stream_start = True
        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self.total_raw_bytes = 0
        self.total_decoded_bytes = 0
        self._completed_lines = 0
        self.total_lines = 0
        self.last_line_bytes = 0
        """Pi's `getLastLineBytes()`: the decoded byte length of the current (last) line."""
        self._has_open_line = False
        self._finished = False

    def append(self, data: bytes) -> None:
        if self._finished:
            raise RuntimeError("Cannot append to a finished output accumulator")
        self.total_raw_bytes += len(data)
        self._append_decoded(self._decode(data, final=False))

    def finish(self) -> None:
        """Flush the decoder: incomplete trailing bytes become U+FFFD."""
        if self._finished:
            return
        self._finished = True
        self._append_decoded(self._decode(b"", final=True))

    def _decode(self, data: bytes, *, final: bool) -> str:
        text = self._decoder.decode(data, final)
        if self._at_stream_start and text:
            self._at_stream_start = False
            if text[0] == _BOM:
                return text[1:]  # exactly one leading BOM of the whole stream
        return text

    @property
    def should_use_temp_file(self) -> bool:
        return (
            self.total_raw_bytes > self.max_bytes
            or self.total_decoded_bytes > self.max_bytes
            or self.total_lines > self.max_lines
        )

    def snapshot(self) -> dict[str, Any]:
        """Pi's `snapshot().truncation`: `truncateTail` of the tail text, with `truncated`,
        `truncatedBy` and the totals recomputed from the WHOLE stream's counters."""
        tail = truncate_tail(
            self._snapshot_text(), max_lines=self.max_lines, max_bytes=self.max_bytes
        )
        truncated = self.total_lines > self.max_lines or self.total_decoded_bytes > self.max_bytes
        truncated_by = None
        if truncated:
            truncated_by = tail["truncatedBy"] or (
                "bytes" if self.total_decoded_bytes > self.max_bytes else "lines"
            )
        return {
            **tail,
            "truncated": truncated,
            "truncatedBy": truncated_by,
            "totalLines": self.total_lines,
            "totalBytes": self.total_decoded_bytes,
            "maxLines": self.max_lines,
            "maxBytes": self.max_bytes,
        }

    def _append_decoded(self, text: str) -> None:
        if not text:
            return
        size = utf8_len(text)
        self.total_decoded_bytes += size
        self._tail_text += text
        self._tail_bytes += size
        if self._tail_bytes > self._max_rolling_bytes * 2:
            self._trim_tail()
        newlines = text.count("\n")
        if newlines == 0:
            self.last_line_bytes += size
            self._has_open_line = True
        else:
            self._completed_lines += newlines
            rest = text[text.rindex("\n") + 1 :]
            self.last_line_bytes = utf8_len(rest)
            self._has_open_line = len(rest) > 0
        self.total_lines = self._completed_lines + (1 if self._has_open_line else 0)

    def _trim_tail(self) -> None:
        data = self._tail_text.encode("utf-8")
        if len(data) <= self._max_rolling_bytes:
            self._tail_bytes = len(data)
            return
        start = len(data) - self._max_rolling_bytes
        while start < len(data) and (data[start] & 0xC0) == 0x80:
            start += 1
        if start != 0:
            self._tail_starts_at_line_boundary = data[start - 1] == 0x0A
        self._tail_text = data[start:].decode("utf-8")
        self._tail_bytes = utf8_len(self._tail_text)

    def _snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text
        first_newline = self._tail_text.find("\n")
        return self._tail_text if first_newline == -1 else self._tail_text[first_newline + 1 :]
