"""Frontmatter extraction and the Minion YAML subset (spec/harness.md WP-14.1 `HAR-010`).

Extraction is direct Pi parity (`parseFrontmatter` in pinned Pi's harness `skills.ts`). It works on
UTF-16 code units, as the JavaScript slices do. The YAML text `T` is then read by the
Minion frontmatter subset (rules 1-7, DIV-004): inside the subset the value equals `yaml@2.9.0`'s,
and outside it the reader raises `FrontmatterError`, which the loader reports as `parse_failed`.
This is a reader for skill metadata only, not a YAML API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from ..auth.js_json import js_trim
from ..tools.builtin._utf16 import from_units, to_units

type Value = str | bool | Number | dict[str, Value] | list[Value] | None


@dataclass(frozen=True, slots=True)
class Number:
    """A YAML core-schema number: only "not a string" is observable."""

    text: str


class FrontmatterError(ValueError):
    """`T` is outside the subset or violates a subset rule (reported as `parse_failed`)."""


@dataclass(frozen=True, slots=True)
class Frontmatter:
    """The extracted YAML text (`None`: no frontmatter at all) and the body."""

    yaml: str | None
    body: str


def extract(content: str) -> Frontmatter:
    """Pi's extraction: CRLF/CR -> LF, no BOM strip, `T = text[4:endIndex]` with the first `\\n---`
    at code-unit index >= 3, and the body after it trimmed by JS `String.prototype.trim`."""
    units = to_units(content.replace("\r\n", "\n").replace("\r", "\n"))
    if not units.startswith("---"):
        return Frontmatter(None, from_units(units))
    end = units.find("\n---", 3)
    if end == -1:
        return Frontmatter(None, from_units(units))
    return Frontmatter(from_units(units[4:end]), js_trim(from_units(units[end + 4 :])))


# ---- the subset reader --------------------------------------------------------------------------

_KEY: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_RESERVED_KEYS: Final = frozenset(
    {"null", "Null", "NULL", "true", "True", "TRUE", "false", "False", "FALSE"}
)
_NULL: Final = re.compile(r"(?:~|[Nn]ull|NULL)?\Z")
_BOOL: Final = re.compile(r"(?:[Tt]rue|TRUE|[Ff]alse|FALSE)\Z")
_NUMBERS: Final = tuple(
    re.compile(p)
    for p in (
        r"0o[0-7]+\Z",
        r"[-+]?[0-9]+\Z",
        r"0x[0-9a-fA-F]+\Z",
        r"(?:[-+]?\.(?:inf|Inf|INF)|\.nan|\.NaN|\.NAN)\Z",
        r"[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+\Z",
        r"[-+]?(?:\.[0-9]+|[0-9]+\.[0-9]*)\Z",
    )
)
_PLAIN_FIRST_FORBIDDEN: Final = frozenset("-?:,[]{}#&*!|>'\"%@`")
# JavaScript `\s` (WhiteSpace + LineTerminator), for the entry-line shape `[^:\s]+:`.
_JS_CODE_POINTS: Final = (
    0x09,
    0x0A,
    0x0B,
    0x0C,
    0x0D,
    0x20,
    0xA0,
    0x1680,
    *range(0x2000, 0x200B),
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
    0xFEFF,
)
_JS_SPACE: Final = "".join(f"\\u{cp:04x}" for cp in _JS_CODE_POINTS)
_ENTRY: Final = re.compile(f"([^:{_JS_SPACE}]+):(?=[ \\t]|\\Z)")
_COMMENT: Final = re.compile(r"[ \t]#")
_AFTER_QUOTE: Final = re.compile(r"(?:[ \t]+(?:#.*)?)?\Z", re.DOTALL)
_BLOCK_HEADER: Final = re.compile(r"([|>])([+-]?)(?:[ \t]+#.*)?\Z", re.DOTALL)
_HEX: Final = {"x": 2, "u": 4, "U": 8}
_SIMPLE_ESCAPES: Final = {"\\": "\\", '"': '"', "/": "/", "t": "\t", "n": "\n"}


def _forbidden(cp: int) -> bool:
    """Rule 1: characters the subset refuses in `T` and in any decoded value."""
    return (
        (cp < 0x20 and cp not in (0x09, 0x0A))
        or cp == 0x7F
        or 0x80 <= cp <= 0x9F
        or cp in (0x2028, 0x2029, 0xFEFF)
        or 0xD800 <= cp <= 0xDFFF
    )


def _resolve_plain(text: str) -> Value:
    if _NULL.match(text):
        return None
    if _BOOL.match(text):
        return text[0] in "tT"
    if any(p.match(text) for p in _NUMBERS):
        return Number(text)
    return text


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _blank(line: str) -> bool:
    return line.strip(" \t") == ""


def _comment(line: str) -> bool:
    return line.lstrip(" \t").startswith("#")


def _blank_or_comment(line: str) -> bool:
    return _blank(line) or _comment(line)


def _plain_line(raw: str) -> str:
    """The same-line plain text: up to a `[ \\t]#` comment, trailing SP/TAB removed."""
    m = _COMMENT.search(raw)
    return (raw[: m.start()] if m else raw).rstrip(" \t")


def _check_plain_segment(text: str, first: bool) -> None:
    # never empty here: a same-line value is non-empty and not a comment, and a continuation line is
    # non-blank, so its text keeps at least one character
    safe_lead = text[0] in "-?:" and len(to_units(text)) > 1 and to_units(text)[1] not in " \t"
    if text[0] in _PLAIN_FIRST_FORBIDDEN and not (first and safe_lead):
        raise FrontmatterError("plain scalar starts with an indicator")
    if re.search(r":[ \t]", text) or text.endswith(":"):
        raise FrontmatterError("plain scalar contains ': ' or ends with ':'")
    if "\t" in text:
        raise FrontmatterError("tab inside a plain scalar")


def _decode_double(body: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        e = body[i + 1] if i + 1 < len(body) else ""
        if e in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[e])
            i += 2
            continue
        width = _HEX.get(e)
        if width is None:
            raise FrontmatterError("unsupported escape")
        digits = body[i + 2 : i + 2 + width]
        if not re.fullmatch(f"[0-9A-Fa-f]{{{width}}}", digits):
            raise FrontmatterError("bad hex escape")
        cp = int(digits, 16)
        if cp > 0x10FFFF or _forbidden(cp):
            raise FrontmatterError("escape outside the subset character set")
        out.append(chr(cp))
        i += 2 + width
    return "".join(out)


class _Reader:
    def __init__(self, text: str) -> None:
        for ch in text:
            if _forbidden(ord(ch)):
                raise FrontmatterError("forbidden character")
        lines = text.split("\n")
        # a final line break terminates the last line; it does not start another one
        self.last_line_terminated = len(lines) > 1 and lines[-1] == ""
        if self.last_line_terminated:
            lines.pop()
        for line in lines:
            if line.lstrip(" ").startswith("\t"):
                raise FrontmatterError("tab in leading whitespace")
        self.lines = lines

    def next_content_indent(self, start: int) -> int:
        for line in self.lines[start:]:
            if not _blank_or_comment(line):
                return _indent(line)
        return -1

    def same_line_scalar(
        self, rest: str, allow_continuation: bool, i: int, n: int
    ) -> tuple[Value, int]:
        if rest.startswith('"'):
            j = 1
            while j < len(rest) and rest[j] != '"':
                j += 2 if rest[j] == "\\" else 1
            if j >= len(rest):
                raise FrontmatterError("unterminated or multi-line double-quoted scalar")
            if not _AFTER_QUOTE.match(rest, j + 1):
                raise FrontmatterError("content after a double-quoted scalar")
            return _decode_double(rest[1:j]), 0
        if rest.startswith("'"):
            j = 1
            out: list[str] = []
            while True:
                if j >= len(rest):
                    raise FrontmatterError("unterminated or multi-line single-quoted scalar")
                if rest[j] == "'":
                    if rest[j + 1 : j + 2] == "'":
                        out.append("'")
                        j += 2
                        continue
                    break
                out.append(rest[j])
                j += 1
            if not _AFTER_QUOTE.match(rest, j + 1):
                raise FrontmatterError("content after a single-quoted scalar")
            return "".join(out), 0
        first = _plain_line(rest)
        _check_plain_segment(first, True)
        # a comment ends a plain scalar: no continuation may follow it
        if not allow_continuation or _COMMENT.search(rest):
            if self.next_content_indent(i + 1) > n:
                raise FrontmatterError("continuation not allowed here")
            return _resolve_plain(first), 0
        text = first
        blanks = 0
        j = i + 1
        while j < len(self.lines):
            line = self.lines[j]
            if _blank(line):
                blanks += 1
                j += 1
                continue
            if _indent(line) <= n:
                break
            if _comment(line):
                raise FrontmatterError("comment inside a multi-line plain scalar")
            # indentation is SP only; any other whitespace is scalar content (WP141-C001)
            stripped = line.lstrip(" ")
            seg = _plain_line(stripped)
            if seg != stripped.rstrip(" \t"):
                raise FrontmatterError("comment after a continuation line")
            _check_plain_segment(seg, False)
            text += f" {seg}" if blanks == 0 else "\n" * blanks + seg
            blanks = 0
            j += 1
        last = j - 1
        while last > i and _blank(self.lines[last]):
            last -= 1
        return _resolve_plain(text), last - i

    def block_scalar(self, header: str, i: int, n: int) -> tuple[Value, int]:
        m = _BLOCK_HEADER.match(header)
        if not m:
            raise FrontmatterError("unsupported block scalar header")
        style, chomp = m.group(1), m.group(2)
        j = i + 1
        while j < len(self.lines) and (_blank(self.lines[j]) or _indent(self.lines[j]) > n):
            j += 1
        region = self.lines[i + 1 : j]
        first_text = next((line for line in region if not _blank(line)), None)
        content_indent = -1 if first_text is None else _indent(first_text)
        content: list[str | None] = []
        for line in region:
            if _blank(line):
                # a TAB is impossible here: a whitespace-only line with a TAB has spaces then a TAB,
                # which the line rule (rule 2) has already rejected
                if (len(line) > 0) if content_indent == -1 else (len(line) > content_indent):
                    raise FrontmatterError(
                        "whitespace-only block scalar line beyond the content indentation"
                    )
                content.append(None)
                continue
            if _indent(line) < content_indent:
                raise FrontmatterError("under-indented block scalar line")
            text = line[content_indent:]
            if style == ">" and text[:1] in (" ", "\t"):
                raise FrontmatterError("more-indented line in a folded scalar")
            content.append(text)
        # trailing empty lines count only if a line break follows them in T (rule 7)
        trailing = 0
        while content and content[-1] is None:
            content.pop()
            trailing += 1
        if trailing > 0 and j == len(self.lines) and not self.last_line_terminated:
            trailing -= 1
        consumed = j - 1 - i
        if not content:
            return ("\n" * trailing if chomp == "+" else ""), consumed
        if style == "|":
            body = "\n".join(piece if piece is not None else "" for piece in content)
        else:
            parts: list[str] = []
            pending = 0
            for k, piece in enumerate(content):
                if piece is None:
                    pending += 1
                    continue
                if k > 0:
                    parts.append(" " if pending == 0 else "\n" * pending)
                parts.append(piece)
                pending = 0
            body = "".join(parts)
        if chomp == "-":
            return body, consumed
        if chomp == "+":
            return body + "\n" + "\n" * trailing, consumed
        return body + "\n", consumed

    def sequence(self, i: int, m: int) -> tuple[Value, int]:
        items: list[Value] = []
        j = i
        while j < len(self.lines):
            line = self.lines[j]
            if _blank_or_comment(line):
                j += 1
                continue
            ind = _indent(line)
            if ind < m:
                break
            if ind > m:
                raise FrontmatterError("unexpected deeper content in a sequence")
            body = line[m:]
            if not body.startswith("- "):
                break
            rest = body[2:].lstrip(" \t")
            if not rest or rest.startswith("#"):
                raise FrontmatterError("empty or nested sequence item")
            value, _ = self.same_line_scalar(rest, False, j, m)
            items.append(value)
            j += 1
        return items, j - 1 - i

    def mapping(self, i: int, n: int) -> tuple[Value, int]:
        entries: dict[str, Value] = {}
        j = i
        while j < len(self.lines):
            line = self.lines[j]
            if _blank_or_comment(line):
                j += 1
                continue
            ind = _indent(line)
            if ind < n:
                break
            if ind > n:
                raise FrontmatterError("unexpected indentation")
            body = line[n:]
            km = _ENTRY.match(body)
            if not km:
                raise FrontmatterError("line is not a mapping entry")
            key = km.group(1)
            if not _KEY.match(key) or key in _RESERVED_KEYS:
                raise FrontmatterError("unsupported key")
            if key in entries:
                raise FrontmatterError("duplicate key")
            rest = body[len(key) + 1 :]
            if rest and rest[0] != " ":
                raise FrontmatterError("the separator after ':' must start with a space")
            rest = rest.lstrip(" \t")
            value: Value
            consumed = 0
            if not rest or rest.startswith("#"):
                k = j + 1
                while k < len(self.lines) and _blank_or_comment(self.lines[k]):
                    k += 1
                nxt = self.next_content_indent(j + 1)
                if nxt > n and _ENTRY.match(self.lines[k][nxt:]):
                    value, consumed = self.mapping(k, nxt)
                    consumed += k - j
                elif nxt >= n and nxt != -1 and self.lines[k][nxt:].startswith("- "):
                    value, consumed = self.sequence(k, nxt)
                    consumed += k - j
                elif nxt > n:
                    raise FrontmatterError("a value on the following line is outside the subset")
                else:
                    value = None
            elif rest[0] in "|>":
                value, consumed = self.block_scalar(rest, j, n)
            else:
                value, consumed = self.same_line_scalar(rest, True, j, n)
            entries[key] = value
            j += consumed + 1
        return entries, j - 1 - i

    def document(self) -> dict[str, Value]:
        k = 0
        while k < len(self.lines) and _blank_or_comment(self.lines[k]):
            k += 1
        if k == len(self.lines):
            return {}
        if _indent(self.lines[k]) != 0:
            raise FrontmatterError("the top-level mapping must start at column 0")
        # Mapping(0) ends only at the end of T (no line is indented below 0, and a deeper line is
        # rejected), so nothing can follow it
        value, _ = self.mapping(k, 0)
        assert isinstance(value, dict)
        return value


def read_subset(text: str) -> dict[str, Value]:
    """The frontmatter mapping (empty for an empty document); raises `FrontmatterError` outside
    the subset."""
    return _Reader(text).document()
