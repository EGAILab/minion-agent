"""`ignore@7.0.5` (`ignores()`, default options), ported (spec/harness.md WP-14.1 `HAR-011`).

Pinned Pi matches skill-discovery ignore rules with the npm `ignore` package at 7.0.5. This module
reproduces its observable `ignores(path)` result. The package compiles each gitignore pattern into
a JavaScript regular expression (`REPLACERS`, then the trailing-wildcard rule) and matches it with
the `i` flag. Three JavaScript runtime properties are reproduced explicitly rather than borrowed
from Python's `re`:

- matching is over UTF-16 **code units** (so `?` matches one unit, never a whole astral character);
- the `i` flag is ECMAScript `Canonicalize` (non-Unicode mode): a code unit is replaced by its
  `toUpperCase` (Unicode 16.0, the pinned runtime's version) only when that is a single unit and
  does not map a non-ASCII unit to ASCII -- unlike `re.IGNORECASE`, U+212A KELVIN SIGN never
  matches `k`;
- `\\s`, `.`, `$`, escapes and character classes keep their JavaScript meanings.

Only `ignores()` is ported, with the default `ignorecase: true`, and only for paths the loader
passes (root-relative, never empty and never `/`-led -- DIV-005 filters those before matching).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..tools.builtin._utf16 import to_units
from ..tools.builtin.collation import pinned_collation

# ECMAScript WhiteSpace + LineTerminator: the set JS `\s` (and `String.prototype.trim`) uses.
_JS_CODE_POINTS = (
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
_JS_SPACE = "".join(map(chr, _JS_CODE_POINTS))
_S = "[" + "".join(f"\\u{ord(c):04x}" for c in _JS_SPACE) + "]"
_NOT_S = "[^" + _S[1:]
# JavaScript `.` without the `s` flag: any code unit except the four line terminators.
_DOT = "[^\\n\\r\\u2028\\u2029]"


def _js(pattern: str) -> re.Pattern[str]:
    """A replacer regex written in JavaScript syntax: `\\s`/`.`/`$` take their JS meanings."""
    out = []
    i = 0
    in_class = False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            out.append(_S[1:-1] if nxt == "s" and in_class else _S if nxt == "s" else c + nxt)
            i += 2
            continue
        if in_class:
            in_class = c != "]"
            out.append(c)
        elif c == "[":
            in_class = True
            out.append(c)
        elif c == ".":
            out.append(_DOT)
        elif c == "$":
            out.append(r"\Z")
        else:
            out.append(c)
        i += 1
    return re.compile("".join(out))


def _clean_range_back_slash(slashes: str) -> str:
    return slashes[: len(slashes) - len(slashes) % 2]


_REGEXP_RANGE = re.compile(r"([0-z])-([0-z])")


def _sanitize_range(rng: str) -> str:
    """Drop out-of-order `x-y` ranges (fatal in a JS RegExp, harmless in gitignore)."""
    return _REGEXP_RANGE.sub(
        lambda m: m.group(0) if ord(m.group(1)) <= ord(m.group(2)) else "", rng
    )


def _range_replacer(m: re.Match[str]) -> str:
    lead, rng, end, close = m.group(1), m.group(2), m.group(3), m.group(4)
    if lead == "\\":
        return f"\\[{rng}{_clean_range_back_slash(end)}{close}"
    if close == "]":
        return f"[{_sanitize_range(rng)}{end}]" if len(end) % 2 == 0 else "[]"
    return "[]"


def _starting(body: str) -> str:
    return "^" if _js(r"\/(?!$)").search(body) else "(?:^|\\/)"


def _globstar(m: re.Match[str]) -> str:
    return "(?:\\/[^\\/]+)*" if m.start() + 6 < len(m.string) else "\\/.+"


def _ending(m: re.Match[str]) -> str:
    match = m.group(0)
    return f"{match}$" if match.endswith("/") else f"{match}(?=$|\\/$)"


type _Replacer = tuple[re.Pattern[str], Callable[[re.Match[str]], str], bool]
"""(JS regex, replacement, global?) -- `String.prototype.replace` without `g` replaces one match."""

_REPLACERS: list[_Replacer] = [
    (_js("^\ufeff"), lambda m: "", False),
    (
        _js(r"((?:\\\\)*?)(\\?\s+)$"),
        lambda m: m.group(1) + (" " if m.group(2).startswith("\\") else ""),
        False,
    ),
    (_js(r"(\\+?)\s"), lambda m: _clean_range_back_slash(m.group(1)) + " ", True),
    (_js(r"[\\$.|*+(){^]"), lambda m: "\\" + m.group(0), True),
    (_js(r"(?!\\)\?"), lambda m: "[^/]", True),
    (_js(r"^\/"), lambda m: "^", False),
    (_js(r"\/"), lambda m: "\\/", True),
    (_js(r"^\^*\\\*\\\*\\\/"), lambda m: "^(?:.*\\/)?", False),
    # index 8: the "starting" replacer needs the body; bound per pattern in `_regex_prefix`
    (_js(r"\\\/\\\*\\\*(?=\\\/|$)"), _globstar, True),
    (
        _js(r"(^|[^\\]+)(\\\*)+(?=.+)"),
        lambda m: m.group(1) + m.group(2).replace("\\*", "[^\\/]*"),
        True,
    ),
    (_js(r"\\\\\\(?=[$.|*+(){^])"), lambda m: "\\", True),
    (_js(r"\\\\"), lambda m: "\\", True),
    (_js(r"(\\)?\[([^\]/]*?)(\\*)($|\])"), _range_replacer, True),
    (_js(r"(?:[^*])$"), _ending, False),
]
_STARTING = _js(r"^(?=[^^])")
_TRAILING_WILDCARD = _js(r"(^|\\\/)?\\\*$")
_BLANK_LINE = _js(r"^\s+$")
_INVALID_TRAILING_BACKSLASH = _js(r"(?:[^\\]|^)\\$")


def _replace(
    regex: re.Pattern[str], repl: Callable[[re.Match[str]], str], text: str, everywhere: bool
) -> str:
    return regex.sub(repl, text, count=0 if everywhere else 1)


def _regex_prefix(body: str) -> str:
    """`makeRegexPrefix`: the REPLACERS chain over the (code-unit) pattern body."""
    text = body
    for index, (regex, repl, everywhere) in enumerate(_REPLACERS):
        if index == 8:
            text = _replace(_STARTING, lambda m: _starting(body), text, False)
        text = _replace(regex, repl, text, everywhere)
    return text


def _trailing_wildcard(m: re.Match[str]) -> str:
    p1 = m.group(1)
    return f"{p1 + '[^/]+' if p1 else '[^/]*'}(?=$|\\/$)"


# ---- JavaScript regex source -> Python, with Canonicalize-based case-insensitivity ---------------


@functools.cache
def _canonicalize_table() -> tuple[str, ...]:
    """ECMAScript `Canonicalize(ch)` (ignoreCase, non-Unicode mode) for every code unit."""
    upper = pinned_collation().upper_unicode16
    table = []
    for unit in range(0x10000):
        ch = chr(unit)
        if 0xD800 <= unit <= 0xDFFF:
            table.append(ch)
            continue
        u = upper(ch)
        if len(to_units(u)) != 1 or (unit >= 128 and ord(u) < 128):
            table.append(ch)
        else:
            table.append(u)
    return tuple(table)


def canonicalize(units: str) -> str:
    table = _canonicalize_table()
    return "".join(table[ord(c)] for c in units)


_CLASS_ESCAPES = {
    "d": frozenset(range(0x30, 0x3A)),
    "w": frozenset([*range(0x30, 0x3A), *range(0x41, 0x5B), *range(0x61, 0x7B), 0x5F]),
    "s": frozenset(ord(c) for c in _JS_SPACE),
}
_ALL_UNITS = frozenset(range(0x10000))
_OCTAL = re.compile(r"[0-7]{1,3}")
_CONTROL_ESCAPES = {"t": 0x09, "n": 0x0A, "v": 0x0B, "f": 0x0C, "r": 0x0D}


@dataclass(frozen=True, slots=True)
class _Escape:
    units: frozenset[int] | None  # a set (class-like) escape, or None
    unit: int | None  # a single code unit
    assertion: str | None  # \b / \B outside a class
    length: int  # source characters consumed, including the backslash


def _escape(src: str, i: int, in_class: bool) -> _Escape:
    """One JavaScript (non-Unicode mode) escape at `src[i] == "\\"`."""
    if i + 1 >= len(src):
        # JS: "\ at end of pattern" is a SyntaxError (never produced: the replacers always end the
        # source with a lookahead or a wildcard rule)
        raise InvalidIgnorePattern("\\ at end of pattern")
    c = src[i + 1]
    lower = c.lower()
    if lower in _CLASS_ESCAPES:
        units = _CLASS_ESCAPES[lower]
        return _Escape(units if c.islower() else _ALL_UNITS - units, None, None, 2)
    if c in "bB":
        if in_class and c == "b":
            return _Escape(None, 0x08, None, 2)
        if not in_class:
            return _Escape(None, None, "\\b" if c == "b" else "\\B", 2)
    if c in _CONTROL_ESCAPES:
        return _Escape(None, _CONTROL_ESCAPES[c], None, 2)
    if c == "x" and re.fullmatch(r"[0-9A-Fa-f]{2}", src[i + 2 : i + 4]):
        return _Escape(None, int(src[i + 2 : i + 4], 16), None, 4)
    if c == "u" and re.fullmatch(r"[0-9A-Fa-f]{4}", src[i + 2 : i + 6]):
        return _Escape(None, int(src[i + 2 : i + 6], 16), None, 6)
    if c == "c":
        # Annex B: `\cX` is a control escape for an ASCII letter (inside a class also a digit or
        # `_`); any other `\c` is a literal backslash, and the `c` is read next as itself
        nxt = src[i + 2] if i + 2 < len(src) else ""
        if nxt.isascii() and (nxt.isalpha() or (in_class and (nxt.isdigit() or nxt == "_"))):
            return _Escape(None, ord(nxt) % 32, None, 3)
        return _Escape(None, ord("\\"), None, 1)
    if c in "01234567":
        # no capture groups exist (pattern parentheses are escaped), so a digit escape is a legacy
        # octal escape (Annex B): up to three octal digits, value at most 0o377
        text = _OCTAL.match(src, i + 1).group(0)  # type: ignore[union-attr]  # src[i + 1] is octal
        if len(text) == 3 and int(text, 8) > 0o377:
            text = text[:2]
        return _Escape(None, int(text, 8), None, 1 + len(text))
    # identity escape: the character itself (`\8`, `\q`, `\/`, `\.` ...)
    return _Escape(None, ord(c), None, 2)


def _units_class(units: frozenset[int], negate: bool) -> str:
    if not units:
        return "[\\x00-\\uffff]" if negate else "(?!)"
    ranges = []
    ordered = sorted(units)
    start = prev = ordered[0]
    for u in ordered[1:]:
        if u == prev + 1:
            prev = u
            continue
        ranges.append((start, prev))
        start = prev = u
    ranges.append((start, prev))
    body = "".join(f"\\u{a:04x}" if a == b else f"\\u{a:04x}-\\u{b:04x}" for a, b in ranges)
    return f"[{'^' if negate else ''}{body}]"


def _canon_units(units: frozenset[int]) -> frozenset[int]:
    table = _canonicalize_table()
    return frozenset(ord(table[u]) for u in units)


def _class(src: str, i: int) -> tuple[str, int]:
    """A JavaScript character class at `src[i] == "["`; returns (Python source, consumed)."""
    j = i + 1
    negate = j < len(src) and src[j] == "^"
    if negate:
        j += 1
    members: set[int] = set()
    while j < len(src) and src[j] != "]":
        if src[j] == "\\":
            esc = _escape(src, j, True)
            j += esc.length
            if esc.units is not None:
                members |= esc.units
                continue
            assert esc.unit is not None
            unit = esc.unit
        else:
            unit = ord(src[j])
            j += 1
        # range `a-b`
        if j + 1 < len(src) and src[j] == "-" and src[j + 1] != "]":
            if src[j + 1] == "\\":
                end_esc = _escape(src, j + 1, True)
                if end_esc.units is not None:
                    # Annex B: `a-\d` is the three members a, '-', \d
                    members |= {unit, ord("-")} | end_esc.units
                    j += 1 + end_esc.length
                    continue
                assert end_esc.unit is not None
                end_unit, j = end_esc.unit, j + 1 + end_esc.length
            else:
                end_unit, j = ord(src[j + 1]), j + 2
            if unit > end_unit:
                raise InvalidIgnorePattern("Range out of order in character class")
            members |= set(range(unit, end_unit + 1))
            continue
        members.add(unit)
    if j >= len(src):
        raise InvalidIgnorePattern("Unterminated character class")  # e.g. `[ab/x`, as JS throws
    return _units_class(_canon_units(frozenset(members)), negate), j + 1 - i


def to_python(source: str) -> re.Pattern[str]:
    """Compile a JavaScript regex source (flag `i`) for matching **canonicalized code units**."""
    table = _canonicalize_table()
    out: list[str] = []
    i = 0
    while i < len(source):
        c = source[i]
        if c == "\\":
            esc = _escape(source, i, False)
            i += esc.length
            if esc.assertion is not None:
                out.append(esc.assertion)
            elif esc.units is not None:
                out.append(_units_class(_canon_units(esc.units), False))
            else:
                assert esc.unit is not None
                out.append(re.escape(table[esc.unit]))
            continue
        if c == "[":
            text, used = _class(source, i)
            out.append(text)
            i += used
            continue
        if c == "(" and source.startswith("(?", i):
            out.append(source[i : i + 3])  # (?: (?= (?!
            i += 3
            continue
        if c == ".":
            out.append(_DOT)
        elif c == "$":
            out.append(r"\Z")
        elif c in "^|()?*+{}":
            out.append(c)
        else:
            out.append(re.escape(table[ord(c)]))
        i += 1
    return re.compile("".join(out), re.ASCII)


# ---- the Ignore object ------------------------------------------------------------------------


class InvalidIgnorePattern(ValueError):
    """The JavaScript RegExp built from a pattern is invalid (pinned `ignore` throws `SyntaxError`
    when the rule is first evaluated). Raised lazily, exactly when Pi's `rule.regex` would be."""


@dataclass(slots=True)
class _Rule:
    negative: bool
    source: str
    _regex: re.Pattern[str] | None = None

    @property
    def regex(self) -> re.Pattern[str]:
        if self._regex is None:
            self._regex = to_python(self.source)
        return self._regex


def _check_pattern(pattern: str) -> bool:
    return (
        bool(pattern)
        and not _BLANK_LINE.search(pattern)
        and not _INVALID_TRAILING_BACKSLASH.search(pattern)
        and not pattern.startswith("#")
    )


def _create_rule(pattern: str) -> _Rule:
    negative = pattern.startswith("!")
    body = pattern[1:] if negative else pattern
    body = re.sub(r"^\\!", "!", body, count=1)
    body = re.sub(r"^\\#", "#", body, count=1)
    prefix = _regex_prefix(body)
    source = _replace(_TRAILING_WILDCARD, _trailing_wildcard, prefix, False)
    return _Rule(negative, source)


class Ignore:
    """`ignore()` with default options: `add(patterns)` and `ignores(path)`."""

    def __init__(self) -> None:
        self._rules: list[_Rule] = []
        self._cache: dict[str, tuple[bool, bool]] = {}

    def add(self, patterns: Sequence[str]) -> None:
        added = False
        for raw in patterns:
            pattern = to_units(raw)
            if _check_pattern(pattern):
                self._rules.append(_create_rule(pattern))
                added = True
        if added:
            self._cache = {}

    def add_valid(self, patterns: Sequence[str]) -> list[str]:
        """DIV-006: add each pattern whose JavaScript RegExp is valid, in order, and return the
        patterns whose RegExp pinned `ignore` would reject (`SyntaxError`) -- dropped, never added.
        A pattern `ignore` itself skips (blank, comment, invalid trailing backslash) is neither
        added nor rejected, as in Pi."""
        rejected: list[str] = []
        valid: list[_Rule] = []
        for raw in patterns:
            pattern = to_units(raw)
            if not _check_pattern(pattern):
                continue
            rule = _create_rule(pattern)
            try:
                rule.regex  # noqa: B018 -- compile now: an invalid RegExp is the rejection test
            except InvalidIgnorePattern:
                rejected.append(raw)
                continue
            valid.append(rule)
        if valid:
            self._rules.extend(valid)
            self._cache = {}
        return rejected

    def _test(self, path: str) -> tuple[bool, bool]:
        """`RuleManager.test(path, checkUnignored=false, MODE_IGNORE)` -> (ignored, unignored)."""
        ignored = unignored = False
        subject = canonicalize(path)
        for rule in self._rules:
            if (unignored == rule.negative and ignored != unignored) or (
                rule.negative and not ignored and not unignored
            ):
                continue
            if rule.regex.search(subject) is None:
                continue
            ignored = not rule.negative
            unignored = rule.negative
        return ignored, unignored

    def _t(self, path: str) -> tuple[bool, bool]:
        """`ignore`'s `_t`: a path is ignored when its nearest cached-or-tested ancestor is, else by
        its own rules; every result is cached. The package recurses once per parent; this walks the
        same parent chain upward with a list, then evaluates it top-down (`WP141-R004`), so a deep
        path never depends on the interpreter's stack. Evaluation order, the short-circuit on an
        ignored parent, and the cache entries written are the recursion's."""
        slices = [s for s in path.split("/") if s]
        chain: list[str] = []
        current = path
        parent: tuple[bool, bool] | None = None
        while True:
            if current in self._cache:
                parent = self._cache[current]
                break
            chain.append(current)
            slices.pop()
            if not slices:
                break
            current = "/".join(slices) + "/"
        for pending in reversed(chain):
            result = parent if parent is not None and parent[0] else self._test(pending)
            self._cache[pending] = result
            parent = result
        assert parent is not None
        return parent

    def ignores(self, path: str) -> bool:
        """`ignores(path)` for a root-relative, non-empty, not `/`-led path."""
        return self._t(to_units(path))[0]
