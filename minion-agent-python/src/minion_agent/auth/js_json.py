"""ECMAScript `JSON.stringify`/`JSON.parse`-faithful conversion between a parsed JSON value and
its own text form (`PROV-012`'s own `L11-SC-R010`/`L11-SC-R013` contract, `spec/auth.md`'s own
"Exact error-message JSON rendering" section), plus `js_trim` (ECMA-262 `String.prototype.trim`
fidelity, shared by every call site that trims a string before an ECMAScript numeric/URL
coercion).

`JSON.stringify`/`JSON.parse` themselves are the complete normative authority for both
directions below, but the two are NOT symmetric in implementation mechanics (`L11-SC-R021` --
this distinction was previously stated too broadly, contradicting the parser's own actual
code): `js_json_stringify` genuinely is a dedicated renderer built to the exact,
independently-verified rendering rules those sections state -- string escaping (including the
unpaired-surrogate exception), negative-zero collapse, `Number::toString`'s own
fixed-vs-exponential notation threshold, and array-index-like property reordering. No
combination of `json.dumps`'s own parameters reproduces that algorithm. `js_json_loads`, by
contrast, deliberately REUSES Python's own standard `json.loads` parser -- ordinary JSON syntax
parsing is not something this project reimplements -- with exactly two semantic hooks applied on
top: rejecting the bare `NaN`/`Infinity`/`-Infinity` extension tokens Python's own `json.loads`
accepts by default but JS's own `JSON.parse` does not, and coercing every number through
IEEE-754 double representation via `parse_int=float`.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import cast

from .credential import JsonValue

_NAMED_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}
_MAX_ARRAY_INDEX = 2**32 - 2
_MAX_ARRAY_INDEX_DIGITS = len(str(_MAX_ARRAY_INDEX))

_JS_WHITESPACE = frozenset(
    {
        0x0009,  # <TAB>
        0x000B,  # <VT>
        0x000C,  # <FF>
        0x0020,  # <SP>
        0x00A0,  # <NBSP>
        0xFEFF,  # <ZWNBSP> (byte-order mark)
        0x000A,  # <LF>
        0x000D,  # <CR>
        0x2028,  # <LS>
        0x2029,  # <PS>
        # <USP>: every other Unicode code point with General_Category Space_Separator (Zs).
        0x1680,
        0x2000,
        0x2001,
        0x2002,
        0x2003,
        0x2004,
        0x2005,
        0x2006,
        0x2007,
        0x2008,
        0x2009,
        0x200A,
        0x202F,
        0x205F,
        0x3000,
    }
)


def js_trim(value: str) -> str:
    """ECMA-262 `String.prototype.trim`: strip leading/trailing `WhiteSpace`/`LineTerminator`
    characters (the exact fixed code-point set above), NOT Python's own `str.strip()` -- confirmed
    live against Node (`L11-SC-R012`, second independent review): `str.strip()` does not remove
    `U+FEFF` (the byte-order mark), which JS's own `trim()` does, so a BOM-prefixed numeric string
    that should coerce to a finite number instead coerces to `NaN` under a naive `str.strip()`.
    Shared by every call site that trims a string before an ECMAScript numeric/URL coercion
    (`_js_number_coerce`, `parse_authorization_input`) rather than duplicated per call site."""
    start = 0
    end = len(value)
    while start < end and ord(value[start]) in _JS_WHITESPACE:
        start += 1
    while end > start and ord(value[end - 1]) in _JS_WHITESPACE:
        end -= 1
    return value[start:end]


def to_usv_string(value: str) -> str:
    """Web IDL `USVString` conversion, the FULL algorithm: scan `value` one code point at a
    time; a HIGH surrogate (`U+D800`-`U+DBFF`) immediately followed by a LOW surrogate
    (`U+DC00`-`U+DFFF`) is a valid PAIR and is COMBINED into the single astral scalar value it
    represents (the standard UTF-16 surrogate-pair formula); any OTHER surrogate-range code
    point -- a high surrogate not immediately followed by a low one, a low surrogate not
    immediately preceded by a high one, or either at a string boundary with no partner -- is
    UNPAIRED and is replaced with `U+FFFD` (the replacement character) individually. Confirmed
    live this exactly matches `new URL(value)` (a `USVString`-typed Web IDL operand): a valid
    adjacent high/low pair renders as the combined astral character (e.g. an emoji), never as two
    separate replacement characters.

    `L11-SC-R011`, targeted convergence re-review, second round: an EARLIER version of this
    function treated EVERY surrogate-range Python code point as necessarily unpaired, reasoning
    (correctly, but INCOMPLETELY) from `_js_json_string`'s own escaping logic, which relies on
    `json.loads` already combining a valid pair before its own input is ever seen -- that
    combining step is specific to `json.loads`'s own decoding path and does NOT hold for this
    function's own general string input (e.g. `parse_authorization_input`'s own manually-pasted
    input, which never passes through `json.loads` at all): a Python string CAN validly contain
    an explicit adjacent high+low surrogate PAIR as two separate code points (confirmed live,
    `chr(0xD83D) + chr(0xDE00)` is exactly such a pair, representing the SAME astral character
    the single Python code point `chr(0x1F600)` would -- and that pair MUST be combined here,
    not independently replaced.

    This function performs only the conversion itself; the caller decides which string (this
    function's own result, or the ORIGINAL unconverted value) to use for which branch -- see
    `parse_authorization_input`'s own docstring for the full URL-construction-vs-fallback rule
    this conversion feeds into."""
    result: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF and index + 1 < length:
            next_codepoint = ord(value[index + 1])
            if 0xDC00 <= next_codepoint <= 0xDFFF:
                combined = 0x10000 + (codepoint - 0xD800) * 0x400 + (next_codepoint - 0xDC00)
                result.append(chr(combined))
                index += 2
                continue
        result.append(chr(0xFFFD) if 0xD800 <= codepoint <= 0xDFFF else value[index])
        index += 1
    return "".join(result)


def _escape_char(codepoint: int) -> str:
    if codepoint in _NAMED_ESCAPES:
        return _NAMED_ESCAPES[codepoint]
    return f"\\u{codepoint:04x}"


def _js_json_string(value: str) -> str:
    """String escaping: `"`/`\\`/control characters (`U+0000`-`U+001F`) via named escapes where
    JSON defines one, else the generic 4-hex-digit `\\uXXXX` form. An UNPAIRED UTF-16 surrogate
    code unit (`U+D800`-`U+DFFF`) is ALSO escaped the same generic way -- confirmed live that
    `JSON.stringify` does this (ECMAScript's own "well-formed JSON.stringify", ES2019) -- which is
    reliably detectable here because Python's own `json.loads` already combines a valid consecutive
    surrogate pair into a single astral codepoint (`U+10000`+), so any Python string character whose
    OWN codepoint still falls in the surrogate range is necessarily an unpaired one; a properly
    combined astral character's own codepoint is never in that range. Every other character --
    every printable ASCII character and every other non-ASCII Unicode character -- passes through
    completely unescaped, literally."""
    out: list[str] = []
    for ch in value:
        codepoint = ord(ch)
        needs_escape = (
            codepoint < 0x20
            or codepoint == 0x22
            or codepoint == 0x5C
            or 0xD800 <= codepoint <= 0xDFFF
        )
        out.append(_escape_char(codepoint) if needs_escape else ch)
    return '"' + "".join(out) + '"'


def _shortest_digits_and_exponent(magnitude: float) -> tuple[str, int]:
    """The shortest round-trip decimal digit string `s` (no leading/trailing zeros) and the
    decimal exponent `n` such that `magnitude == int(s) * 10 ** (n - len(s))` -- ECMA-262's own
    Number::toString variables. Python's own `repr(float)` has produced the correctly-rounded
    shortest round-trip decimal representation since Python 3.1, the SAME mathematical property
    ECMAScript's own algorithm targets, so the DIGITS agree between the two languages; only the
    notation chosen to DISPLAY them differs (`_format_fixed_or_exponential` below). `Decimal`
    parses whichever notation `repr` chose (plain or scientific) into an exact `(digits, exponent)`
    pair; trailing zeros in that raw tuple are decorative (e.g. `repr(100.0) == "100.0"` parses to
    digits `(1, 0, 0, 0)`, exponent `-1`) and are stripped here, with the exponent adjusted to
    compensate, to recover the true minimal significant-digit count."""
    sign, digits, exponent = Decimal(repr(magnitude)).as_tuple()
    assert sign == 0
    assert isinstance(exponent, int)
    digit_list = list(digits)
    while len(digit_list) > 1 and digit_list[-1] == 0:
        digit_list.pop()
        exponent += 1
    s = "".join(str(d) for d in digit_list)
    n = exponent + len(s)
    return s, n


def _format_fixed_or_exponential(s: str, n: int) -> str:
    """ECMA-262 Number::toString's own notation-choice algorithm, applied to the shortest digit
    string `s` (length `k`) and exponent `n` -- empirically confirmed live against Node for every
    boundary case named in `spec/auth.md`'s own contract (fixed for `1e-6`, exponential for `1e-7`;
    fixed for `1e20`, exponential for `1e21`)."""
    k = len(s)
    if k <= n <= 21:
        return s + "0" * (n - k)
    if 0 < n <= 21:
        return s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + s
    exponent_sign = "+" if n - 1 >= 0 else "-"
    exponent_digits = str(abs(n - 1))
    if k == 1:
        return f"{s}e{exponent_sign}{exponent_digits}"
    return f"{s[0]}.{s[1:]}e{exponent_sign}{exponent_digits}"


def _js_number_to_string(value: float) -> str:
    """Number formatting: negative zero collapses to the bare digit `0` (confirmed live -- no
    minus sign, unlike Python's own sign-preserving float rendering); every other value is rendered
    via ECMA-262's own `Number::toString` notation algorithm, not Python's own `repr`/`json.dumps`
    formatting, which chooses different notation thresholds (confirmed independently: Python's own
    compact `json.dumps` already diverges from JS at `1e19`, exponential in Python but still fixed
    in JS at that magnitude)."""
    if value == 0.0:
        return "0"
    sign = "-" if value < 0 else ""
    s, n = _shortest_digits_and_exponent(abs(value))
    return sign + _format_fixed_or_exponential(s, n)


def _is_array_index(key: str) -> bool:
    """ECMA-262's own "array index" property-key predicate: the canonical decimal string of an
    integer in `[0, 2**32 - 2]`, no leading zeros (so `"0"` qualifies, `"01"` does not).

    Digit membership is checked one ASCII code point at a time (`"0" <= ch <= "9"`), NOT via
    Python's own Unicode-aware `str.isdigit()` -- confirmed live against Node (`L11-SC-R014`):
    ECMA-262's own array-index grammar is ASCII-decimal-only, so a Unicode decimal digit (e.g.
    Arabic-Indic digit one) is never part of an array-index key, unlike `str.isdigit()`, which
    accepts it. The key's own length is also bounded before calling `int()` -- an
    all-ASCII-digit key longer than the maximum array index's own digit count can never satisfy
    the `<= _MAX_ARRAY_INDEX` bound anyway, and this avoids Python 3.11+'s integer-string-
    conversion length limit (`ValueError`) for a very long digit-only key."""
    if not key or not all("0" <= ch <= "9" for ch in key):
        return False
    if key != "0" and key[0] == "0":
        return False
    if len(key) > _MAX_ARRAY_INDEX_DIGITS:
        return False
    return int(key) <= _MAX_ARRAY_INDEX


def _js_property_order(obj: dict[str, JsonValue]) -> list[str]:
    """ECMA-262's own own-property enumeration order: array-index keys FIRST, in ascending numeric
    order, followed by every other key in the source's own original insertion order -- confirmed
    live this reorders even when it contradicts the source JSON's own key order, DISPROVING a
    plain "preserve source order" assumption for this case specifically."""
    array_index_keys = sorted((k for k in obj if _is_array_index(k)), key=int)
    other_keys = [k for k in obj if not _is_array_index(k)]
    return array_index_keys + other_keys


def js_json_stringify(value: JsonValue) -> str:
    """The complete renderer: `null`/`true`/`false` literals, strings (`_js_json_string`), numbers
    (`_js_number_to_string`), arrays (each element rendered recursively, comma-joined), and objects
    (each property rendered as `"key":value`, comma-joined, in `_js_property_order`'s own order).
    Matches ECMAScript `JSON.stringify`'s own compact form exactly -- no whitespace after `:` or
    `,`, matching pinned Pi's own real call sites, none of which pass an indentation argument.

    A NON-FINITE number (`inf`/`-inf`/`nan`) renders as the JSON literal `null`, at ANY position
    -- top-level or nested (`L11-SC-R024`, mandatory final-complete review): ECMA-262's own
    `SerializeJSONProperty` step checks `Number::isFinite` BEFORE calling `Number::toString` at
    all, returning the literal `"null"` immediately for a non-finite value -- confirmed live
    against Node this applies UNIFORMLY regardless of nesting depth (`JSON.stringify(Infinity)`
    at the very top level ALSO returns the string `"null"`, not `undefined` -- that is a
    DIFFERENT, unrelated case: `JSON.stringify(undefined)` is what returns `undefined`, and this
    function's own `JsonValue` input domain has no representation for `undefined` in the first
    place, so no separate top-level special case is needed here). This is reachable through a
    real call site: `js_json_loads` already correctly parses a JSON numeric literal like `1e400`
    into `inf` (JSON number syntax permits an exponent this large; ECMAScript's own numeric
    overflow behavior applies, distinct from the bare invalid token `Infinity`, which
    `L11-SC-R013` correctly rejects), and the resulting `inf` can reach this renderer when
    building an "invalid response" error message embedding the parsed body verbatim."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _js_json_string(value)
    if isinstance(value, int | float):
        magnitude = float(value)
        if not math.isfinite(magnitude):
            return "null"
        return _js_number_to_string(magnitude)
    if isinstance(value, list):
        return "[" + ",".join(js_json_stringify(item) for item in value) + "]"
    return (
        "{"
        + ",".join(
            f"{_js_json_string(key)}:{js_json_stringify(value[key])}"
            for key in _js_property_order(value)
        )
        + "}"
    )


def _reject_js_incompatible_constant(token: str) -> float:
    """Python's own `json.loads` accepts the bare tokens `NaN`/`Infinity`/`-Infinity` as an
    extension beyond the JSON grammar (`parse_constant`'s own default); JavaScript's `JSON.parse`
    does NOT -- confirmed live (matching this project's own already-established `PROV-011`
    precedent, `openai_codex.py::decode_jwt`, which fixed the identical gap for JWT-payload
    decoding specifically; this is the same fix generalized for ordinary HTTP response-body
    parsing, `L11-SC-R013`). Wiring this as `json.loads`'s own `parse_constant` hook makes Python
    raise for exactly the same three tokens Pi's `JSON.parse` rejects."""
    raise ValueError(f"not valid JSON: unexpected token {token!r}")


def js_json_loads(text: str) -> JsonValue:
    """Parse `text` the way ECMAScript `JSON.parse` would -- rejecting the bare
    `NaN`/`Infinity`/`-Infinity` extension tokens Python's own `json.loads` accepts by default, and
    coercing every JSON number literal (including integers) through IEEE-754 double representation
    via `parse_int=float`, matching JS's own single numeric type exactly (an integer literal beyond
    `2**53` silently loses precision the same way JS's own `JSON.parse` does, and `-0` preserves its
    own sign) -- the identical `PROV-011` `decode_jwt` fidelity model
    (`L11-SA-R001`/`L11-SC-R013`), used here for ordinary HTTP response-body JSON, not only JWT
    payloads. Raises `ValueError`/`json.JSONDecodeError` for genuinely malformed JSON, propagated
    UNCHANGED to the caller, matching `JSON.parse`'s own `SyntaxError` -- this function never
    silently repairs or swallows a parse failure itself."""
    return cast(
        JsonValue,
        json.loads(text, parse_int=float, parse_constant=_reject_js_incompatible_constant),
    )
