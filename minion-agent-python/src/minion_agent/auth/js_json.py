"""ECMAScript `JSON.stringify`-faithful rendering of a parsed JSON value (`PROV-012`'s own
`L11-SC-R010` contract, `spec/auth.md`'s own "Exact error-message JSON rendering" section).

`JSON.stringify` itself is the complete normative authority; this module is a dedicated renderer
built to the exact, independently-verified rules that section states -- string escaping (including
the unpaired-surrogate exception), negative-zero collapse, `Number::toString`'s own fixed-vs-
exponential notation threshold, and array-index-like property reordering. It is NOT a `json.dumps`
call with adjusted options; no combination of `json.dumps` parameters reproduces this algorithm.
"""

from __future__ import annotations

from decimal import Decimal

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
    integer in `[0, 2**32 - 2]`, no leading zeros (so `"0"` qualifies, `"01"` does not)."""
    if not key.isdigit():
        return False
    if key != "0" and key[0] == "0":
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
    `,`, matching pinned Pi's own real call sites, none of which pass an indentation argument."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _js_json_string(value)
    if isinstance(value, int | float):
        return _js_number_to_string(float(value))
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
