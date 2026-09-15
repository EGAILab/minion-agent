"""`js_json_stringify` (`PROV-012`'s own `L11-SC-R010` rendering contract).

Every value below was independently cross-checked against a live Node 22 `JSON.stringify` call
before being committed here as a fixture (see the session's own verification discipline,
`spec/auth.md`'s own "Exact error-message JSON rendering" section) -- these are not assumed
results.
"""

from __future__ import annotations

import json
import math

import pytest

from minion_agent.auth.credential import JsonValue
from minion_agent.auth.js_json import js_json_loads, js_json_stringify


def test_null_true_false() -> None:
    assert js_json_stringify(None) == "null"
    assert js_json_stringify(True) == "true"
    assert js_json_stringify(False) == "false"


def test_negative_zero_collapses_to_bare_digit() -> None:
    assert js_json_stringify(-0.0) == "0"
    assert js_json_stringify(0.0) == "0"


def test_number_fixed_notation() -> None:
    assert js_json_stringify(5.0) == "5"
    assert js_json_stringify(100.0) == "100"
    assert js_json_stringify(123.456) == "123.456"
    assert js_json_stringify(0.1) == "0.1"


def test_number_small_exponent_notation_boundary() -> None:
    """Fixed at 1e-6, exponential at 1e-7 -- confirmed live against Node."""
    assert js_json_stringify(1e-6) == "0.000001"
    assert js_json_stringify(1e-7) == "1e-7"


def test_number_large_exponent_notation_boundary() -> None:
    """Fixed at 1e20, exponential at 1e21 -- confirmed live against Node. Python's own
    default `repr`/`json.dumps` formatting diverges from this threshold (already exponential
    by 1e19), so a passing witness here specifically rules out reusing Python's own float
    formatting."""
    assert js_json_stringify(1e20) == "100000000000000000000"
    assert js_json_stringify(1e21) == "1e+21"


def test_number_exponential_notation_with_multi_digit_mantissa() -> None:
    """A multi-significant-digit value in exponential notation inserts a decimal point after
    the first digit (`d.ddd...e+NN`), distinct from the single-digit case above."""
    assert js_json_stringify(1.5e300) == "1.5e+300"


def test_array() -> None:
    assert js_json_stringify([1, "a", True, None, -0.0]) == '[1,"a",true,null,0]'


def test_empty_object_and_array() -> None:
    assert js_json_stringify({}) == "{}"
    assert js_json_stringify([]) == "[]"


def test_non_ascii_string_passes_through_unescaped() -> None:
    """A properly-paired astral character (an emoji) and an ordinary non-ASCII BMP character
    both pass through literally -- confirmed live, the OPPOSITE of Python's own `json.dumps`
    `ensure_ascii=True` default."""
    assert js_json_stringify("café") == '"café"'
    assert js_json_stringify("😀") == '"😀"'


def test_lone_surrogate_is_escaped_not_passed_through() -> None:
    """A lone (unpaired) UTF-16 surrogate code unit is ESCAPED via the generic 4-hex-digit form,
    not passed through literally -- confirmed live (`L11-SC-R010`, sixth review): a naive
    "non-ASCII passes through unescaped" implementation gets exactly this case wrong. Python's
    own `json.loads` combines a valid consecutive surrogate pair into one astral codepoint, so a
    Python string character whose own codepoint remains in the surrogate range is necessarily an
    unpaired one."""
    lone_high_surrogate = json.loads(r'"\ud800"')
    assert js_json_stringify(lone_high_surrogate) == '"\\ud800"'
    lone_low_surrogate = json.loads(r'"\udc00"')
    assert js_json_stringify(lone_low_surrogate) == '"\\udc00"'


def test_paired_surrogate_forms_one_literal_astral_character() -> None:
    paired = json.loads(r'"𐀀"')
    assert js_json_stringify(paired) == '"𐀀"'


def test_control_characters_use_named_escapes_where_defined() -> None:
    assert js_json_stringify('a\tb\nc"d\\e') == '"a\\tb\\nc\\"d\\\\e"'


def test_control_character_with_no_named_escape_uses_generic_four_digit_form() -> None:
    """`U+0001` has no named JSON escape; it renders as the six-character sequence backslash,
    lowercase `u`, then four hex digits -- NOT a two-digit form."""
    assert js_json_stringify("\x01") == '"\\u0001"'


def test_object_preserves_insertion_order_when_no_index_like_keys() -> None:
    assert js_json_stringify({"b": 2, "a": 1}) == '{"b":2,"a":1}'


def test_object_reorders_array_index_like_keys_numerically_first() -> None:
    """ECMAScript's own own-property enumeration reorders array-index-like keys ahead of every
    other key, in ascending numeric order, regardless of source insertion order -- confirmed live
    this DIFFERS from plain source-order preservation (which Python's own `dict`/`json.loads`
    pipeline always uses)."""
    value: JsonValue = {"2": "b", "1": "a", "x": 0}
    assert js_json_stringify(value) == '{"1":"a","2":"b","x":0}'


def test_object_index_key_boundary_is_exactly_2_32_minus_2() -> None:
    """`"4294967294"` (2**32 - 2) IS an array-index key and sorts first; `"4294967295"`
    (2**32 - 1) is NOT and keeps its own source position -- confirmed live."""
    under_boundary: JsonValue = {"0": "zero", "4294967294": "max", "x": 1}
    assert js_json_stringify(under_boundary) == '{"0":"zero","4294967294":"max","x":1}'

    over_boundary: JsonValue = {"4294967295": "too_big_not_index", "x": 1}
    assert js_json_stringify(over_boundary) == '{"4294967295":"too_big_not_index","x":1}'


def test_object_leading_zero_key_is_not_an_array_index() -> None:
    """`"01"` is digit-only but has a leading zero, so it is NOT an array-index key and keeps
    its own relative source position among the other non-index keys -- confirmed live."""
    value: JsonValue = {"01": "a", "2": "b", "x": 0}
    assert js_json_stringify(value) == '{"2":"b","01":"a","x":0}'


def test_unicode_digit_key_is_not_treated_as_an_array_index() -> None:
    """`L11-SC-R014` -- confirmed live against Node: ECMA-262's own array-index grammar is
    ASCII-decimal-only, so a Unicode decimal digit key (Arabic-Indic digit one, `U+0661`) is
    NOT reordered ahead of other keys, unlike Python's own Unicode-aware `str.isdigit()`, which
    would incorrectly treat it as one."""
    arabic_indic_one = chr(0x0661)
    value: JsonValue = {arabic_indic_one: "arabic", "2": "two", "x": 0}
    assert js_json_stringify(value) == f'{{"2":"two","{arabic_indic_one}":"arabic","x":0}}'


def test_very_long_digit_only_key_does_not_raise() -> None:
    """`L11-SC-R014` -- a key longer than the maximum array index's own digit count can never
    satisfy the array-index bound, so it must be recognized as non-index WITHOUT ever calling
    `int()` on it -- Python 3.11+'s integer-string-conversion length limit raises `ValueError`
    for a naive `int()` call on a key this long (5000 ASCII digits)."""
    huge_key = "9" * 5000
    value: JsonValue = {huge_key: "huge", "x": 0}
    assert js_json_stringify(value) == f'{{"{huge_key}":"huge","x":0}}'


def test_nested_object_and_array() -> None:
    value: JsonValue = {"nested": {"x": [1, 2], "y": None}}
    assert js_json_stringify(value) == '{"nested":{"x":[1,2],"y":null}}'


# --- js_json_loads (`L11-SC-R013`) -----------------------------------------------------------


def test_js_json_loads_parses_ordinary_values() -> None:
    assert js_json_loads('{"a":1,"b":[true,false,null,"x"]}') == {
        "a": 1.0,
        "b": [True, False, None, "x"],
    }


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_js_json_loads_rejects_bare_invalid_constants(token: str) -> None:
    """Confirmed live against Node: `JSON.parse` raises a `SyntaxError` for each of these bare
    tokens; Python's own `json.loads` accepts them by default as a non-standard extension."""
    with pytest.raises(ValueError, match="not valid JSON"):
        js_json_loads(token)


def test_js_json_loads_rejects_embedded_invalid_constant() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        js_json_loads('{"expires_in":NaN}')


def test_js_json_loads_propagates_genuine_syntax_errors_unchanged() -> None:
    with pytest.raises(ValueError):
        js_json_loads("not json at all")


def test_js_json_loads_coerces_integers_through_ieee754_double() -> None:
    """`parse_int=float` matches JS's own single numeric type -- every JSON number, including a
    bare integer literal, comes back as a Python `float`, and `-0` preserves its own sign."""
    result = js_json_loads('{"n":5,"neg_zero":-0}')
    assert isinstance(result, dict)
    assert result["n"] == 5.0
    assert isinstance(result["n"], float)
    neg_zero = result["neg_zero"]
    assert isinstance(neg_zero, float)
    assert math.copysign(1.0, neg_zero) == -1.0
