"""`L0506D005-C003`: the `arg_isolation` runner constructs and observes numbers by the certified
authorities -- `JSON.parse`'s binary64 decoding (`raw_arguments_runner.number`) and the prepared
runtime token (`prepared_runtime_runner.render`, ECMAScript Number::toString) -- never Python's
arbitrary-precision integers. Each case is a pinned-Pi neighbor the earlier runner got wrong."""

from __future__ import annotations

import math

from .arg_isolation_runner import build, observe, parse_raw


def test_a_raw_integer_beyond_2p53_is_its_binary64_value() -> None:
    raw = parse_raw('{"n":9007199254740993}')
    assert raw["n"] == 9007199254740992
    assert observe(raw) == {"o": [["n", {"n": "9007199254740992"}]]}


def test_an_inserted_integer_spelled_by_number_tostring_observes_as_spelled() -> None:
    value = build({"n": "1000000000000000100"})
    assert value == 1000000000000000128  # the binary64 value's exact integer
    assert observe(value) == {"n": "1000000000000000100"}


def test_an_inserted_exponent_spelled_number_observes_as_spelled() -> None:
    assert observe(build({"n": "1e+300"})) == {"n": "1e+300"}


def test_named_numbers_round_trip() -> None:
    for token in ("-0", "+Infinity", "-Infinity"):
        assert observe(build({"n": token})) == {"n": token}
    assert math.isnan(build({"n": "NaN"}))
    assert observe(build({"n": "NaN"})) == {"n": "NaN"}
    assert observe(parse_raw("-0")) == {"n": "-0"}


def test_an_integer_that_is_not_binary64_is_never_rounded_into_a_matching_token() -> None:
    assert observe(9007199254740993) == {"non_binary64_int": hex(9007199254740993)}


# CE-L0506-D005-01 (C003, review 2): the binary64 range boundary of an integer-shaped literal.
# The smallest integer that rounds (ties-to-even) to Infinity.
OVERFLOW_THRESHOLD = 2**1024 - 2**970


def test_the_largest_finite_rounding_integer_is_the_largest_binary64_value() -> None:
    raw = parse_raw(f'{{"n":{OVERFLOW_THRESHOLD - 1}}}')
    assert raw["n"] == int(float.fromhex("0x1.fffffffffffffp+1023"))
    assert observe(raw) == {"o": [["n", {"n": "1.7976931348623157e+308"}]]}


def test_an_integer_at_the_overflow_threshold_is_infinity() -> None:
    assert observe(parse_raw(f'{{"n":{OVERFLOW_THRESHOLD}}}')) == {"o": [["n", {"n": "+Infinity"}]]}


def test_a_401_digit_integer_is_signed_infinity() -> None:
    digits = "1" + "0" * 400
    assert observe(parse_raw(f'{{"n":{digits}}}')) == {"o": [["n", {"n": "+Infinity"}]]}
    assert observe(parse_raw(f'{{"n":-{digits}}}')) == {"o": [["n", {"n": "-Infinity"}]]}


def test_an_exponent_spelled_overflow_agrees() -> None:
    assert observe(parse_raw('{"n":1e400}')) == {"o": [["n", {"n": "+Infinity"}]]}
