"""ECMAScript number semantics the built-in tools' observable output depends on.

Pinned Pi's `read`/`ls`/image code does its arithmetic on IEEE-754 doubles and renders numbers with
`Number.prototype.toString`/`toFixed`; `Math.round` rounds halves up; `Array.prototype.slice`
truncates fractional indices toward zero and counts a negative index from the end. A host-language
shortcut (Python's `round`, `"%.2f"`, `str(float)`, plain slicing) changes model-visible text or
the selected range, so every such operation goes through here. Values arriving as JSON integers are
converted to doubles first, as `JSON.parse` does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal


def to_number(value: float) -> float:
    """A JSON number as an ECMAScript Number (a double)."""
    return float(value)


def number_to_string(value: float) -> str:
    """ECMAScript `Number::toString(x)` (radix 10): shortest round-trip digits, integer form below
    1e21, `"0.000001"` down to 1e-6, exponent form (`"1e+21"`, `"1.5e-7"`) outside that range."""
    x = float(value)
    if math.isnan(x):
        return "NaN"
    if x == 0:
        return "0"
    if x < 0:
        return "-" + number_to_string(-x)
    if math.isinf(x):
        return "Infinity"
    # repr() yields the shortest digit string that round-trips, the same digits ECMAScript picks.
    _sign, digit_tuple, exponent = Decimal(repr(x)).normalize().as_tuple()
    assert isinstance(exponent, int)
    digits = "".join(map(str, digit_tuple))
    k = len(digits)
    n = exponent + k
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    exp = ("+" if e >= 0 else "-") + str(abs(e))
    if k == 1:
        return f"{digits}e{exp}"
    return f"{digits[0]}.{digits[1:]}e{exp}"


def math_round(value: float) -> float:
    """`Math.round`: the nearest integer, halves toward +Infinity. Exact for every double, unlike
    `floor(x + 0.5)` (which rounds 0.49999999999999994 up to 1)."""
    x = float(value)
    if math.isnan(x) or math.isinf(x):
        return x
    floor = math.floor(x)
    return float(floor + 1 if x - floor >= 0.5 else floor)


def to_fixed(value: float, digits: int) -> str:
    """`Number.prototype.toFixed(digits)`: the EXACT binary value rounded half away from zero
    (ECMAScript picks the larger `n` on a tie after taking the sign off); `-0` prints without a
    sign, while a negative that rounds to zero keeps it (`(-0.001).toFixed(2)` is `"-0.00"`)."""
    x = float(value)
    if math.isnan(x):
        return "NaN"
    if abs(x) >= 1e21 or math.isinf(x):
        return number_to_string(x)
    if x == 0:
        x = 0.0
    return str(Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def math_min(a: float, b: float) -> float:
    """`Math.min(a, b)`: NaN if either operand is NaN."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a if a <= b else b


def math_max(a: float, b: float) -> float:
    """`Math.max(a, b)`: NaN if either operand is NaN."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a if a >= b else b


def _relative_index(index: float, length: int) -> int:
    """`ToIntegerOrInfinity` then `Array.prototype.slice`'s clamp: truncate toward zero, count a
    negative index from the end, clamp into `[0, length]`."""
    if math.isnan(index):
        return 0
    if math.isinf(index):
        return length if index > 0 else 0
    whole = math.trunc(index)
    if whole < 0:
        return max(length + whole, 0)
    return min(whole, length)


def js_slice[T](items: Sequence[T], start: float, end: float | None = None) -> list[T]:
    """`Array.prototype.slice(start, end)`."""
    length = len(items)
    lo = _relative_index(start, length)
    hi = length if end is None else _relative_index(end, length)
    return list(items[lo:hi]) if lo < hi else []
