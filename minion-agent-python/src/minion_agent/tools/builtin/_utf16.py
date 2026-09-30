"""JavaScript string semantics for WP-13.2 (spec/tools.md WP-13.2 "String semantics").

Pinned Pi's `write`/`edit` work on JavaScript strings: sequences of UTF-16 code units, where every
length, index, `indexOf`, `slice` and `split("")` counts code units. This module represents such a
string as a Python `str` whose characters ARE the code units (an astral character becomes its two
surrogate characters), so ordinary `str` operations on it are exactly the JavaScript ones. Text
enters that form once (`to_units`) and leaves it once (`from_units`); everything between runs on
code units.
"""

from __future__ import annotations

import re

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def to_units(text: str) -> str:
    """A Python string as its UTF-16 code units, one character per unit."""
    if text.isascii():
        return text
    data = text.encode("utf-16-le", "surrogatepass")
    return "".join(chr(data[i] | (data[i + 1] << 8)) for i in range(0, len(data), 2))


def from_units(units: str) -> str:
    """The inverse of `to_units`: valid surrogate pairs recombine; a lone surrogate stays itself."""
    if units.isascii():
        return units
    return units.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def utf16_length(text: str) -> int:
    """`String.prototype.length` of a Python string."""
    return len(text) + sum(1 for ch in text if ord(ch) > 0xFFFF)


def decode_utf8(data: bytes) -> str:
    """Node's `Buffer#toString("utf-8")`: WHATWG UTF-8 decode, one U+FFFD per maximal invalid
    subpart, the BOM kept -- the decoding `read` certifies (`IMPL-C005`)."""
    return data.decode("utf-8", "replace")


def encode_utf8(text: str) -> bytes:
    """Node's `fs.writeFile(path, string, "utf-8")`: WHATWG UTF-8 encode, an unpaired surrogate
    written as U+FFFD (`EF BF BD`). Takes a Python string (pairs already combined)."""
    return _LONE_SURROGATE.sub(chr(0xFFFD), text).encode("utf-8")
