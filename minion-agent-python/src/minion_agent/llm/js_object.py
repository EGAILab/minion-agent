"""ECMAScript object key order for tool-argument objects (`L0206-D001`, K1; spec/llm.md
"Tool-argument object key order").

Pinned Pi's argument objects are JavaScript objects: their own keys enumerate in ECMAScript
`OrdinaryOwnPropertyKeys` order -- array-index keys (the canonical decimal string of an integer
0 .. 4294967294) first, ascending, then every other key in insertion order. A Python `dict`
enumerates in pure insertion order, so a binding must keep that order itself.

`JsObject` is a `dict` whose own insertion order IS that order at all times: an array-index key
assigned for the first time is moved to its ascending position among the indices; any other new key
goes last; an existing key keeps its position. Iteration, `keys`/`items`/`values`, `repr` and
JSON encoding therefore enumerate by the rule without overriding them. `js_object` applies it
recursively, through arrays and nested objects, to a value built from plain `dict`s.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_MAX_ARRAY_INDEX = 4294967294


def is_array_index(key: str) -> bool:
    """ECMAScript's array index: the canonical decimal string of an integer 0 .. 2**32 - 2. `"0"`
    is one; `"00"`, `"01"`, `"-0"`, `"+1"`, `"1.0"`, `" 1"` and `"4294967295"` are not."""
    if not key or not key.isascii() or not key.isdigit():
        return False
    if len(key) > 1 and key[0] == "0":
        return False
    return int(key) <= _MAX_ARRAY_INDEX


class JsObject(dict[str, Any]):
    """A tool-argument object that enumerates its keys in ECMAScript order (module docstring).
    Values assigned into it are converted by `js_object`, so nested objects keep the rule too."""

    __slots__ = ()

    def __init__(self, items: Mapping[str, Any] | Iterable[tuple[str, Any]] = (), /) -> None:
        super().__init__()
        pairs = items.items() if isinstance(items, Mapping) else items
        for key, value in pairs:
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        value = js_object(value)
        if key in self or not is_array_index(key):
            super().__setitem__(key, value)
            return
        # A new index key: everything after its ascending position moves behind it.
        position = int(key)
        later = [k for k in self if not is_array_index(k) or int(k) > position]
        moved = [(k, super(JsObject, self).pop(k)) for k in later]
        super().__setitem__(key, value)
        for k, v in moved:
            super().__setitem__(k, v)

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        pairs = other.items() if isinstance(other, Mapping) else other
        for key, value in pairs:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None, /) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other: Any) -> JsObject:  # type: ignore[override]
        self.update(other)
        return self

    def __or__(self, other: Any) -> JsObject:  # type: ignore[override]
        merged = JsObject(self)
        merged.update(other)
        return merged

    def copy(self) -> JsObject:
        return JsObject(self)


def js_object(value: Any) -> Any:
    """`value` with every object in it (at any depth, through arrays) a `JsObject`: a plain `dict`
    becomes one, enumerating in ECMAScript order as if its keys had been assigned in its own
    iteration order; a list is converted in place (its identity kept); anything else is returned
    unchanged. An existing `JsObject` is returned as is."""
    if isinstance(value, JsObject):
        return value
    if isinstance(value, dict):
        return JsObject(value)
    if isinstance(value, list):
        for index, item in enumerate(value):
            converted = js_object(item)
            if converted is not item:
                value[index] = converted
        return value
    return value
