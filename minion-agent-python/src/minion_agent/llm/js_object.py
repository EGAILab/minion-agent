"""ECMAScript object key order for tool-argument objects (`L0206-D001`, K1; spec/llm.md
"Tool-argument object key order").

Pinned Pi's argument objects are JavaScript objects: their own keys enumerate in ECMAScript
`OrdinaryOwnPropertyKeys` order -- array-index keys (the canonical decimal string of an integer
0 .. 4294967294) first, ascending, then every other key in insertion order -- and they are SHARED
by reference: a hook that keeps a child object and mutates it later, or appends an object to an
array, changes what `execute` receives. A Python `dict` enumerates in pure insertion order, so the
binding keeps the order itself WITHOUT copying:

- `order_in_place(value)` re-sequences every `dict` reachable from `value` (through lists and nested
  objects) into the rule's order, IN PLACE: each object keeps its identity, so every reference a
  caller or hook holds still sees -- and still mutates -- the delivered object. The pipeline calls
  it at every boundary (construction, a `prepare_arguments` result, validation, between
  pre-execute listeners, before `execute`).
- `JsObject` is the validated top-level arguments object: a `dict` that keeps the rule on its OWN
  assignments immediately (a new array-index key moves to its ascending position). Values assigned
  into it are stored as given -- never copied -- and ordered at the next boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_MAX_ARRAY_INDEX = 4294967294
_MAX_ARRAY_INDEX_DIGITS = len(str(_MAX_ARRAY_INDEX))


def is_array_index(key: str) -> bool:
    """ECMAScript's array index: the canonical decimal string of an integer 0 .. 2**32 - 2. `"0"`
    is one; `"00"`, `"01"`, `"-0"`, `"+1"`, `"1.0"`, `" 1"` and `"4294967295"` are not. Total over
    every string: the length is decided before any integer conversion (`L0206-D001-R003`), so an
    arbitrarily long decimal key is an ordinary key, not an error."""
    if not key or len(key) > _MAX_ARRAY_INDEX_DIGITS or not key.isascii() or not key.isdigit():
        return False
    if len(key) > 1 and key[0] == "0":
        return False
    return int(key) <= _MAX_ARRAY_INDEX


def es_order(keys: Iterable[str]) -> list[str]:
    """`keys` (in insertion order) in ECMAScript `OrdinaryOwnPropertyKeys` order."""
    ordered = list(keys)
    indices = sorted((k for k in ordered if is_array_index(k)), key=int)
    return [*indices, *(k for k in ordered if not is_array_index(k))]


class JsObject(dict[str, Any]):
    """The validated top-level arguments object: keeps the rule on its own assignments (module
    docstring). Values are stored as given; nested objects are ordered by `order_in_place`."""

    __slots__ = ()

    def __init__(self, items: Mapping[str, Any] | Iterable[tuple[str, Any]] = (), /) -> None:
        super().__init__()
        pairs = items.items() if isinstance(items, Mapping) else items
        for key, value in pairs:
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
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


def order_in_place(value: Any) -> Any:
    """Re-sequence every `dict` reachable from `value` into ECMAScript order, in place, and return
    `value` itself. Identity is preserved everywhere (no object or list is replaced), shared and
    cyclic references are visited once, and a `dict` already in order is left untouched."""
    seen: set[int] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if not isinstance(item, dict | list) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, list):
            pending.extend(item)
            continue
        keys = list(item)
        wanted = es_order(keys)
        if wanted != keys:
            values = {k: dict.__getitem__(item, k) for k in keys}
            dict.clear(item)
            for k in wanted:
                dict.__setitem__(item, k, values[k])
        pending.extend(dict.values(item))
    return value


def order_raw(arguments: Any) -> Any:
    """`L0206-D001-R004`: order a call's RAW arguments object where it is observed or serialized
    (session encoding, the session tool-call record, the execution-start and update payloads). The
    raw object is shared and mutable after construction, so construction-time ordering alone is
    not enough. A separate name from `order_in_place` so each boundary family is controllable."""
    return order_in_place(arguments)


def adopt(value: Any) -> Any:
    """`CE-L0206-D001-01`: make every object in a value the pipeline is about to OWN (a call's raw
    arguments, as a provider decoded them) a `JsObject`, recursively, so that any later mutation of
    an object the pipeline owns keeps the rule immediately -- for every observer, including the
    mutating observer itself, as a JavaScript object does. Lists are converted in place (identity
    kept); an existing `JsObject` is kept as is, so adoption never replaces an object a hook or
    caller already shares with the pipeline. Plain objects a listener assigns LATER are not adopted
    (no copy, `R002`): they are ordered in place at every observer invocation."""
    if isinstance(value, JsObject):
        for item in dict.values(value):
            adopt(item)
        return value
    if isinstance(value, dict):
        adopted = JsObject()
        for key, item in value.items():
            dict.__setitem__(adopted, key, adopt(item))
        return order_in_place(adopted)
    if isinstance(value, list):
        for index, item in enumerate(value):
            converted = adopt(item)
            if converted is not item:
                value[index] = converted
    return value
