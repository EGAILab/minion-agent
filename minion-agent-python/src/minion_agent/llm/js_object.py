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

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, SupportsIndex

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
    """A tool-argument object that keeps ECMAScript order on its own assignments (a new array-index
    key moves to its ascending position) and is a Minion graph seam (Owner K1 Q1 decision,
    `minion-agent#100` comment `5947071963`): a value ATTACHED to it is ordered in place before the
    assignment returns -- the same object, never a copy (`R002`) -- and every value READ through it
    is ordered in place before it is exposed, so a retained alias mutated out of order is repaired
    by the next graph-mediated read."""

    __slots__ = ()

    def __init__(self, items: Mapping[str, Any] | Iterable[tuple[str, Any]] = (), /) -> None:
        super().__init__()
        pairs = items.items() if isinstance(items, Mapping) else items
        for key, value in pairs:
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        order_in_place(value)  # attachment: the same object, ordered before this returns
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

    def __getitem__(self, key: str) -> Any:
        return order_in_place(super().__getitem__(key))

    def get(self, key: str, default: Any = None, /) -> Any:
        return order_in_place(super().get(key, default))

    def values(self) -> Any:
        order_in_place(self)
        return super().values()

    def items(self) -> Any:
        order_in_place(self)
        return super().items()

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


class JsArray(list[Any]):
    """A tool-argument array: the Minion graph seam for array mutation (Owner K1 Q1 decision, §4).
    `append`, `insert`, `extend`, `+=` and element or slice replacement order the attached value in
    place before returning; element reads, slices and iteration order what they expose (§5)."""

    __slots__ = ()

    def __init__(self, items: Iterable[Any] = (), /) -> None:
        super().__init__()
        self.extend(items)

    def append(self, value: Any) -> None:
        super().append(order_in_place(value))

    def insert(self, index: SupportsIndex, value: Any) -> None:
        super().insert(index, order_in_place(value))

    def extend(self, values: Iterable[Any]) -> None:
        super().extend([order_in_place(value) for value in values])

    def __iadd__(self, values: Iterable[Any]) -> JsArray:  # type: ignore[misc]
        self.extend(values)
        return self

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice):
            super().__setitem__(index, [order_in_place(item) for item in value])
        else:
            super().__setitem__(index, order_in_place(value))

    def __getitem__(self, index: Any) -> Any:
        result = super().__getitem__(index)
        if isinstance(index, slice):
            for item in result:
                order_in_place(item)
            return result
        return order_in_place(result)

    def __iter__(self) -> Iterator[Any]:
        for item in list.__iter__(self):
            yield order_in_place(item)


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
            pending.extend(list.__iter__(item))
            continue
        keys = list(dict.__iter__(item))
        wanted = es_order(keys)
        if wanted != keys:
            values = {k: dict.__getitem__(item, k) for k in keys}
            dict.clear(item)
            for k in wanted:
                dict.__setitem__(item, k, values[k])
        pending.extend(dict.values(item))
    return value


def structured_clone(value: Any) -> Any:
    """`L0506-D005` (`TOOL-003`): pinned Pi's `structuredClone` of the prepared arguments. Every
    object and array reachable from `value` is copied exactly once -- a container reached twice
    becomes ONE copy, so aliases and cycles inside the graph survive -- as a `JsObject` / `JsArray`
    enumerating as its source does. Every other value is carried as is. The copy shares no
    container with `value`. Iterative, so nesting depth is not bounded by the interpreter stack."""

    def fresh(item: Any) -> Any:
        return JsObject() if isinstance(item, dict) else JsArray()

    if not isinstance(value, (dict, list)):
        return value
    memo: dict[int, Any] = {id(value): fresh(value)}
    pending = [value]
    while pending:
        source = pending.pop()
        target = memo[id(source)]
        children = (
            dict.items(source) if isinstance(source, dict) else enumerate(list.__iter__(source))
        )
        for key, child in children:
            if isinstance(child, (dict, list)):
                if id(child) not in memo:
                    memo[id(child)] = fresh(child)
                    pending.append(child)
                child = memo[id(child)]
            if isinstance(target, dict):
                dict.__setitem__(target, key, child)
            else:
                list.append(target, child)
    return order_in_place(memo[id(value)])


def order_raw(arguments: Any) -> Any:
    """`L0206-D001-R004`: order a call's RAW arguments object where it is observed or serialized
    (session encoding, the session tool-call record, the execution-start and update payloads). The
    raw object is shared and mutable after construction, so construction-time ordering alone is
    not enough. A separate name from `order_in_place` so each boundary family is controllable."""
    return order_in_place(arguments)


def _children(container: Any) -> list[Any]:
    return list(dict.values(container)) if isinstance(container, dict) else list(container)


def _graph_reachable(value: Any) -> dict[int, Any]:
    """Every container at or below an existing `JsObject`/`JsArray` anywhere in `value`, by id."""
    seen: set[int] = set()
    roots: list[Any] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if not isinstance(item, (dict, list)) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (JsObject, JsArray)):
            roots.append(item)
        pending.extend(_children(item))
    kept: dict[int, Any] = {}
    pending = roots
    while pending:
        item = pending.pop()
        if not isinstance(item, (dict, list)) or id(item) in kept:
            continue
        kept[id(item)] = item
        pending.extend(_children(item))
    return kept


def adopt(value: Any) -> Any:
    """`CE-L0206-D001-01`: make every object and array in a value the pipeline is about to OWN a
    `JsObject` / `JsArray`, recursively, so the graph's own seams hold the rule from then on. The
    pipeline owns a call's raw arguments as a provider decoded them (construction) and the graph a
    `prepare_arguments` shim hands it (`L0206-D001-R007`).

    Only the native frontier is adopted: a plain `dict`/`list` that no existing graph container
    reaches, and its plain descendants. Everything at or below an existing `JsObject`/`JsArray` is
    kept as is, wherever else the value also places it, so adoption never replaces an object a hook
    or caller already shares with the graph, nor a native container attached into the graph later
    (no copy, `R002`; Owner Q2: it stays native), and never splits a reference the value holds twice
    across that frontier (`R4-C001`). A plain container reached twice becomes one adopted container,
    so aliasing and cycles inside the adopted value survive."""
    memo = _graph_reachable(value)

    def convert(item: Any) -> Any:
        if not isinstance(item, (dict, list)):
            return item
        if id(item) in memo:
            return memo[id(item)]
        if isinstance(item, dict):
            js_object = JsObject()
            memo[id(item)] = js_object
            for key, child in dict.items(item):
                dict.__setitem__(js_object, key, convert(child))
            return order_in_place(js_object)
        array = JsArray()
        memo[id(item)] = array
        list.extend(array, [convert(child) for child in item])
        return array

    return convert(value)
