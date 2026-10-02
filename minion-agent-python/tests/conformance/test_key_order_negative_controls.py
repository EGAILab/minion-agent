"""L0206-D001 (K1) negative controls (contract section 4): each realistic WRONG key-order
implementation, installed at the seam it would live in, must make the canonical `key-order` corpus
FAIL, while the unmodified code passes. Every mutant is a runtime monkeypatch of production code;
the runner and the scenarios are unchanged."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from minion_agent.llm import ToolCallBlock
from minion_agent.llm import content as content_module
from minion_agent.llm import js_object as js_module
from minion_agent.session import derive as derive_module
from minion_agent.tools import execute as execute_module

from . import key_order_runner as runner
from .test_key_order_conformance import CASES


async def _failures(tmp_path: Path) -> list[str]:
    failed = []
    for index, case in enumerate(CASES):
        root = tmp_path / str(index)
        root.mkdir()
        try:
            runner.check(case, await runner.run_case(case, str(root)))
        except Exception:  # an assertion mismatch or an escaping exception both fail the case
            failed.append(case["id"])
    return failed


async def test_the_unmodified_code_passes_every_case(tmp_path: Path) -> None:
    assert await _failures(tmp_path) == []


def _plain_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(js_module.JsObject, "__setitem__", dict.__setitem__)


def _insertion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The binding as it was: insertion order everywhere."""
    monkeypatch.setattr(js_module, "es_order", list)
    _plain_assignment(monkeypatch)


def _sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rust's prepared BTreeMap: keys sorted."""
    monkeypatch.setattr(js_module, "es_order", sorted)
    _plain_assignment(monkeypatch)


def _top_level_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the top-level arguments object is ordered; nested objects keep insertion order."""

    def top_only(value: Any) -> Any:
        if isinstance(value, dict):
            ordered = {k: value[k] for k in js_module.es_order(list(value))}
            dict.clear(value)
            dict.update(value, ordered)
        return value

    monkeypatch.setattr(content_module, "order_in_place", top_only)
    monkeypatch.setattr(execute_module, "order_in_place", top_only)


def _non_canonical_numeral_as_index(monkeypatch: pytest.MonkeyPatch) -> None:
    def loose(key: str) -> bool:  # "01", "00", "4294967295" treated as indices
        return key.isascii() and key.isdigit() and len(key) < 12

    monkeypatch.setattr(js_module, "is_array_index", loose)


def _ordered_at_construction_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordered when the call is built, never again: a hook's assignments, appended objects and
    retained-reference mutations reach execute in insertion order (L0206-D001-R001)."""
    monkeypatch.setattr(execute_module, "order_in_place", lambda value: value)
    _plain_assignment(monkeypatch)


def _copy_on_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rejected checkpoint-1 mechanism: an assigned object is stored as an ordered COPY, so a
    hook's later mutation through its retained reference is lost (L0206-D001-R002)."""
    original = js_module.JsObject.__setitem__

    def setitem(self: Any, key: str, value: Any) -> None:
        original(self, key, js_module.order_in_place(copy.deepcopy(value)))

    monkeypatch.setattr(js_module.JsObject, "__setitem__", setitem)


def _crashing_index_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rejected checkpoint-1 classifier: int() on any decimal key (L0206-D001-R003)."""

    def converting(key: str) -> bool:
        if not key or not key.isascii() or not key.isdigit() or (len(key) > 1 and key[0] == "0"):
            return False
        return int(key) <= 4294967294

    monkeypatch.setattr(js_module, "is_array_index", converting)


def _schema_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation imposes the declared property order (then the rest)."""
    original = execute_module._validate

    def validate(definition: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = original(definition, arguments)
        declared = list((definition.parameters or {}).get("properties", {}))
        keys = [k for k in declared if k in validated] + [k for k in validated if k not in declared]
        return {k: validated[k] for k in keys}

    monkeypatch.setattr(execute_module, "_validate", validate)


def _replay_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Order is lost on replay: the decoded call's arguments come back key-sorted."""
    original = derive_module.decode_message

    def decode(data: Any) -> Any:
        message = original(data)
        blocks = tuple(
            dataclasses.replace(b, arguments=dict(sorted(b.arguments.items())))
            if isinstance(b, ToolCallBlock)
            else b
            for b in message.content
        )
        return dataclasses.replace(message, content=blocks)

    monkeypatch.setattr(runner, "decode_message", decode)


def _raw_boundaries_unordered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single seam (L0206-D001-R004): the raw object is ordered at construction only -- a later
    mutation of a call's arguments reaches persistence and the start/update payloads unordered.
    Every other boundary (validation, listeners, execute) is untouched."""
    from minion_agent.agent_loop import driver as driver_module
    from minion_agent.tools import batch as batch_module

    for module in (derive_module, driver_module, batch_module, execute_module):
        monkeypatch.setattr(module, "order_raw", lambda value: value)


MUTANTS: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "insertion-order": _insertion,
    "sorted-order": _sorted,
    "declared-schema-order": _schema_order,
    "top-level-only": _top_level_only,
    "non-canonical-numeral-as-index": _non_canonical_numeral_as_index,
    "ordered-at-construction-only": _ordered_at_construction_only,
    "order-lost-on-replay": _replay_sorted,
    "copy-on-assignment": _copy_on_assignment,
    "raw-boundaries-unordered": _raw_boundaries_unordered,
    "index-check-converts-any-decimal": _crashing_index_check,
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
async def test_a_wrong_implementation_fails_the_corpus(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    MUTANTS[name](monkeypatch)
    assert await _failures(tmp_path), f"{name} survived the K1 corpus"
