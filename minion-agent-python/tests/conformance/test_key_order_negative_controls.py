"""L0206-D001 (K1) negative controls (contract section 4): each realistic WRONG key-order
implementation, installed at the seam it would live in, must make the canonical `key-order` corpus
FAIL, while the unmodified code passes. Every mutant is a runtime monkeypatch of production code;
the runner and the scenarios are unchanged."""

from __future__ import annotations

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


def _recursive(order: Callable[[dict[str, Any]], list[str]], deep: bool = True) -> Any:
    def convert(value: Any, top: bool = True) -> Any:
        if isinstance(value, dict):
            keys = order(value)
            return {k: (convert(value[k], False) if deep else value[k]) for k in keys}
        if isinstance(value, list):
            return [convert(item, False) if deep else item for item in value]
        return value

    return convert


def _install(monkeypatch: pytest.MonkeyPatch, convert: Any) -> None:
    """Replace the binding's ES ordering everywhere it is applied with `convert` (plain dicts)."""
    monkeypatch.setattr(content_module, "js_object", convert)
    monkeypatch.setattr(execute_module, "js_object", convert)
    monkeypatch.setattr(execute_module, "JsObject", lambda value=(): convert(dict(value)))


def _insertion(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _recursive(list))


def _sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _recursive(sorted))


def _top_level_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _recursive(lambda d: list(js_module.JsObject(dict.fromkeys(d))), False))


def _non_canonical_numeral_as_index(monkeypatch: pytest.MonkeyPatch) -> None:
    def loose(key: str) -> bool:  # "01", "00", "4294967295" treated as indices
        return key.isascii() and key.isdigit()

    monkeypatch.setattr(js_module, "is_array_index", loose)


def _ordered_at_construction_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordered when built, but a later assignment (a hook's in-place mutation) simply appends."""
    monkeypatch.setattr(
        js_module.JsObject,
        "__setitem__",
        lambda self, key, value: dict.__setitem__(self, key, js_module.js_object(value)),
    )
    original = js_module.JsObject.__init__

    def init(self: Any, items: Any = (), /) -> None:
        pairs = list(items.items() if isinstance(items, dict) else items)
        last = dict(pairs)
        indices = sorted((k for k in last if js_module.is_array_index(k)), key=int)
        original(self, [(k, last[k]) for k in [*indices, *(k for k in last if k not in indices)]])

    monkeypatch.setattr(js_module.JsObject, "__init__", init)


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


MUTANTS: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "insertion-order": _insertion,
    "sorted-order": _sorted,
    "declared-schema-order": _schema_order,
    "top-level-only": _top_level_only,
    "non-canonical-numeral-as-index": _non_canonical_numeral_as_index,
    "ordered-at-construction-only": _ordered_at_construction_only,
    "order-lost-on-replay": _replay_sorted,
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
async def test_a_wrong_implementation_fails_the_corpus(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    MUTANTS[name](monkeypatch)
    assert await _failures(tmp_path), f"{name} survived the K1 corpus"
