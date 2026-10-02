"""K1 / L0206-D001 independence witness for WP-13.2 (Owner decision, #99, section 6).

ECMAScript object key enumeration order (K1) is tracked separately as L0206-D001. Before recording
it as non-blocking for WP-13.2, this witness shows mechanically that no TOOL-029..033 owned output
depends on argument key order. Semantically identical calls whose argument objects enumerate their
keys in different orders -- including array-index-like extra keys ("0", "1", "10"), which the open
write/edit schemas admit -- must produce identical results, details (diff/patch), filesystem bytes,
error text, queue outcomes and cancellation outcomes through the real Layer-06 pipeline.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import LocalFileSystem
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin import create_edit_tool, create_write_tool
from minion_agent.tools.builtin import edit as edit_module
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

FIXTURE = b"alpha\nbeta\ngamma\n"


def _permute(mapping: dict[str, Any], order: list[str]) -> dict[str, Any]:
    """A fresh copy (L0206-D001: a ToolCallBlock orders its arguments IN PLACE, so variants must
    not share nested objects with each other or with the module fixtures)."""
    assert sorted(order) == sorted(mapping)
    return {key: copy.deepcopy(mapping[key]) for key in order}


def _orders(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    """Insertion orders whose enumeration differs in every relevant way: as written, reversed,
    index-like keys first ascending (ECMAScript), and sorted (a sorted-map binding)."""
    keys = list(mapping)
    index_like = sorted((k for k in keys if k.isdigit()), key=int)
    other = [k for k in keys if not k.isdigit()]
    return [
        _permute(mapping, keys),
        _permute(mapping, keys[::-1]),
        _permute(mapping, index_like + other),
        _permute(mapping, sorted(keys)),
    ]


async def _run(
    root: Path, calls: list[tuple[str, dict[str, Any]]], aborted: bool = False
) -> dict[str, Any]:
    fs = LocalFileSystem(str(root))
    registry = ToolRegistry()
    registry.register(create_write_tool(fs))  # type: ignore[arg-type]
    registry.register(create_edit_tool(fs))  # type: ignore[arg-type]
    ctx = Context()
    declare_tools_events(ctx.events)
    controller = RunAbortController()
    if aborted:
        controller.abort()
    results = await asyncio.gather(
        *(
            execute_call(
                ToolCallBlock(id=f"c{index}", name=name, arguments=arguments),
                registry=registry,
                ctx=ctx,
                signal=controller.signal,
            )
            for index, (name, arguments) in enumerate(calls)
        )
    )
    observed = []
    for result in results:
        first = result.content[0]
        observed.append(
            {
                "is_error": result.is_error,
                "text": first.text if isinstance(first, TextBlock) else repr(first),
                "details": json.dumps(result.details, sort_keys=True, default=repr),
            }
        )
    files = {p.name: p.read_bytes() for p in sorted(root.iterdir()) if p.is_file()}
    return {"results": observed, "files": files}


async def _all_equal(
    tmp_path: Path, variants: list[list[tuple[str, dict[str, Any]]]], aborted: bool = False
) -> None:
    outcomes = []
    for index, calls in enumerate(variants):
        root = tmp_path / str(index)
        root.mkdir(parents=True)
        (root / "f.txt").write_bytes(FIXTURE)
        outcomes.append(await _run(root, calls, aborted))
    assert all(outcome == outcomes[0] for outcome in outcomes[1:]), outcomes


EXTRA = {"0": "x", "10": "y", "1": "z", "b": True}

WRITE = {"path": "f.txt", "content": "new\ncontent\n", **EXTRA}
EDIT = {"path": "f.txt", "edits": [{"oldText": "beta", "newText": "BETA", **EXTRA}], **EXTRA}
EDIT_MULTI = {
    "path": "f.txt",
    "edits": [
        {"oldText": "alpha", "newText": "A", "2": 0},
        {"oldText": "gamma", "newText": "G", "1": 0},
    ],
    **EXTRA,
}
EDIT_NOT_FOUND = {
    "path": "f.txt",
    "edits": [{"oldText": "missing", "newText": "x", **EXTRA}],
    **EXTRA,
}
WRITE_DIRECTORY = {"path": ".", "content": "x", **EXTRA}  # an error classification path


def _item_orders(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Top-level orders x nested edit-item orders."""
    variants = []
    for top in _orders(arguments):
        if "edits" in top:
            for index in range(4):
                variants.append({**top, "edits": [_orders(item)[index] for item in top["edits"]]})
        else:
            variants.append(top)
    return variants


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("write", WRITE),
        ("edit", EDIT),
        ("edit", EDIT_MULTI),
        ("edit", EDIT_NOT_FOUND),
        ("write", WRITE_DIRECTORY),
    ],
    ids=["write", "edit", "edit-multi", "edit-not-found", "write-directory-error"],
)
async def test_owned_outputs_do_not_depend_on_key_order(
    name: str, arguments: dict[str, Any], tmp_path: Path
) -> None:
    await _all_equal(tmp_path, [[(name, variant)] for variant in _item_orders(arguments)])


async def test_edits_as_a_json_string_do_not_depend_on_key_order(tmp_path: Path) -> None:
    """The prepareEditArguments JSON.parse path: identical edits, different key orders."""
    variants = []
    for item in _orders({"oldText": "beta", "newText": "BETA", **EXTRA}):
        for top in _orders({"path": "f.txt", "edits": json.dumps([item]), **EXTRA}):
            variants.append([("edit", top)])
    await _all_equal(tmp_path, variants)


async def test_queue_outcomes_do_not_depend_on_key_order(tmp_path: Path) -> None:
    """Two concurrent mutations of one file (the mutation queue serializes them in call order)."""
    first = {"path": "f.txt", "edits": [{"oldText": "beta", "newText": "B1"}], **EXTRA}
    second = {"path": "f.txt", "content": "after\n", **EXTRA}
    variants = [
        [("edit", a), ("write", b)] for a, b in zip(_orders(first), _orders(second), strict=True)
    ]
    await _all_equal(tmp_path, variants)


async def test_cancellation_outcomes_do_not_depend_on_key_order(tmp_path: Path) -> None:
    await _all_equal(
        tmp_path, [[("edit", variant)] for variant in _item_orders(EDIT)], aborted=True
    )
    await _all_equal(
        tmp_path / "w", [[("write", variant)] for variant in _orders(WRITE)], aborted=True
    )


def test_the_variants_really_enumerate_differently() -> None:
    """Guard: the witness is only discriminating if its variants' key enumerations differ."""
    enumerations = {tuple(variant) for variant in _orders(EDIT)}
    assert len(enumerations) == 4
    nested = {tuple(variant["edits"][0]) for variant in _item_orders(EDIT)}
    assert len(nested) == 4


async def test_an_order_dependent_output_is_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative control: a mutant edit whose result text leaks argument key enumeration order (a
    realistic wrong implementation, e.g. echoing the arguments) must make the witness fail."""
    real = edit_module.create_edit_tool

    def leaky(fs: Any) -> Any:
        definition = real(fs)
        original = definition.execute

        async def execute(tool_call_id: str, arguments: dict[str, Any], *rest: Any) -> Any:
            result = await original(tool_call_id, arguments, *rest)
            first = result.content[0]
            text = (first.text if isinstance(first, TextBlock) else "") + " " + ",".join(arguments)
            return dataclasses.replace(result, content=(TextBlock(text=text),))

        return dataclasses.replace(definition, execute=execute)

    monkeypatch.setitem(globals(), "create_edit_tool", leaky)
    with pytest.raises(AssertionError):
        await _all_equal(tmp_path, [[("edit", variant)] for variant in _item_orders(EDIT)])
