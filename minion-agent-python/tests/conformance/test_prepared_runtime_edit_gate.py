"""Execute the gate-`WP-13.2` `conformance/agent/prepared-runtime/*.yaml` cases (L0506-D001-R003):
the integration witness through the REAL built-in `edit` tool (TOOL-030 x TOOL-041).

Thin by design: each case gets a fresh root holding `f.txt` = "a\\n" (the scenario schema's
`editCase` fixture) and one call through the real Layer 06 `execute_call` with the real `edit`
tool over the real `LocalFileSystem` -- its own `prepare_arguments` (`prepareEditArguments`),
validation, the `tools/pre-execute` hook and `execute`. The hook records the prepared value at
each observed pointer; the runner only renders it to a token (`prepared_runtime_runner.render`).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from minion_agent.execution import LocalFileSystem
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools.builtin import create_edit_tool
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

from .prepared_runtime_runner import at, preflight, render

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "prepared-runtime"
DOCUMENTS = [
    yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(SCENARIO_DIR.glob("*.yaml"))
]
CASES = [
    case
    for document in DOCUMENTS
    if document["gate"] == "WP-13.2"
    for case in document["prepared_runtime"]["cases"]
]


def test_wp132_edit_gate_cases_exist() -> None:
    assert len(CASES) == 8
    assert all(case["tool"] == "edit" for case in CASES)


async def run_edit_case(case: dict[str, Any], root: Path) -> dict[str, Any]:
    preflight(case)
    (root / "f.txt").write_bytes(b"a\n")
    raw = copy.deepcopy(case["arguments"])
    seen: list[str] = []
    registry = ToolRegistry()
    registry.register(create_edit_tool(LocalFileSystem(str(root))))
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(call: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        seen.extend(render(at(arguments, p)) for p in case["observe"])
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    call = ToolCallBlock(id="call-1", name="edit", arguments=case["arguments"])
    result = await execute_call(call, registry=registry, ctx=ctx)
    first = result.content[0]
    text = first.text if isinstance(first, TextBlock) else ""
    return {
        "is_error": result.is_error,
        "text": text,
        "hook": seen,
        "raw_unchanged": call.arguments == raw and case["arguments"] == raw,
        "file": (root / "f.txt").read_bytes(),
    }


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_wp132_edit_gate_case(case: dict[str, Any], tmp_path: Path) -> None:
    run = await run_edit_case(case, tmp_path)
    expect = case["expect"]
    assert run["raw_unchanged"], f"{case['id']}: raw ToolCall arguments changed"
    if expect["outcome"] == "prepared":
        assert not run["is_error"], (case["id"], run["text"])
        assert run["hook"] == [expect["observed"][p] for p in case["observe"]], (
            case["id"],
            run["hook"],
        )
        assert run["text"] == expect["result_text"], (case["id"], run["text"])
        assert run["file"] == b"b\n", case["id"]  # the edit itself applied
    else:
        assert run["is_error"] and "invalid arguments" in run["text"], (case["id"], run["text"])
        assert run["hook"] == [], case["id"]


def _projected(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _projected(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_projected(item) for item in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


@pytest.mark.parametrize(
    ("name", "mutant"),
    [
        # JSON-domain parsing: a JSON-only runtime cannot hold an overflow; it becomes null
        ("overflow mapped to null", lambda original: lambda text: _projected(original(text))),
        # Python's own int parsing: no binary64 rounding, and "-0" collapses to integer 0
        ("integers kept exact", lambda original: lambda text: __import__("json").loads(text)),
    ],
)
async def test_wp132_edit_gate_kills_runtime_domain_mutants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, mutant: Any
) -> None:
    """Negative control: each realistic wrong `JSON.parse` fails at least one gate case."""
    from minion_agent.tools.builtin import edit as edit_module

    del name
    monkeypatch.setattr(edit_module, "_json_parse", mutant(edit_module._json_parse))
    failures = 0
    for index, case in enumerate(CASES):
        root = tmp_path / str(index)
        root.mkdir()
        run = await run_edit_case(case, root)
        want = [case["expect"]["observed"][p] for p in case["observe"]]
        failures += run["hook"] != want or run["is_error"]
    assert failures > 0
