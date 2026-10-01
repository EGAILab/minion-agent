"""Execute the gate-`WP-13.2` `conformance/agent/prepared-runtime-string/*.yaml` cases (L0506-D002):
the integration witness through the REAL built-in `edit` tool (TOOL-030 x TOOL-041 string domain).

Thin by design: each case gets a fresh root holding `f.txt` = "a\n" (the scenario schema's
`editCase` fixture) and one call through the real Layer 06 `execute_call` with the real `edit`
tool over the real `LocalFileSystem`. The hook records the prepared value at each observed pointer;
the runner only renders it to UTF-16 code units. The final bytes are the UTF-8 projection
boundary: an unpaired surrogate is written as EF BF BD there, never earlier.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import LocalFileSystem
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools.builtin import create_edit_tool
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

from .prepared_string_runner import at, preflight, units
from .test_prepared_string_conformance import DOCUMENTS

CASES = [
    case
    for document in DOCUMENTS
    if document["gate"] == "WP-13.2"
    for case in document["prepared_string"]["cases"]
]


def test_wp132_string_edit_gate_cases_exist() -> None:
    assert len(CASES) == 20
    assert all(case["tool"] == "edit" for case in CASES)


async def run_edit_case(case: dict[str, Any], root: Path) -> dict[str, Any]:
    preflight(case)
    (root / "f.txt").write_bytes(b"a\n")
    raw = copy.deepcopy(case["arguments"])
    seen: list[Any] = []
    registry = ToolRegistry()
    registry.register(create_edit_tool(LocalFileSystem(str(root))))
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(call: Any, d: Any, arguments: Any, signal: Any, next_: Any) -> Any:
        seen.extend(units(at(arguments, p)) for p in case["observe"])
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    call = ToolCallBlock(id="call-1", name="edit", arguments=case["arguments"])
    result = await execute_call(call, registry=registry, ctx=ctx)
    first = result.content[0]
    return {
        "is_error": result.is_error,
        "text": first.text if isinstance(first, TextBlock) else "",
        "hook": seen,
        "raw_unchanged": call.arguments == raw and case["arguments"] == raw,
        "file": (root / "f.txt").read_bytes(),
    }


def check_edit(case: dict[str, Any], run: dict[str, Any]) -> None:
    expect = case["expect"]
    assert run["raw_unchanged"], f"{case['id']}: raw ToolCall arguments changed"
    assert not run["is_error"], (case["id"], run["text"])
    assert run["hook"] == [expect["observed"][p] for p in case["observe"]], (
        case["id"],
        run["hook"],
    )
    assert run["text"] == expect["result_text"], (case["id"], run["text"])
    assert run["file"].hex() == expect["file_utf8_hex"], (case["id"], run["file"])


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_wp132_string_edit_gate_case(case: dict[str, Any], tmp_path: Path) -> None:
    check_edit(case, await run_edit_case(case, tmp_path))
