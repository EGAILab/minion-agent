"""Execute the L0506-D003 `conformance/agent/tool-result-domain/*.json` cases (the tool-result
runtime value domain):
every case's result observed at the after-hook, `tools/execution-end`, the ToolResultMessage and the
session log
replay; the gate-WP-13.2 document through the REAL edit tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .tool_result_domain_runner import check, preflight, run_case

ROOT = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "tool-result-domain"
DOCUMENTS = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*.json"))]
CASES = [
    (document, case) for document in DOCUMENTS for case in document["tool_result_domain"]["cases"]
]


def test_the_tool_result_corpus_is_complete() -> None:
    assert len(DOCUMENTS) == 7
    assert len(CASES) == 143
    gates = [case for document, case in CASES if document.get("gate") == "WP-13.2"]
    assert [case["id"] for case in gates] == ["edit/lone-high-new-text"]


@pytest.mark.parametrize(("document", "case"), CASES, ids=[case["id"] for _, case in CASES])
async def test_tool_result_domain_case(
    document: dict[str, Any], case: dict[str, Any], tmp_path: Path
) -> None:
    preflight(case)
    check(case, await run_case(case, tmp_path))
