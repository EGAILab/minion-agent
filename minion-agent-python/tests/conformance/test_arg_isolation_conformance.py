"""Execute the L0506-D005 `conformance/agent/arg-isolation/*.json` cases against the real
`execute_call` pipeline (`TOOL-003`; minion-agent#129)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .arg_isolation_runner import run_case

ROOT = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "arg-isolation"
DOCUMENTS = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*.json"))]


def test_the_arg_isolation_corpus_is_complete() -> None:
    assert len(DOCUMENTS) == 21


@pytest.mark.parametrize("document", DOCUMENTS, ids=[d["name"] for d in DOCUMENTS])
async def test_arg_isolation_case(document: dict[str, Any]) -> None:
    scenario = document["arg_isolation"]
    observed = await run_case(scenario)
    expect = scenario["expect"]
    assert observed["outcome"] == expect["outcome"]
    assert observed["hook_entries"] == expect["hook_entries"]
    assert observed["facts"] == expect["facts"]
    assert observed["execute"] == expect["execute"]
    assert observed["updates"] == expect["updates"]
    assert observed["raw_after"] == expect["raw_after"]
