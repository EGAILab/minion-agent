"""Execute the delta-gated `conformance/agent/prepared-runtime/*.yaml` cases (L0506-D001,
`TOOL-041`)."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from .prepared_runtime_runner import run_case

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "prepared-runtime"
DOCUMENTS = [
    yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(SCENARIO_DIR.glob("*.yaml"))
]
# L0506-D001-R003: the delta's certification gate, selected explicitly by `gate`. The gate-WP-13.2
# real-edit document runs with WP-13.2, never here (not skipped, aliased or simulated).
CASES = [
    case
    for document in DOCUMENTS
    if document["gate"] == "L0506-D001"
    for case in document["prepared_runtime"]["cases"]
]


def test_prepared_runtime_delta_gate_cases_exist() -> None:
    assert len(CASES) == 50  # 19 + 31 numeric-keyword cases (L0506-D001-RC002)
    assert {document["gate"] for document in DOCUMENTS} == {"L0506-D001", "WP-13.2"}


def check(case: dict[str, Any], run: dict[str, Any]) -> None:
    expect = case["expect"]
    assert run["raw_unchanged"], f"{case['id']}: raw ToolCall arguments changed"
    assert run["outcome"] == expect["outcome"], (case["id"], run["text"])
    if expect["outcome"] == "prepared":
        want = [expect["observed"][p] for p in case["observe"]]
        assert run["hook"] == want, (case["id"], "hook", run["hook"])
        assert run["execute"] == want, (case["id"], "execute", run["execute"])
    else:
        assert run["hook"] == [] and run["execute"] == [], case["id"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
async def test_prepared_runtime_case(case: dict[str, Any]) -> None:
    check(case, await run_case(case))
