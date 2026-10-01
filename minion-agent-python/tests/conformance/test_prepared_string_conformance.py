"""Execute the delta-gated `conformance/agent/prepared-runtime-string/*.yaml` cases (L0506-D002,
`TOOL-041` string domain)."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from .prepared_string_runner import check, run_case

SCENARIO_DIR = (
    Path(__file__).resolve().parents[3] / "conformance" / "agent" / "prepared-runtime-string"
)
DOCUMENTS = [
    yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(SCENARIO_DIR.glob("*.yaml"))
]
# The delta's certification gate, selected explicitly by `gate`. The gate-WP-13.2 real-edit document
# runs in test_prepared_string_edit_gate.py, never here (not skipped, aliased or simulated).
CASES = [
    case
    for document in DOCUMENTS
    if document["gate"] == "L0506-D002"
    for case in document["prepared_string"]["cases"]
]


def test_prepared_string_delta_gate_cases_exist() -> None:
    assert len(CASES) == 173  # 160 neighborhood x schema + 9 positions/keys + 4 replacements
    assert {document["gate"] for document in DOCUMENTS} == {"L0506-D002", "WP-13.2"}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
async def test_prepared_string_case(case: dict[str, Any]) -> None:
    check(case, await run_case(case))
