"""Execute `conformance/agent/raw-arguments/*.json` (L0206-D002, `AI-003` raw value domain)."""

import json
from pathlib import Path
from typing import Any

import pytest

from .raw_arguments_runner import check, run_case

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "raw-arguments"
DOCUMENTS = [
    json.loads(path.read_text(encoding="utf-8")) for path in sorted(SCENARIO_DIR.glob("*.json"))
]
CASES = [case for document in DOCUMENTS for case in document["raw_arguments"]["cases"]]


def test_raw_arguments_cases_exist() -> None:
    assert len(CASES) == 33  # 22 strings + 10 numbers + 1 key case


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
async def test_raw_arguments_case(case: dict[str, Any]) -> None:
    check(case, await run_case(case))
