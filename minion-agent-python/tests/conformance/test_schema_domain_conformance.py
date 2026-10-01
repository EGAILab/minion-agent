"""Execute `conformance/agent/schema-domain/*.json` (L05-D001, `TOOL-016` / `TOOL-003`)."""

import json
from pathlib import Path
from typing import Any

import pytest

from .schema_domain_runner import check, run_case

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "schema-domain"
DOCUMENTS = [
    json.loads(path.read_text(encoding="utf-8")) for path in sorted(SCENARIO_DIR.glob("*.json"))
]
CASES = [case for document in DOCUMENTS for case in document["schema_domain"]["cases"]]


def test_schema_domain_cases_exist() -> None:
    assert len(DOCUMENTS) == 9 and len(CASES) == 729  # 9 roles x 9 schema x 9 instance members


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
async def test_schema_domain_case(case: dict[str, Any]) -> None:
    check(case, await run_case(case))
