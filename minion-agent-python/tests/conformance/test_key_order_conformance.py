"""Execute the L0206-D001 (K1) `conformance/agent/key-order/*.json` cases: ECMAScript key
enumeration order of tool-argument objects at every argument boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .key_order_runner import check, run_case

ROOT = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "key-order"
DOCUMENTS = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*.json"))]
CASES = [case for document in DOCUMENTS for case in document["key_order"]["cases"]]


def test_the_key_order_corpus_is_complete() -> None:
    assert len(DOCUMENTS) == 1
    assert len(CASES) == 28


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_key_order_case(case: dict[str, Any], tmp_path: Path) -> None:
    check(case, await run_case(case, str(tmp_path)))
