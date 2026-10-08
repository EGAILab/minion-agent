"""WP-14.2 canonical prompt-assembly scenarios through the real functions (spec/harness.md WP-14.2;
`conformance/agent/prompt-assembly/`). Each document passes the schema's normative preflight
first: no unpaired surrogate anywhere (Owner decision `WP142-R001`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .prompt_assembly_runner import run
from .test_schema_validation import _prompt_preflight

SCENARIOS = sorted(
    (Path(__file__).resolve().parents[3] / "conformance" / "agent" / "prompt-assembly").glob(
        "*.json"
    )
)


def test_every_canonical_scenario_is_collected() -> None:
    assert len(SCENARIOS) == 66


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_prompt_assembly_scenario(path: Path) -> None:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert _prompt_preflight(document) == []
    case = document["prompt_assembly"]
    assert run(case) == case["expected"]
