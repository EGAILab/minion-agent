"""WP-14.1 canonical skill-discovery scenarios through the real loader over the real `ctx.fs`
(spec/harness.md WP-14.1; `conformance/agent/skill-discovery/`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from .skill_discovery_runner import diagnostic_matches, observed, run

SCENARIOS = sorted(
    (Path(__file__).resolve().parents[3] / "conformance" / "agent" / "skill-discovery").glob(
        "*.json"
    )
)


def test_every_canonical_scenario_is_collected() -> None:
    assert len(SCENARIOS) == 96


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
async def test_skill_discovery_scenario(path: Path, tmp_path: Path) -> None:
    scenario: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    spec = scenario["skill_discovery"]
    if spec.get("posix_only") and sys.platform == "win32":
        pytest.skip("POSIX-only names (Windows cannot create them)")
    result, base = await run(tmp_path, scenario)
    actual = observed(result, base)
    assert actual["skills"] == spec["expect"]["skills"]
    expected_diagnostics = spec["expect"]["diagnostics"]
    assert len(actual["diagnostics"]) == len(expected_diagnostics), actual["diagnostics"]
    for want, got in zip(expected_diagnostics, actual["diagnostics"], strict=True):
        assert diagnostic_matches(want, got), (want, got)
