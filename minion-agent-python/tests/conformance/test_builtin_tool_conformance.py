"""Execute every `conformance/agent/*.yaml` built-in tool (WP-13.1 `read`/`ls`) scenario."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from .builtin_tool_runner import run_builtin_tool_scenario

AGENT_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent"


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


SCENARIOS = sorted(p for p in AGENT_DIR.glob("*.yaml") if "builtin_tool" in (_load(p) or {}))


def test_builtin_tool_scenarios_exist() -> None:
    assert SCENARIOS


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_builtin_tool_scenario(scenario: Path) -> None:
    for outcome in await run_builtin_tool_scenario(_load(scenario)):
        observed, expected, label = outcome["observed"], outcome["expected"], outcome["id"]
        for key in ("is_error", "text", "text_sha256", "image", "details", "fs_calls"):
            if key in expected:
                assert observed[key] == expected[key], f"{label}: {key}"
        if "text_tail" in expected:
            assert observed["text"].endswith(expected["text_tail"]), f"{label}: text_tail"
        if "probed_entries" in expected:
            assert observed["probed_entries"] == expected["probed_entries"], f"{label}: probes"
