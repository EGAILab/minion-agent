"""Execute every conformance/agent/*.yaml scenario that scripts a `transform:` block.

These share the `agent` canonical family/directory with full-loop agent scenarios but exercise a
different observable surface (transform_messages()'s own output, not a turn's projected event
stream) -- see agent-transform-scenario.schema.json's own docstring for why this is a second
schema for the same directory, not a fourth canonical family.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from .placeholder import is_placeholder
from .transform_runner import run_transform_scenario

AGENT_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent"


def _is_transform_scenario(path: Path) -> bool:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(document, dict) and "transform" in document


SCENARIOS = sorted(p for p in AGENT_DIR.glob("*.yaml") if _is_transform_scenario(p))


def _for_comparison(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """A tool_result's `timestamp` is real, observable state for a pass-through result but
    wall-clock time for a synthesized one (spec/target-model-transformation.md) -- a scenario
    omitting `timestamp` from its expected tool_result is declining to assert it, not asserting
    it is absent, so drop it from the actual dict before comparing in that one case."""
    if actual.get("role") == "tool_result" and "timestamp" not in expected:
        return {k: v for k, v in actual.items() if k != "timestamp"}
    return actual


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
def test_transform_scenario(scenario: Path) -> None:
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    if is_placeholder(document):
        pytest.xfail(f"{scenario.stem}: TO_BE_FILLED placeholder, see pi-parity-manifest.yaml")

    outcome = run_transform_scenario(document)

    expected_error = document["expect"].get("error")
    if expected_error is None:
        assert outcome["error"] is None, outcome["error"]
        expected_messages = document["expect"]["messages"]
        actual_messages = outcome["messages"]
        assert len(actual_messages) == len(expected_messages)
        comparable = [
            _for_comparison(actual, expected)
            for actual, expected in zip(actual_messages, expected_messages, strict=True)
        ]
        assert comparable == expected_messages
    else:
        assert outcome["error"] is not None, "expected the scenario to raise"
        assert outcome["error"]["type"] == expected_error["type"]
        if "message_contains" in expected_error:
            assert expected_error["message_contains"] in outcome["error"]["message"]
