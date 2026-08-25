"""Execute every `conformance/agent/*.yaml` tool-registry (Layer 05) scenario."""

from pathlib import Path

import pytest
import yaml

from .tool_registry_runner import run_tool_registry_scenario

AGENT_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent"


def _is_tool_registry_scenario(path: Path) -> bool:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(document, dict) and "tool_registry" in document


SCENARIOS = sorted(p for p in AGENT_DIR.glob("*.yaml") if _is_tool_registry_scenario(p))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_tool_registry_scenario(scenario: Path) -> None:
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    observed = await run_tool_registry_scenario(document)

    for query_id, expected in document["expect"].items():
        actual = observed[query_id]
        if "names" in expected:
            assert actual["names"] == expected["names"], query_id
        if "schemas" in expected:
            assert actual["schemas"] == expected["schemas"], query_id
        if "resolve" in expected:
            assert actual["resolve"] == expected["resolve"], query_id
