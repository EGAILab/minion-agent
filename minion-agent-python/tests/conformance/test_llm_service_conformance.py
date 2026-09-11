"""Execute every `conformance/agent/*.yaml` llm-service (Layer 10) scenario."""

from pathlib import Path

import pytest
import yaml

from .llm_service_runner import run_llm_service_scenario

AGENT_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent"


def _is_llm_service_scenario(path: Path) -> bool:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(document, dict) and "llm_service" in document


SCENARIOS = sorted(p for p in AGENT_DIR.glob("*.yaml") if _is_llm_service_scenario(p))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_llm_service_scenario(scenario: Path) -> None:
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    observed = await run_llm_service_scenario(document)

    for observation_id, expected in document["expect"].items():
        actual = observed[observation_id]
        if "resolve" in expected:
            assert actual["resolve"] == expected["resolve"], observation_id
        if "models" in expected:
            assert actual["models"] == expected["models"], observation_id
        if "settled" in expected:
            assert actual["settled"] == expected["settled"], observation_id
            if "error_message" in expected:
                assert actual.get("error_message") == expected["error_message"], observation_id
        if "raised" in expected:
            assert actual.get("raised") is True, observation_id
