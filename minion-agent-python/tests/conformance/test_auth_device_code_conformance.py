"""Execute every `conformance/agent/*.yaml` auth-device-code (Layer 11 Pass 1) scenario."""

from pathlib import Path

import pytest
import yaml

from .auth_device_code_runner import run_auth_device_code_scenario

AGENT_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent"


def _is_auth_device_code_scenario(path: Path) -> bool:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(document, dict) and "auth_device_code" in document


SCENARIOS = sorted(p for p in AGENT_DIR.glob("*.yaml") if _is_auth_device_code_scenario(p))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_auth_device_code_scenario(scenario: Path) -> None:
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    observed = await run_auth_device_code_scenario(document)
    expected = document["expect"]

    assert observed["poll_count"] == expected["poll_count"]
    if "complete" in expected:
        assert observed.get("complete") == expected["complete"]
    if "error" in expected:
        assert observed.get("error", {}).get("type") == expected["error"]["type"]
        if "message_contains" in expected["error"]:
            assert expected["error"]["message_contains"] in observed["error"]["message"]
