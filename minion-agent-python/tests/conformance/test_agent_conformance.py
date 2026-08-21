"""Execute every conformance/agent/*.yaml scenario."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from .agent_runner import run_agent_scenario

SCENARIOS = sorted((Path(__file__).resolve().parents[3] / "conformance" / "agent").glob("*.yaml"))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_agent_scenario(scenario: Path) -> None:
    document: dict[str, Any] = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    outcome = await run_agent_scenario(document)

    expected_error = document.get("expect_error")
    if expected_error is None:
        assert outcome["error"] is None, outcome["error"]
    else:
        assert outcome["error"] is not None, "expected the scenario to raise"
        assert outcome["error"]["type"] == expected_error["type"]
        if "message_contains" in expected_error:
            assert expected_error["message_contains"] in outcome["error"]["message"]

    if "expect_events" in document:
        assert outcome["events"] == document["expect_events"]

    if "expect_messages" in document:
        assert outcome["messages"] == document["expect_messages"]

    if "expect_causes" in document:
        observed = [[cause["origin"] for cause in turn] for turn in outcome["causes"]]
        assert observed == document["expect_causes"]

    if "expect_assistant_stop_reasons" in document:
        assert outcome["assistant_stop_reasons"] == document["expect_assistant_stop_reasons"]

    if "expect_tool_completion_order" in document:
        assert outcome["tool_completion_order"] == document["expect_tool_completion_order"]

    if "expect_request_tools" in document:
        assert outcome["request_tools"] == document["expect_request_tools"]
