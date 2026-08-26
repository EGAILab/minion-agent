"""Execute every conformance/agent/*.yaml scenario that drives a full agent-loop turn.

Excludes XFORM (transform_messages()) scenarios in the same directory: those script a `transform`
block instead of `provider_script`/`steps` and are executed by test_transform_conformance.py
against the real transform_messages() seam directly, not through a full agent-loop turn -- see
agent-transform-scenario.schema.json's own docstring for why a pure transformer scenario does not
force a full run. Also excludes Layer-05 tool-registry scenarios (top-level `tool_registry` key,
tool-registry-scenario.schema.json): those exercise the real ToolRegistry/Context/scope seam
directly, not a full agent-loop turn either, and are executed by
test_tool_registry_conformance.py.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from .agent_runner import run_agent_scenario
from .placeholder import is_placeholder

AGENT_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent"


def _is_full_loop_scenario(path: Path) -> bool:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return not (
        isinstance(document, dict) and ("transform" in document or "tool_registry" in document)
    )


SCENARIOS = sorted(p for p in AGENT_DIR.glob("*.yaml") if _is_full_loop_scenario(p))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_agent_scenario(scenario: Path) -> None:
    document: dict[str, Any] = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    if is_placeholder(document):
        pytest.xfail(f"{scenario.stem}: TO_BE_FILLED placeholder, see pi-parity-manifest.yaml")

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
        expected_messages = document["expect_messages"]
        assert len(outcome["messages"]) == len(expected_messages)
        for actual, expected in zip(outcome["messages"], expected_messages, strict=True):
            assert actual["role"] == expected["role"]
            if "text_contains" in expected:
                # Host-library-specific error text (e.g. pydantic's own validation-error
                # formatting) is not a stable cross-language contract (TOOL-017) -- only the
                # Minion-authored prefix/substring is asserted, never the whole message.
                assert expected["text_contains"] in actual["text"]
            else:
                assert actual["text"] == expected["text"]

    if "expect_causes" in document:
        observed = [[cause["origin"] for cause in turn] for turn in outcome["causes"]]
        assert observed == document["expect_causes"]

    if "expect_assistant_stop_reasons" in document:
        assert outcome["assistant_stop_reasons"] == document["expect_assistant_stop_reasons"]

    if "expect_assistant_details" in document:
        assert outcome["assistant_details"] == document["expect_assistant_details"]

    if "expect_tool_completion_order" in document:
        assert outcome["tool_completion_order"] == document["expect_tool_completion_order"]

    if "expect_request_tools" in document:
        assert outcome["request_tools"] == document["expect_request_tools"]

    if "expect_updates" in document:
        assert outcome["updates"] == [list(entry) for entry in document["expect_updates"]]
