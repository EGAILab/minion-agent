"""Execute every `conformance/agent/*.yaml` Layer-07 inbox scenario.

Discriminated by a top-level `agent_inbox` key (`agent-inbox-scenario.schema.json`) -- these
exercise the real `Inbox` primitive directly (steer/follow_up/inject/claim/clear/
has_queued_messages), with no provider, no tool, and no run-loop timing at all, matching the
independent Rust review's own explicit request for direct, language-neutral Layer-07 canonical
evidence (`L07-R004`).
"""

from pathlib import Path

import pytest
import yaml

from .agent_inbox_runner import run_agent_inbox_scenario

AGENT_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent"


def _is_agent_inbox_scenario(path: Path) -> bool:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(document, dict) and "agent_inbox" in document


SCENARIOS = sorted(p for p in AGENT_DIR.glob("*.yaml") if _is_agent_inbox_scenario(p))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
def test_agent_inbox_scenario(scenario: Path) -> None:
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    observed = run_agent_inbox_scenario(document)

    expected = document["expect"]
    assert set(observed) == set(expected), (
        f"observed/expected observation names differ: {sorted(observed)} vs {sorted(expected)}"
    )
    for name, value in expected.items():
        assert observed[name] == value, name
