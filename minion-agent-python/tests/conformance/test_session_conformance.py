"""Execute every conformance/session/*.yaml scenario."""

from pathlib import Path

import pytest
import yaml

from .session_runner import run_session_scenario

SCENARIOS = sorted((Path(__file__).resolve().parents[2] / "conformance" / "session").glob("*.yaml"))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
def test_session_scenario(scenario: Path) -> None:
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))

    assert run_session_scenario(document) == document["expect_messages"]
