"""Execute every conformance/runtime/*.yaml scenario against the runtime."""

from pathlib import Path

import pytest
import yaml

from .runner import run_runtime_scenario

SCENARIOS = sorted((Path(__file__).resolve().parents[3] / "conformance" / "runtime").glob("*.yaml"))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_runtime_scenario(scenario: Path) -> None:
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    outcome = await run_runtime_scenario(document)

    if "expect_error" in document:
        expected = document["expect_error"]
        observed = outcome.error
        failed_fibers = [
            entry
            for entry in outcome.trace
            if entry["event"] == "fiber_state" and entry["state"] == "failed"
        ]
        assert observed is not None or failed_fibers, (
            f"expected {expected['type']} to surface as a raised error or a failed fiber"
        )
        if observed is not None:
            assert type(observed).__name__ == expected["type"]
            if "message_contains" in expected:
                assert expected["message_contains"] in str(observed)
    else:
        assert outcome.error is None, f"scenario raised: {outcome.error!r}"

    assert outcome.trace == document["expect_trace"]

    if "expect_result" in document:
        assert outcome.result == document["expect_result"]
