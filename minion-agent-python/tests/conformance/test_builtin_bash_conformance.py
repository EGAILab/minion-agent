"""Execute every `conformance/agent/*.yaml` WP-13.3 `bash` scenario on this host's platform."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from .builtin_bash_runner import HOST_PLATFORM, run_builtin_bash_scenario

# Own directory (as WP-13.2's builtin-mutation): the generic runners glob conformance/agent/*.yaml.
BASH_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "builtin-bash"


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


SCENARIOS = sorted(p for p in BASH_DIR.glob("*.yaml") if "builtin_bash" in _load(p))


def test_builtin_bash_scenarios_exist() -> None:
    assert len(SCENARIOS) >= 36


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_builtin_bash_scenario(scenario: Path) -> None:
    outcome = await run_builtin_bash_scenario(_load(scenario))
    if outcome is None:  # pragma: no cover - every scenario covers both certified platforms
        pytest.skip(f"no expectation for {HOST_PLATFORM}")
    observed, expected = outcome["observed"], outcome["expected"]
    assert observed["is_error"] == expected["is_error"]
    if expected.get("validation_rejected"):
        # This binding's Layer 06 validator rejection (TOOL-003); its wording is binding-specific.
        assert observed["text"].startswith("invalid arguments: ")
    elif "text" in expected:
        assert observed["text"] == expected["text"]
    else:
        assert observed["text_length"] == expected["text_length"]
        assert observed["text_head"] == expected["text_head"]
        assert observed["text_tail"] == expected["text_tail"]
    assert observed["details"] == expected["details"]
    assert observed["full_output"] == expected["full_output"]
    if observed["content_matches"] is not None:
        assert observed["content_matches"], "details.truncation.content is not the shown text"
