"""Execute every `conformance/agent/builtin-mutation/*.yaml` (WP-13.2 `write`/`edit`/queue)
scenario, and replay pinned Pi's `normalizeForFuzzyMatch` results against `fuzzy_normalize`."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from minion_agent.tools.builtin._utf16 import from_units, to_units
from minion_agent.tools.builtin.edit_diff import fuzzy_normalize

from .builtin_mutation_runner import run_cases, run_queue

AGENT_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent"
SCENARIO_DIR = AGENT_DIR / "builtin-mutation"
FUZZY_FIXTURE = AGENT_DIR / "fixtures" / "wp132-fuzzy-normalize" / "fuzzy_normalize.json"


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


SCENARIOS = sorted(SCENARIO_DIR.glob("*.yaml"))


def test_builtin_mutation_scenarios_exist() -> None:
    assert len(SCENARIOS) == 26


def check_case(outcome: dict[str, Any]) -> None:
    observed, expected, label = outcome["observed"], outcome["expected"], outcome["id"]
    assert observed["is_error"] == expected["is_error"], f"{label}: is_error ({observed['text']!r})"
    if expected.get("argument_validation_failure"):
        # Layer 06's certified immediate argument-validation error; its text is Layer 06's own.
        assert "invalid arguments" in observed["text"], f"{label}: {observed['text']!r}"
    else:
        assert observed["text"] == expected["text"], f"{label}: text"
    for key in ("details", "fs_calls", "files_after"):
        if key in expected:
            assert observed[key] == expected[key], f"{label}: {key}"


def check_queue(run: dict[str, Any], expect: dict[str, Any]) -> None:
    log = run["log"]
    assert run["pending"] == [], f"calls never settled: {run['pending']}\n{log}"
    for call_id, want in expect["results"].items():
        got = run["results"][call_id]
        assert got["is_error"] == want["is_error"] and got["text"] == want["text"], (
            call_id,
            got,
            log,
        )
        if "details" in want:
            assert got["details"] == want["details"], (call_id, got)
    for first, second in expect.get("order", []):
        assert first in log, (first, log)
        if second in log:
            assert log.index(first) < log.index(second), (first, second, log)
    for event in expect.get("logged", []):
        assert event in log, (event, log)
    for event in expect.get("never", []):
        assert event not in log, (event, log)
    assert run["files_after"] == expect.get("files_after", []), run["files_after"]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_builtin_mutation_scenario(scenario: Path) -> None:
    document = _load(scenario)
    if "queue" in document["builtin_mutation"]:
        check_queue(await run_queue(document), document["builtin_mutation"]["queue"]["expect"])
        return
    for outcome in await run_cases(document):
        check_case(outcome)


def test_fuzzy_normalize_replays_pinned_pi() -> None:
    cases = json.loads(FUZZY_FIXTURE.read_text(encoding="utf-8"))["cases"]
    assert cases
    for case in cases:
        assert from_units(fuzzy_normalize(to_units(case["text"]))) == case["normalized"], case["id"]
