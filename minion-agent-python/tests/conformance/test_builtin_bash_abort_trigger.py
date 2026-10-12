"""`TEST-BASH-ABORT-TIMING` (#197, WP133-D001): the `builtin-bash-abort-during` scenario's abort
trigger, under a deterministic slow shell start-up (the mechanism of the full-suite flake).

A 600 ms delay before the spawn stands in for Git Bash start-up under load. With the original
400 ms wall-clock trigger the abort lands before `printf` runs, so `before` is lost (the observed
flake); with the `abort_on_output` trigger the abort waits for the command's own output and the
scenario's expectation holds."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from . import builtin_bash_runner
from .builtin_bash_runner import HOST_PLATFORM, run_builtin_bash_scenario

SCENARIO = (
    Path(__file__).resolve().parents[3]
    / "conformance"
    / "agent"
    / "builtin-bash"
    / "builtin-bash-abort-during.yaml"
)
SLOW_START_S = 0.6


def _scenario() -> dict[str, Any]:
    return yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))


@pytest.fixture
def slow_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtin_bash_runner.LocalSubprocess.spawn

    async def spawn(self: Any, argv: Any, options: Any = None) -> Any:
        await asyncio.sleep(SLOW_START_S)
        return await real(self, argv, options)

    monkeypatch.setattr(builtin_bash_runner.LocalSubprocess, "spawn", spawn)


def test_the_scenario_uses_the_output_trigger() -> None:
    spec = _scenario()["builtin_bash"]
    assert spec["abort_on_output"] == "before"
    assert "abort_after_ms" not in spec


async def test_output_trigger_holds_under_slow_start(slow_spawn: None) -> None:
    outcome = await run_builtin_bash_scenario(_scenario())
    assert outcome is not None
    assert outcome["observed"]["is_error"] is True
    assert outcome["observed"]["text"] == outcome["expected"]["text"]


async def test_wall_clock_trigger_loses_the_output_under_slow_start(slow_spawn: None) -> None:
    # The pre-WP133-D001 trigger: the flake's mechanism, reproduced deterministically.
    document = copy.deepcopy(_scenario())
    spec = document["builtin_bash"]
    del spec["abort_on_output"]
    spec["abort_after_ms"] = 400
    outcome = await run_builtin_bash_scenario(document)
    assert outcome is not None
    assert outcome["observed"]["text"] != outcome["expected"]["text"]
    assert "before" not in outcome["observed"]["text"]
    assert HOST_PLATFORM in ("win32", "linux")
