"""Execute every L12-D011 `spawn-nul` canonical case on this host's platform (spec/execution.md
section 21): the direct `ctx.subprocess.spawn` cases and the `find`/`grep`/`bash` tool cases."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from minion_agent.tools.builtin.search_engines import EngineStore

from .builtin_search_runner import temporary_store
from .spawn_nul_runner import HOST_PLATFORM, SPAWN_NUL_DIR, run_spawn_case, run_tool_case, text

DOCUMENT = json.loads((SPAWN_NUL_DIR / "spawn-nul.json").read_text(encoding="utf-8"))
CASES = DOCUMENT["spawn_nul"]["cases"]
TOOLS = DOCUMENT["spawn_nul"]["tools"]


@pytest.fixture(scope="module")
def engine_store() -> EngineStore:
    artifacts = os.environ.get("MINION_SEARCH_ENGINE_ARTIFACTS")
    if not artifacts:  # pragma: no cover - the gates always provision the pinned engines
        pytest.skip(
            "MINION_SEARCH_ENGINE_ARTIFACTS is not set (pinned search engines not provisioned)"
        )
    return temporary_store(artifacts)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
async def test_spawn_case(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    expected = case["expect"][HOST_PLATFORM]
    observed = await run_spawn_case(case, monkeypatch)
    assert observed["code"] == expected["code"]
    if expected["code"] == "aborted":
        return
    if expected["message"] == "os":
        assert not observed["nul_input"], "a control must reach the OS, not the NUL check"
    else:
        assert observed["nul_input"]
        assert observed["message"] == text(expected["message"])


@pytest.mark.parametrize("case", TOOLS, ids=lambda c: c["id"])
async def test_tool_case(case: dict[str, Any], tmp_path: Path, engine_store: EngineStore) -> None:
    expected = case["expect"][HOST_PLATFORM]
    observed = await run_tool_case(case, tmp_path, engine_store)
    assert observed["is_error"] is expected["is_error"]
    assert observed["text"] == text(expected["text"])


def test_every_case_runs_on_both_platforms() -> None:
    assert len(CASES) == 57 and len(TOOLS) == 7
    assert all(set(c["expect"]) == {"linux", "win32"} for c in CASES + TOOLS)
