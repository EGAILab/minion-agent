"""Execute every WP-13.4 `find`/`grep` canonical scenario on this host's platform, with the
certified pinned engines provisioned from `MINION_SEARCH_ENGINE_ARTIFACTS` (a directory holding the
official release artifacts, verified by SHA-256 on provisioning)."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from minion_agent.tools.builtin.search_engines import EngineStore

from .builtin_search_runner import (
    HOST_PLATFORM,
    SEARCH_DIR,
    run_builtin_search_scenario,
    temporary_store,
)

SCENARIOS = sorted(SEARCH_DIR.glob("builtin-search-*.yaml"))
_GREP_LINE = re.compile(r"^(.*?)(?::(-?\d+(?:\.\d+)?): |-(-?\d+(?:\.\d+)?)- )")


@pytest.fixture(scope="session")
def engine_store() -> EngineStore:
    artifacts = os.environ.get("MINION_SEARCH_ENGINE_ARTIFACTS")
    if not artifacts:  # pragma: no cover - the gates always provision the pinned engines
        pytest.skip(
            "MINION_SEARCH_ENGINE_ARTIFACTS is not set (pinned search engines not provisioned)"
        )
    return temporary_store(artifacts)


def _split(text: str) -> tuple[str, str | None]:
    body, sep, notice = text.partition("\n\n[")
    return body, ("[" + notice) if sep else None


def grep_files(body: str) -> dict[str, list[str]]:
    """Per-file line lists in output order; each file's lines must also be contiguous."""
    files: dict[str, list[str]] = {}
    order: list[str] = []
    for line in body.split("\n"):
        match = _GREP_LINE.match(line)
        key = match.group(1) if match else ""
        if not order or order[-1] != key:
            assert key not in files, f"{key}: lines are not contiguous"
            order.append(key)
        files.setdefault(key, []).append(line)
    return files


def compare(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    assert observed["is_error"] == expected["is_error"]
    mode = expected["mode"]
    if mode == "exact":
        assert observed["text"] == expected["text"]
        assert observed["details"] == expected["details"]
        return
    body, notice = _split(observed["text"])
    assert notice == expected["notice"]
    if mode == "find_multiset":
        assert sorted(body.split("\n")) == expected["entries"]
    elif mode == "find_subset":
        entries = body.split("\n")
        assert len(entries) == expected["count"]
        assert not Counter(entries) - Counter(expected["of"])
    elif mode == "grep_by_file":
        assert grep_files(body) == expected["files"]
    else:  # summary
        assert len(body.split("\n")) == expected["lines"]
        assert len(body.encode("utf-8")) == expected["body_bytes"]
        details = dict(observed["details"])
        if "truncation" in details:
            details["truncation"] = {
                k: v for k, v in details["truncation"].items() if k != "content"
            }
        assert details == expected["details"]
        return
    assert observed["details"] == expected["details"]


def test_builtin_search_scenarios_exist() -> None:
    assert len(SCENARIOS) >= 190


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_builtin_search_scenario(scenario: Path, engine_store: EngineStore) -> None:
    outcome = await run_builtin_search_scenario(
        yaml.safe_load(scenario.read_text(encoding="utf-8")), engine_store
    )
    if outcome is None:
        pytest.skip(f"no expectation for {HOST_PLATFORM}")
    compare(outcome["observed"], outcome["expected"])
