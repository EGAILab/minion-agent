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


def _compare_details(observed: dict[str, Any], expected: dict[str, Any], body: str) -> None:
    """Details exact, except `truncation.content`: it must be the returned body itself (which
    entries survived depends on the unspecified traversal; the content must not)."""
    details = dict(observed)
    if "truncation" in details:
        truncation = dict(details["truncation"])
        assert truncation.pop("content", None) == body, "truncation.content is not the body"
        details["truncation"] = truncation
    assert details == expected


def _compare_grep(files: dict[str, list[str]], expected: dict[str, Any]) -> None:
    """Every file block exact and contiguous. Under a match limit or byte truncation, which
    files survive is unspecified and the last block may stop early (a prefix of its block); the
    returned line count is then fixed by the expected total unless bytes cut it."""
    partial = "truncation" in expected["details"] or "matchLimitReached" in expected["details"]
    allowed = expected["files"]
    names = list(files)
    for position, name in enumerate(names):
        assert name in allowed, f"{name}: not in the expected result"
        block, full = files[name], allowed[name]
        if partial and position == len(names) - 1:
            assert block == full[: len(block)], f"{name}: not a prefix of its block"
        else:
            assert block == full, f"{name}: block differs"
    if not partial:
        assert set(names) == set(allowed), "missing files"
    elif "truncation" not in expected["details"]:
        total = sum(len(lines) for lines in files.values())
        assert total == sum(len(lines) for lines in allowed.values()), "wrong line count"


def compare(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    assert observed["is_error"] == expected["is_error"]
    mode = expected["mode"]
    if mode == "exact":
        assert observed["text"] == expected["text"]
        _compare_details(observed["details"], expected["details"], observed["text"])
        return
    body, notice = _split(observed["text"])
    assert notice == expected["notice"]
    if mode == "find_multiset":
        assert sorted(body.split("\n")) == expected["entries"]
    elif mode == "find_subset":
        entries = body.split("\n")
        assert len(entries) == expected["count"]
        assert not Counter(entries) - Counter(expected["of"]), "entries outside the result"
    else:  # grep_by_file
        _compare_grep(grep_files(body), expected)
    _compare_details(observed["details"], expected["details"], body)


def test_builtin_search_scenarios_exist() -> None:
    assert len(SCENARIOS) >= 196


# ---- comparator controls (WP134-IMPL-R003): the comparisons reject wrong results ----


def _expected(name: str) -> dict[str, Any]:
    document = yaml.safe_load((SEARCH_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    expect: dict[str, Any] = document["builtin_search"]["expect"]
    return dict(expect.get(HOST_PLATFORM) or next(iter(expect.values())))


def _find_bulk(entries: list[str], content: str | None = None, *, omit: bool = False) -> Any:
    expected = _expected("builtin-search-find-bulk-default-limit-and-bytes")
    body = "\n".join(entries)
    truncation = {**expected["details"]["truncation"]}
    if not omit:
        truncation["content"] = body if content is None else content
    observed = {
        "is_error": False,
        "text": body + "\n\n" + expected["notice"],
        "details": {**expected["details"], "truncation": truncation},
    }
    return observed, expected


def _rejects(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    try:
        compare(observed, expected)
    except AssertionError:
        return True
    return False


def test_find_bulk_comparison_accepts_any_valid_subset_and_rejects_wrong_ones() -> None:
    expected = _expected("builtin-search-find-bulk-default-limit-and-bytes")
    allowed, count = expected["of"], expected["count"]
    valid = list(reversed(allowed))[:count]  # another traversal's subset, in another order
    assert not _rejects(*_find_bulk(valid))
    fabricated = ["Q" * 58] * count  # the reviewer's control: same bytes, nothing real
    assert _rejects(*_find_bulk(fabricated))
    assert _rejects(*_find_bulk([*valid[:-1], valid[0]]))  # a duplicate the corpus lacks
    assert _rejects(*_find_bulk(valid[:-1]))  # a missing entry
    assert _rejects(*_find_bulk([*valid[:-1], "f9999-extra.txt"]))  # an extra entry
    assert _rejects(*_find_bulk(valid, content="incorrect"))
    assert _rejects(*_find_bulk(valid, omit=True))


def test_grep_comparison_rejects_wrong_blocks() -> None:
    expected = _expected("builtin-search-grep-bulk-default-limit-and-bytes")
    lines = expected["files"]["w.txt"]

    def observed(body_lines: list[str]) -> dict[str, Any]:
        text = "\n".join(body_lines) + "\n\n" + expected["notice"]
        return {"is_error": False, "text": text, "details": dict(expected["details"])}

    assert not _rejects(observed(lines), expected)
    assert _rejects(observed([lines[1], lines[0], *lines[2:]]), expected)  # within-file order
    assert _rejects(observed(lines[:-1]), expected)  # a missing match
    assert _rejects(observed([*lines[:-1], "w.txt:999: fabricated"]), expected)
    assert _rejects(observed([*lines[:-1], lines[0]]), expected)  # a duplicate
    assert _rejects(observed([*lines[:-1], "z.txt:1: other"]), expected)  # an extra file
    unlimited = {
        **expected,
        "details": {},
        "notice": None,
        "files": {"a": ["a:1: x"], "b": ["b:1: y"]},
    }
    both = {"is_error": False, "text": "b:1: y\na:1: x", "details": {}}
    assert not _rejects(both, unlimited)  # files in another order
    assert _rejects({**both, "text": "a:1: x"}, unlimited)  # a missing file
    assert _rejects({**both, "text": "a:1: x\nb:1: y\na:2: x"}, unlimited)  # split block


def test_find_multiset_rejects_a_lost_duplicate() -> None:
    expected = {"is_error": False, "mode": "find_multiset", "notice": None, "details": {},
                "entries": ["src/same.ts", "src/same.ts"]}  # fmt: skip
    assert not _rejects(
        {"is_error": False, "text": "src/same.ts\nsrc/same.ts", "details": {}}, expected
    )
    assert _rejects({"is_error": False, "text": "src/same.ts", "details": {}}, expected)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
async def test_builtin_search_scenario(scenario: Path, engine_store: EngineStore) -> None:
    outcome = await run_builtin_search_scenario(
        yaml.safe_load(scenario.read_text(encoding="utf-8")), engine_store
    )
    if outcome is None:
        pytest.skip(f"no expectation for {HOST_PLATFORM}")
    compare(outcome["observed"], outcome["expected"])
