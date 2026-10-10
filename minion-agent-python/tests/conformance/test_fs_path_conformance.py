"""Execute the L12-D001 `conformance/agent/fs-path-domain/*.json` cases (the filesystem path
JavaScript-string
domain) against the real `LocalFileSystem`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .fs_path_runner import applies, check, run_case

ROOT = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "fs-path-domain"
DOCUMENTS = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*.json"))]
CASES = [
    case
    for document in DOCUMENTS
    if "fs_path_domain" in document
    for case in document["fs_path_domain"]["cases"]
]


def test_the_fs_path_corpus_is_complete() -> None:
    assert len(DOCUMENTS) == 8
    # L12-D006 adds fs-path-nul.json; L12-D007 adds fs-error-codes.json.
    assert len(CASES) == 122 + 237 + 176
    assert len(TOOL_CASES) == 5
    # L12-D001-R001: the error-origin cases declared for one platform only, each with its reason.
    limited = [
        case for case in CASES if "platforms" in case and not case["id"].startswith("errors/")
    ]
    assert len(limited) == 10
    assert all(
        case["platforms"] == ["linux"] and "#67" in case["platform_note"] for case in limited
    )
    # L12-D007: its Windows-only conditions (name class, sharing, byte-range lock), each with a
    # reason.
    windows_only = [
        case for case in CASES if case["id"].startswith("errors/") and "platforms" in case
    ]
    assert len(windows_only) == 64
    assert all(case["platforms"] == ["win32"] and case["platform_note"] for case in windows_only)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_fs_path_domain_case(case: dict[str, Any], tmp_path: Path) -> None:
    if not applies(case):
        pytest.skip(case["platform_note"])
    check(case, await run_case(case, tmp_path))


TOOL_CASES = [
    case
    for document in DOCUMENTS
    if "fs_path_tools" in document
    for case in document["fs_path_tools"]["cases"]
]


@pytest.mark.parametrize("case", TOOL_CASES, ids=[case["id"] for case in TOOL_CASES])
async def test_fs_path_tools_case(case: dict[str, Any], tmp_path: Path) -> None:
    from .fs_path_runner import run_tool_case

    check(case, await run_tool_case(case, tmp_path))
