"""Execute the L12-D001 `conformance/agent/fs-path-domain/*.json` cases (the filesystem path
JavaScript-string
domain) against the real `LocalFileSystem`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .fs_path_runner import check, run_case

ROOT = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "fs-path-domain"
DOCUMENTS = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*.json"))]
CASES = [
    case
    for document in DOCUMENTS
    if "fs_path_domain" in document
    for case in document["fs_path_domain"]["cases"]
]


def test_the_fs_path_corpus_is_complete() -> None:
    assert len(DOCUMENTS) == 5
    assert len(CASES) == 56
    assert len(TOOL_CASES) == 5


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_fs_path_domain_case(case: dict[str, Any], tmp_path: Path) -> None:
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
