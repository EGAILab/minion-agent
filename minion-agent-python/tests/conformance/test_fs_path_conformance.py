"""Execute the L12-D001 `conformance/agent/fs-path-domain/*.json` cases (the filesystem path
JavaScript-string
domain) against the real `LocalFileSystem`."""

from __future__ import annotations

import json
import sys
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
    assert len(DOCUMENTS) == 7
    assert len(CASES) == 122 + 237  # L12-D006 adds fs-path-nul.json
    assert len(TOOL_CASES) == 5
    # L12-D001-R001: the error-origin cases declared for one platform only, each with its reason.
    limited = [case for case in CASES if "platforms" in case]
    assert len(limited) == 10
    assert all(
        case["platforms"] == ["linux"] and "#67" in case["platform_note"] for case in limited
    )


def _l12_d006_pending(case_id: str) -> bool:
    """L12-D006 contract candidate (minion-agent#65 + #133-F1): the NUL-path cases the certified
    binding does not meet yet -- an escaped `ValueError`, the Linux `canonical_path`/`resolve`
    component walk, and a projected logical fallback. Removed with the Python correction."""
    if not case_id.startswith("nul/") or case_id.startswith("nul/aborted/"):
        return False
    where, op = case_id.split("/")[1], case_id.rsplit("/", 1)[1]
    if where == "url-control":  # L12D006-C001 controls: only the rename to a `%00` URL is pending
        return op == "rename_file-to-url-nul"
    if op in {"absolute_path", "read_text_lines-max0", "check_readable", "check_read_write"}:
        return False
    if op in {"canonical_path", "target_key"}:
        return sys.platform != "win32" or where == "lone-surrogate-and-nul"
    return True


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case["id"],
            marks=[pytest.mark.xfail(strict=True, reason="L12-D006 pending")]
            if _l12_d006_pending(case["id"])
            else [],
        )
        for case in CASES
    ],
)
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
