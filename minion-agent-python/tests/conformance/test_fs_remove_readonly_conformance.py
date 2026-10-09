"""Execute the L12-D005 `conformance/agent/fs-remove-readonly/*.json` cases against the real
`LocalFileSystem.remove` (minion-agent#188, provenance #126)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from .fs_remove_runner import applies, needs_unprivileged, run_case

ROOT = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "fs-remove-readonly"
DOCUMENTS = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*.json"))]

# L12-D005 contract candidate: the certified Python binding does not yet ignore the Windows
# read-only attribute. These cases fail until the correction lands. The markers are strict, so the
# defect is demonstrated and no marker can outlive the fix.
L12_D005_PENDING_WIN32 = {
    "fs-remove-readonly-rec-readonly-child-file",
    "fs-remove-readonly-rec-readonly-target-file",
    "fs-remove-readonly-nonrec-readonly-target-file",
    "fs-remove-readonly-rec-nested-readonly-files",
    "fs-remove-readonly-rec-readonly-dir-attribute-in-tree",
    "fs-remove-readonly-rec-readonly-empty-dir-target",
    "fs-remove-readonly-rec-readonly-nonempty-dir-target",
    "fs-remove-readonly-rec-mixed-readonly-tree",
    "fs-remove-readonly-rec-readonly-symlink-itself",
    "fs-remove-readonly-rec-readonly-symlink-to-readonly-external-file",
}


def test_the_fs_remove_readonly_corpus_is_complete() -> None:
    assert len(DOCUMENTS) == 21
    assert {d["name"] for d in DOCUMENTS} >= L12_D005_PENDING_WIN32


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(
            d,
            id=d["name"],
            marks=pytest.mark.xfail(
                sys.platform == "win32" and d["name"] in L12_D005_PENDING_WIN32,
                strict=True,
                reason="L12-D005: Python correction pending",
            ),
        )
        for d in DOCUMENTS
    ],
)
async def test_fs_remove_readonly_case(document: dict[str, Any], tmp_path: Path) -> None:
    if not applies(document):
        pytest.skip(f"{document['name']}: runs on {document['platforms']} only")
    if needs_unprivileged(document):
        pytest.skip(f"{document['name']}: root bypasses POSIX permission checks; run as non-root")
    observed = await run_case(document, tmp_path)
    case = document["fs_remove"]
    assert observed["expect"] == case["expect"]
    assert observed["expect_left"] == case["expect_left"]
    if "expect_external" in case:
        assert observed["expect_external"] == case["expect_external"]
