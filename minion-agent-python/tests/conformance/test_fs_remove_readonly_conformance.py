"""Execute the L12-D005 `conformance/agent/fs-remove-readonly/*.json` cases against the real
`LocalFileSystem.remove` (minion-agent#188, provenance #126)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .fs_remove_runner import applies, needs_unprivileged, run_case

ROOT = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "fs-remove-readonly"
DOCUMENTS = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*.json"))]


def test_the_fs_remove_readonly_corpus_is_complete() -> None:
    assert len(DOCUMENTS) == 21


@pytest.mark.parametrize("document", DOCUMENTS, ids=[d["name"] for d in DOCUMENTS])
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
