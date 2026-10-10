"""Execute the L0506-D005 `conformance/agent/arg-isolation/*.json` cases against the real
`execute_call` pipeline (`TOOL-003`; minion-agent#129)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .arg_isolation_runner import run_case

ROOT = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "arg-isolation"
DOCUMENTS = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*.json"))]

# L0506-D005 contract candidate: the raw-schema path validates a SHALLOW copy, so these cases fail
# until the Python correction lands (then this set and its markers are removed). The cyclic case is
# characterization-only (Owner decision L0506D005-Q001; finding #193) and is not in this corpus.
L0506_D005_PENDING = {
    "arg-isolation-blocked-after-nested-mutation",
    "arg-isolation-hook-inserts-exponent-spelled-number",
    "arg-isolation-hook-inserts-integer-spelled-by-number-tostring",
    "arg-isolation-hook-nested-existing-gains-index",
    "arg-isolation-hook-pushes-object-into-nested-array",
    "arg-isolation-hook-reorders-nested-by-index-key",
    "arg-isolation-hook-replaces-and-deletes-nested",
    "arg-isolation-hook-sets-into-nested-object",
    "arg-isolation-nested-runtime-values-survive-the-clone",
    "arg-isolation-prepared-reused-raw-child-is-isolated",
    "arg-isolation-raw-integer-beyond-2p53-is-binary64",
    "arg-isolation-raw-exponent-overflow-is-infinity",
    "arg-isolation-raw-integer-401-digits-is-infinity",
    "arg-isolation-raw-integer-at-overflow-threshold-is-infinity",
    "arg-isolation-raw-integer-largest-finite-binary64",
    "arg-isolation-raw-negative-integer-401-digits-is-negative-infinity",
    "arg-isolation-two-hooks-share-the-validated-graph",
}


def test_the_arg_isolation_corpus_is_complete() -> None:
    assert len(DOCUMENTS) == 21
    assert {d["name"] for d in DOCUMENTS} >= L0506_D005_PENDING


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(
            d,
            id=d["name"],
            marks=[pytest.mark.xfail(strict=True, reason="L0506-D005 pending")]
            if d["name"] in L0506_D005_PENDING
            else [],
        )
        for d in DOCUMENTS
    ],
)
async def test_arg_isolation_case(document: dict[str, Any]) -> None:
    scenario = document["arg_isolation"]
    observed = await run_case(scenario)
    expect = scenario["expect"]
    assert observed["outcome"] == expect["outcome"]
    assert observed["hook_entries"] == expect["hook_entries"]
    assert observed["facts"] == expect["facts"]
    assert observed["execute"] == expect["execute"]
    assert observed["updates"] == expect["updates"]
    assert observed["raw_after"] == expect["raw_after"]
