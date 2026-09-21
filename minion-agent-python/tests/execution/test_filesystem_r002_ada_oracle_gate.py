"""`L12-PY-R002` version-stability mechanism 2 (root-characterization checkpoint --
`minion-agent-docs#121` @ `00afd5178d5c1bed4ec5175eea873061a9928fb1`, `AGREED FOR
IMPLEMENTATION: YES`, section 14): a permanent, mechanically-checked differential gate proving
the exact-pinned `ada-url==1.15.3` still reproduces the direct Ada 2.9.2 oracle this checkpoint
built and differentially tested exactly, over its own full 8,246-case systematic corpus -- not
merely the exact dependency pin in `pyproject.toml` alone. If a future PIN CHANGE is ever made
deliberately, THIS test fails loudly against the new version's own behavior before any
Pi-fidelity regression reaches production, forcing an explicit re-characterization pass
(matching this project's own convergence-checkpoint discipline) rather than a silent drift.

Data files (`tests/execution/data/r002_ada_oracle/`) are a verbatim copy of the checkpoint's own
committed evidence (`minion-agent-docs` repo, `assurance/layers/data/12-python-r002-ada-oracle/
systematic_corpus.txt` and `systematic_ada292.txt`) -- copied here, not referenced across repos,
so this gate is self-contained and runs in any checkout that has this repo alone. The oracle
values (`systematic_ada292.txt`) are the direct Ada 2.9.2 C++ engine's own output (built from the
vendored, checksum-verified `ada-2.9.2.{h,cpp}` singleheader source, confirmed to be the EXACT
version Node v22.19.0 vendors) -- an executable oracle, not a hand-derived expectation.

This runner deliberately performs NO decoding/validation logic of its own -- it only loads the
two data files and calls the real `_file_url_to_path` under test, keeping the behavior entirely
in production code."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from minion_agent.execution.filesystem import _file_url_to_path

_DATA_DIR = Path(__file__).parent / "data" / "r002_ada_oracle"


def _load_corpus() -> list[str]:
    with open(_DATA_DIR / "systematic_corpus.txt", encoding="utf-8") as f:
        return [line.rstrip("\n").rstrip("\r") for line in f if line.strip()]


def _load_oracle() -> dict[str, str | None]:
    d: dict[str, str | None] = {}
    with open(_DATA_DIR / "systematic_ada292.txt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if "\t" not in line:
                continue
            url, val = line.split("\t", 1)
            d[url] = None if val == "PARSE_ERROR" else val
    return d


@pytest.mark.skipif(
    os.name != "nt", reason="the committed oracle corpus targets Windows UNC output"
)
def test_r002_ada_oracle_differential_gate() -> None:
    corpus = _load_corpus()
    oracle = _load_oracle()

    assert len(corpus) == 8246, (
        f"committed systematic_corpus.txt has {len(corpus)} lines, expected 8246 -- has it been "
        f"edited without regenerating/re-verifying against the Ada 2.9.2 oracle?"
    )
    assert set(corpus) == set(oracle.keys()), (
        "systematic_corpus.txt and systematic_ada292.txt key sets differ -- one was edited "
        "without the other; regenerate both together from the checkpoint's own committed tooling"
    )

    failures: list[str] = []
    for url in corpus:
        expected_host = oracle[url]
        try:
            got = _file_url_to_path(url)
        except (ValueError, OSError):
            if expected_host is not None:
                failures.append(f"{url}: expected accepted host {expected_host!r}, but rejected")
            continue
        if expected_host is None:
            failures.append(f"{url}: expected rejection, but accepted -> {got!r}")
            continue
        expected = "\\\\" + expected_host + "\\share"
        if got != expected:
            failures.append(f"{url}: expected {expected!r}, got {got!r}")

    assert not failures, (
        f"{len(failures)}/{len(corpus)} cases diverge from the direct Ada 2.9.2 oracle "
        f"(showing up to 20):\n" + "\n".join(failures[:20])
    )
