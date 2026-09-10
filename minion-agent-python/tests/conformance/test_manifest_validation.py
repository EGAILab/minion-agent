"""Structural validation of `pi-parity-manifest.yaml`, the single-source Pi-parity manifest
(`agent-workflow.md` section 8).

`L10-R008`: a manual, ad-hoc "N rows / N unique IDs" check run by hand each pass reported success
even though two Layer-10 evidence entries and two pre-existing `AG-007` entries were YAML mappings,
not strings -- an unquoted `: ` inside a plain scalar list item silently turns it into a one-key
mapping, and `yaml.safe_load` accepts that without complaint. This module makes that structural
check a permanent, automated gate instead of a manually-run one-liner that only checks row-ID
uniqueness.
"""

from pathlib import Path
from typing import Any

import yaml

MANIFEST_PATH = Path(__file__).resolve().parents[3] / "pi-parity-manifest.yaml"

REQUIRED_DISPOSITIONS = {"adopted", "deferred parity", "intentional divergence"}
REQUIRED_ROW_FIELDS = {
    "id",
    "phase",
    "pi",
    "surface",
    "rule",
    "tests",
    "python",
    "rust",
    "disposition",
}


def _load_rows() -> list[dict[str, Any]]:
    document = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = document["rows"]
    return rows


def test_manifest_row_ids_are_unique() -> None:
    rows = _load_rows()
    ids = [row["id"] for row in rows]
    duplicates = {row_id for row_id in ids if ids.count(row_id) > 1}
    assert not duplicates, f"duplicate manifest row id(s): {sorted(duplicates)}"


def test_every_row_has_the_required_fields() -> None:
    rows = _load_rows()
    for row in rows:
        missing = REQUIRED_ROW_FIELDS - set(row)
        assert not missing, f"{row.get('id', '<no id>')}: missing field(s) {sorted(missing)}"


def test_every_row_has_a_valid_disposition() -> None:
    """`agent-workflow.md` section 8: every relevant row must have an explicit disposition from
    the closed set below -- silence, or any other value, is not valid."""
    rows = _load_rows()
    for row in rows:
        assert row["disposition"] in REQUIRED_DISPOSITIONS, (
            f"{row['id']}: disposition {row['disposition']!r} is not one of "
            f"{sorted(REQUIRED_DISPOSITIONS)}"
        )


def test_every_tests_entry_is_a_non_empty_string() -> None:
    """`L10-R008`: an unquoted `: ` inside a plain YAML scalar list item silently produces a
    one-key mapping instead of a string -- `yaml.safe_load` accepts it without error, so a mere
    "parses without exception" check does not catch malformed evidence entries. Every `tests:`
    member must be a genuine, non-empty evidence string."""
    rows = _load_rows()
    malformed: list[tuple[str, int, str]] = []
    for row in rows:
        for index, entry in enumerate(row["tests"]):
            if not isinstance(entry, str) or not entry.strip():
                malformed.append((row["id"], index, type(entry).__name__))
    assert not malformed, f"non-string/empty tests[] entries: {malformed}"


def test_every_row_has_at_least_one_tests_entry() -> None:
    rows = _load_rows()
    empty = [row["id"] for row in rows if not row["tests"]]
    assert not empty, f"row(s) with an empty tests[] list: {empty}"
