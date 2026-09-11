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


def _tests_field_violations(row: dict[str, Any]) -> list[str]:
    """Every violation of the `tests:` field's own structural contract for `row` -- empty when
    `row["tests"]` is well-formed. `tests` must be an actual `list` (`L10-R008`, second re-review):
    a scalar STRING is itself iterable, yielding one-character strings that are each individually
    non-empty, so a check that only inspects what iterating `row["tests"]` yields -- without first
    asserting the container itself is a `list` -- silently accepts a scalar string as if it were a
    well-formed one-entry list (the exact defect an earlier revision of this module had: it
    iterated `row["tests"]` directly and checked only the yielded entries' own non-emptiness, so
    `tests: "evidence"` passed both this check and the separate "at least one entry" truthiness
    check). The container-type assertion runs BEFORE any iteration, so a malformed container is
    rejected at the door. Factored out so the check itself is directly unit-testable against a
    synthetic scalar-container row, not only through the real manifest."""
    tests = row["tests"]
    row_id = row.get("id", "<no id>")
    if not isinstance(tests, list):
        return [f"{row_id}: tests must be a list, got {type(tests).__name__} ({tests!r})"]
    if not tests:
        return [f"{row_id}: tests is an empty list"]
    return [
        f"{row_id}: tests[{index}] is not a non-empty string ({type(entry).__name__})"
        for index, entry in enumerate(tests)
        if not isinstance(entry, str) or not entry.strip()
    ]


def test_every_row_tests_field_is_well_formed() -> None:
    """`L10-R008`: an unquoted `: ` inside a plain YAML scalar list item silently produces a
    one-key mapping instead of a string -- `yaml.safe_load` accepts it without error, so a mere
    "parses without exception" check does not catch malformed evidence entries. Every row's own
    `tests:` field must be a list, non-empty, of genuine non-empty evidence strings."""
    rows = _load_rows()
    violations = [violation for row in rows for violation in _tests_field_violations(row)]
    assert not violations, "\n".join(violations)


def test_tests_field_violations_rejects_a_scalar_string_container() -> None:
    """`L10-R008`, second re-review's own exact discriminating probe: a scalar string container
    (`tests: "evidence"`) must be rejected, not silently accepted because iterating it yields
    non-empty one-character strings."""
    row = {"id": "X", "tests": "evidence"}
    violations = _tests_field_violations(row)
    assert violations, "a scalar string tests field must be rejected, not silently accepted"
    assert "must be a list" in violations[0]


def test_tests_field_violations_accepts_a_well_formed_list() -> None:
    """The positive counterpart: a genuine non-empty list of non-empty strings passes cleanly."""
    row = {"id": "X", "tests": ["some-evidence"]}
    assert _tests_field_violations(row) == []


def test_tests_field_violations_rejects_an_empty_list() -> None:
    row = {"id": "X", "tests": []}
    violations = _tests_field_violations(row)
    assert violations, "an empty tests list must be rejected"
    assert "empty list" in violations[0]


def test_tests_field_violations_rejects_a_non_string_entry() -> None:
    row = {"id": "X", "tests": ["ok", {"a": "b"}]}
    violations = _tests_field_violations(row)
    assert violations, "a non-string entry must be rejected"
    assert "tests[1]" in violations[0]
