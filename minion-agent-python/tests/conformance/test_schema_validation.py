"""Every conformance scenario validates against its shape's JSON Schema.

Two scenario shapes currently coexist during the Pi-fidelity realignment (see
`minion-agent-docs/process/implementation-conformance-workflow.md` section 8.1): the legacy
per-family shape (`provider_script`/`steps`/`expect_*`, one schema file per family) and the newer
unified shape (`family`/`status`/`authority`/`pi_revision`/`given`/`when`/`expect`, one shared
schema). A scenario's own top-level `family` key is the discriminator -- its presence means the
unified schema governs, not the legacy per-family one.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

CONFORMANCE = Path(__file__).resolve().parents[3] / "conformance"

LEGACY_FAMILIES = {
    "runtime": CONFORMANCE / "schema" / "runtime-scenario.schema.json",
    "agent": CONFORMANCE / "schema" / "agent-scenario.schema.json",
    "session": CONFORMANCE / "schema" / "session-scenario.schema.json",
}
UNIFIED_SCHEMA = CONFORMANCE / "schema" / "scenario.schema.json"

# Families whose scenarios arrive in a later plan. Their schema must still exist
# and must still be a valid JSON Schema. Empty now that every family is
# populated; kept as the seam a new family is added through.
UNPOPULATED: set[str] = set()


def _scenarios(family: str) -> list[Path]:
    return sorted((CONFORMANCE / family).glob("*.yaml"))


def _schema_path_for(document: dict[str, Any], family: str) -> Path:
    """The unified shape's own `family` key is the discriminator (see module docstring)."""
    if "family" in document:
        return UNIFIED_SCHEMA
    return LEGACY_FAMILIES[family]


@pytest.mark.parametrize("family", sorted(LEGACY_FAMILIES))
def test_family_has_scenarios(family: str) -> None:
    if family in UNPOPULATED:
        pytest.skip(f"conformance/{family}/ is populated in a later plan")
    assert _scenarios(family), f"conformance/{family}/ has no scenarios"


@pytest.mark.parametrize(
    "schema_path", [*LEGACY_FAMILIES.values(), UNIFIED_SCHEMA], ids=lambda p: p.stem
)
def test_family_schema_is_wellformed(schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    ("family", "scenario"),
    [(f, s) for f in sorted(LEGACY_FAMILIES) for s in _scenarios(f)],
    ids=lambda value: value.stem if isinstance(value, Path) else value,
)
def test_scenario_validates(family: str, scenario: Path) -> None:
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    schema = json.loads(_schema_path_for(document, family).read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: list(error.path),
    )
    assert not errors, "\n".join(
        f"{'/'.join(str(part) for part in error.path)}: {error.message}" for error in errors
    )
