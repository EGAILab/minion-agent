"""Every conformance scenario validates against its family's JSON Schema."""

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

CONFORMANCE = Path(__file__).resolve().parents[2] / "conformance"

FAMILIES = {
    "runtime": CONFORMANCE / "schema" / "runtime-scenario.schema.json",
    "agent": CONFORMANCE / "schema" / "agent-scenario.schema.json",
}

# Families whose scenarios arrive in a later plan. Their schema must still exist
# and must still be a valid JSON Schema.
UNPOPULATED = {"agent"}


def _scenarios(family: str) -> list[Path]:
    return sorted((CONFORMANCE / family).glob("*.yaml"))


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_has_scenarios(family: str) -> None:
    if family in UNPOPULATED:
        pytest.skip(f"conformance/{family}/ is populated in a later plan")
    assert _scenarios(family), f"conformance/{family}/ has no scenarios"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_schema_is_wellformed(family: str) -> None:
    schema = json.loads(FAMILIES[family].read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    ("family", "scenario"),
    [(f, s) for f in sorted(FAMILIES) for s in _scenarios(f)],
    ids=lambda value: value.stem if isinstance(value, Path) else value,
)
def test_scenario_validates(family: str, scenario: Path) -> None:
    schema = json.loads(FAMILIES[family].read_text(encoding="utf-8"))
    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: list(error.path),
    )
    assert not errors, "\n".join(
        f"{'/'.join(str(part) for part in error.path)}: {error.message}" for error in errors
    )
