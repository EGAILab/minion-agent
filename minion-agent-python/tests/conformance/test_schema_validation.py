"""Every conformance scenario validates against its shape's JSON Schema.

Three scenario shapes currently coexist during the Pi-fidelity realignment (see
`minion-agent-docs/process/implementation-conformance-workflow.md` section 8.1): the legacy
per-family shape (`provider_script`/`steps`/`expect_*`, one schema file per family), the unified
shape (`family`/`status`/`authority`/`pi_revision`/`given`/`when`/`expect`, one shared schema), and
the transform (XFORM) shape (`transform`/`expect`, `agent-transform-scenario.schema.json`) -- a
second schema for `conformance/agent/`'s own directory, not a fourth canonical family, since XFORM
scenarios exercise `transform_messages()` directly rather than a full agent-loop turn. A scenario's
own top-level `transform` key routes to the transform schema; `family` routes to the unified schema;
otherwise the legacy per-family schema governs.
"""

import json
import re
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
TRANSFORM_SCHEMA = CONFORMANCE / "schema" / "agent-transform-scenario.schema.json"

# Families whose scenarios arrive in a later plan. Their schema must still exist
# and must still be a valid JSON Schema. Empty now that every family is
# populated; kept as the seam a new family is added through.
UNPOPULATED: set[str] = set()


def _scenarios(family: str) -> list[Path]:
    return sorted((CONFORMANCE / family).glob("*.yaml"))


def _schema_path_for(document: dict[str, Any], family: str) -> Path:
    """The unified shape's own `family` key and the transform shape's own `transform` key are the
    discriminators (see module docstring)."""
    if "transform" in document:
        return TRANSFORM_SCHEMA
    if "family" in document:
        return UNIFIED_SCHEMA
    return LEGACY_FAMILIES[family]


@pytest.mark.parametrize("family", sorted(LEGACY_FAMILIES))
def test_family_has_scenarios(family: str) -> None:
    if family in UNPOPULATED:
        pytest.skip(f"conformance/{family}/ is populated in a later plan")
    assert _scenarios(family), f"conformance/{family}/ has no scenarios"


@pytest.mark.parametrize(
    "schema_path",
    [*LEGACY_FAMILIES.values(), UNIFIED_SCHEMA, TRANSFORM_SCHEMA],
    ids=lambda p: p.stem,
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


def _session_document(append: dict[str, Any]) -> dict[str, Any]:
    return {"name": "t", "steps": [{"append": append}], "expect_messages": []}


@pytest.mark.parametrize(
    "append",
    [
        pytest.param(
            {"role": "user", "content": [{"type": "thinking", "thinking": "x"}]},
            id="user+thinking",
        ),
        pytest.param(
            {
                "role": "user",
                "content": [{"type": "tool_call", "id": "1", "name": "n", "arguments": {}}],
            },
            id="user+tool_call",
        ),
        pytest.param(
            {"role": "assistant", "content": [{"type": "image", "mime_type": "m", "data": "x"}]},
            id="assistant+image",
        ),
        pytest.param(
            {
                "role": "tool_result",
                "tool_name": "t",
                "content": [{"type": "thinking", "thinking": "x"}],
            },
            id="tool_result+thinking",
        ),
        pytest.param(
            {
                "role": "tool_result",
                "tool_name": "t",
                "content": [{"type": "tool_call", "id": "1", "name": "n", "arguments": {}}],
            },
            id="tool_result+tool_call",
        ),
    ],
)
def test_session_role_invalid_content_combinations_are_rejected(append: dict[str, Any]) -> None:
    """A role cannot carry a content-block variant Pi's frozen per-role union
    forbids (`packages/ai/src/types.ts`) -- delta finding C."""
    schema = json.loads(LEGACY_FAMILIES["session"].read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(_session_document(append)))
    assert errors, f"expected role/content mismatch to be rejected: {append}"


@pytest.mark.parametrize(
    "append",
    [
        pytest.param({"role": "user", "content": [{"type": "text", "text": "x"}]}, id="user+text"),
        pytest.param(
            {"role": "user", "content": [{"type": "image", "mime_type": "m", "data": "x"}]},
            id="user+image",
        ),
        pytest.param(
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]},
            id="assistant+thinking",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "id": "1", "name": "n", "arguments": {}}],
            },
            id="assistant+tool_call",
        ),
        pytest.param(
            {
                "role": "tool_result",
                "tool_name": "t",
                "content": [{"type": "image", "mime_type": "m", "data": "x"}],
            },
            id="tool_result+image",
        ),
    ],
)
def test_session_role_valid_content_combinations_are_accepted(append: dict[str, Any]) -> None:
    """The positive counterpart to the rejection test above -- confirms the
    restriction narrows exactly to Pi's frozen per-role union, not further."""
    schema = json.loads(LEGACY_FAMILIES["session"].read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(_session_document(append)))
    assert not errors, f"expected role/content combination to validate: {append}\n{errors}"


@pytest.mark.parametrize(
    ("name", "expect_valid"),
    [
        ("plugin/foo", True),
        ("plugin/foo-bar", True),
        ("plugin/foo_bar", True),
        ("plugin2/foo", True),
        ("Plugin/foo", False),
        ("plugin-name/foo", False),
        ("plugin//foo", False),
        ("/foo", False),
        ("plugin/", False),
    ],
)
def test_session_event_name_pattern_matches_the_canonical_rule(
    name: str, expect_valid: bool
) -> None:
    """Pins the exact canonical event-name shape so a future implementation
    (Rust's validator currently disagrees) can be checked against it, not
    against prose (delta finding F). `plugin-name/foo` is invalid: the first
    segment excludes `-`; only later segments allow it."""
    schema = json.loads(LEGACY_FAMILIES["session"].read_text(encoding="utf-8"))
    pattern = schema["$defs"]["step"]["properties"]["append"]["properties"]["role"]["pattern"]
    matched = re.fullmatch(pattern, name) is not None
    assert matched == expect_valid, f"{name!r}: matched={matched}, expected={expect_valid}"
