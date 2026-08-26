"""Every conformance scenario validates against its shape's JSON Schema.

Five scenario shapes currently coexist during the Pi-fidelity realignment (see
`minion-agent-docs/process/implementation-conformance-workflow.md` section 8.1): the legacy
per-family shape (`provider_script`/`steps`/`expect_*`, one schema file per family), the unified
shape (`family`/`status`/`authority`/`pi_revision`/`given`/`when`/`expect`, one shared schema), the
transform (XFORM) shape (`transform`/`expect`, `agent-transform-scenario.schema.json`), the
tool-registry (Layer 05) shape (`tool_registry`/`expect`, `tool-registry-scenario.schema.json`),
and the agent-inbox (Layer 07) shape (`agent_inbox`/`expect`, `agent-inbox-scenario.schema.json`)
-- the second through fourth are all extra schemas for `conformance/agent/`'s own directory, not
additional canonical families, since XFORM/tool-registry/agent-inbox scenarios each exercise a pure
library seam (`transform_messages()`, the real `ToolRegistry`/`Context`/scope seam, the real
`Inbox` primitive) rather than a full agent-loop turn. A scenario's own top-level `tool_registry`
key routes to the tool-registry schema; `transform` routes to the transform schema; `agent_inbox`
routes to the agent-inbox schema; `family` routes to the unified schema; otherwise the legacy
per-family schema governs.
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
TOOL_REGISTRY_SCHEMA = CONFORMANCE / "schema" / "tool-registry-scenario.schema.json"
AGENT_INBOX_SCHEMA = CONFORMANCE / "schema" / "agent-inbox-scenario.schema.json"

# Families whose scenarios arrive in a later plan. Their schema must still exist
# and must still be a valid JSON Schema. Empty now that every family is
# populated; kept as the seam a new family is added through.
UNPOPULATED: set[str] = set()


def _scenarios(family: str) -> list[Path]:
    return sorted((CONFORMANCE / family).glob("*.yaml"))


def _schema_path_for(document: dict[str, Any], family: str) -> Path:
    """The unified shape's own `family` key, the transform shape's own `transform` key, the
    tool-registry shape's own `tool_registry` key, and the agent-inbox shape's own `agent_inbox`
    key are the discriminators (see module docstring)."""
    if "tool_registry" in document:
        return TOOL_REGISTRY_SCHEMA
    if "agent_inbox" in document:
        return AGENT_INBOX_SCHEMA
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
    [
        *LEGACY_FAMILIES.values(),
        UNIFIED_SCHEMA,
        TRANSFORM_SCHEMA,
        TOOL_REGISTRY_SCHEMA,
        AGENT_INBOX_SCHEMA,
    ],
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


_MISSING = object()


def _tool_registry_document(
    *,
    omit_parameters: bool = False,
    parameters: Any = None,
    constrained_sampling: Any = _MISSING,
) -> dict[str, Any]:
    tool: dict[str, Any] = {"name": "t", "description": "d", "label": "T"}
    if not omit_parameters:
        tool["parameters"] = parameters
    if constrained_sampling is not _MISSING:
        tool["constrained_sampling"] = constrained_sampling
    return {
        "name": "probe",
        "family": "agent",
        "authority": "a",
        "pi_revision": "p",
        "tool_registry": {
            "plugins": [{"id": "p1", "tools": [tool]}],
            "steps": [{"mount": "p1"}],
            "queries": [{"id": "q1"}],
        },
        "expect": {"q1": {"names": ["t"]}},
    }


@pytest.mark.parametrize(
    ("kwargs", "expect_valid"),
    [
        pytest.param(
            {"parameters": {"type": "object", "properties": {}}}, True, id="empty-object-schema"
        ),
        pytest.param(
            {
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                }
            },
            True,
            id="explicit-object-schema",
        ),
        pytest.param({"parameters": {"type": "string"}}, True, id="non-object-instance-schema"),
        pytest.param(
            {"parameters": {"oneOf": [{"type": "string"}, {"type": "number"}]}},
            True,
            id="top-level-combinator-schema",
        ),
        pytest.param({"omit_parameters": True}, False, id="missing"),
        pytest.param({"parameters": None}, False, id="null"),
        pytest.param({"parameters": True}, False, id="boolean-true"),
        pytest.param({"parameters": False}, False, id="boolean-false"),
    ],
)
def test_tool_registry_parameters_domain(kwargs: dict[str, Any], expect_valid: bool) -> None:
    """`L05-R005`: `parameters` is required and object-valued -- the JSON *representation* is a
    mapping, matching pinned Pi's `Tool<TParameters extends TSchema>` (generic over TypeBox's
    whole `TSchema` domain, not narrowed to `TObject`). A non-object-instance schema
    (`{"type": "string"}`) and a top-level combinator (`oneOf`) are therefore valid, same as an
    object-instance schema; only missing, explicit null, and the JSON-Schema-spec
    boolean-shorthand forms are outside the domain. This canonical schema itself already admitted
    the non-object-instance/combinator cases correctly (its own `"type": "object"` constrains the
    *value* to be a mapping, not a nested keyword) -- the regression this pins was in the Python
    public constructor, not here."""
    schema = json.loads(TOOL_REGISTRY_SCHEMA.read_text(encoding="utf-8"))
    document = _tool_registry_document(**kwargs)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if expect_valid:
        assert not errors, [error.message for error in errors]
    else:
        assert errors, "expected this parameters value to be rejected"


@pytest.mark.parametrize(
    ("constrained_sampling", "expect_valid"),
    [
        pytest.param(_MISSING, True, id="absent"),
        pytest.param(False, True, id="false"),
        pytest.param({"type": "json_schema", "strict": "require"}, True, id="json_schema"),
        pytest.param({"type": "grammar", "variants": {}}, True, id="grammar-empty-variants"),
        pytest.param(
            {"type": "grammar", "variants": {"openai_lark": "x"}}, True, id="grammar-lark-only"
        ),
        pytest.param(
            {"type": "grammar", "variants": {"openai_regex": "x"}}, True, id="grammar-regex-only"
        ),
        pytest.param(
            {"type": "grammar", "variants": {"openai_lark": "x", "openai_regex": "y"}},
            True,
            id="grammar-both",
        ),
        pytest.param(
            {"type": "grammar", "variants": {"unknown_key": "x"}}, False, id="grammar-unknown-key"
        ),
        pytest.param(None, False, id="explicit-null"),
    ],
)
def test_tool_registry_constrained_sampling_domain(
    constrained_sampling: Any, expect_valid: bool
) -> None:
    """`L05-R001` (grammar keys closed to pinned Pi's two formats, empty `variants: {}` is
    Pi-valid at the Tool-model boundary -- Pi's own runtime rejection of an empty grammar
    selection happens at provider request-construction time, Real Providers/Layer 11 territory)
    and `L05-R006` (explicit `null` is not a fifth alias for the absent state; a scenario meaning
    "absent" omits the key entirely)."""
    schema = json.loads(TOOL_REGISTRY_SCHEMA.read_text(encoding="utf-8"))
    document = _tool_registry_document(
        parameters={"type": "object", "properties": {}}, constrained_sampling=constrained_sampling
    )
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if expect_valid:
        assert not errors, [error.message for error in errors]
    else:
        assert errors, "expected this constrained_sampling value to be rejected"


def _agent_inbox_document(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "t",
        "family": "agent",
        "authority": "x",
        "pi_revision": "x",
        "agent_inbox": {"actions": [action]},
        "expect": {},
    }


@pytest.mark.parametrize(
    "action",
    [
        pytest.param({"steer": {"text": "A"}, "follow_up": {"text": "B"}}, id="steer+follow_up"),
        pytest.param(
            {"claim": {"queue": "steering", "mode": "all"}, "clear": {"queue": "all"}},
            id="claim+clear",
        ),
        pytest.param({"observe": "x"}, id="observe-with-no-operation"),
    ],
)
def test_agent_inbox_action_rejects_ambiguous_or_empty_operations(
    action: dict[str, Any],
) -> None:
    """`L07-R004` (second independent Rust review): an action naming more than one
    operation key -- or none at all -- has no defined runner order/effect and must
    be rejected by the schema itself, not merely avoided by convention in fixtures."""
    schema = json.loads(AGENT_INBOX_SCHEMA.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(_agent_inbox_document(action)))
    assert errors, f"expected this action to be rejected: {action}"


def test_agent_inbox_action_accepts_exactly_one_operation() -> None:
    """The positive counterpart: a single operation key still validates cleanly."""
    schema = json.loads(AGENT_INBOX_SCHEMA.read_text(encoding="utf-8"))
    document = _agent_inbox_document({"steer": {"text": "A"}})
    errors = list(Draft202012Validator(schema).iter_errors(document))
    assert not errors, [error.message for error in errors]


@pytest.mark.parametrize(
    "action",
    [
        pytest.param({"steer": {"text": "A"}, "observe": "x"}, id="steer+observe"),
        pytest.param({"follow_up": {"text": "A"}, "observe": "x"}, id="follow_up+observe"),
        pytest.param({"inject": {"text": "A"}, "observe": "x"}, id="inject+observe"),
        pytest.param({"clear": {"queue": "all"}, "observe": "x"}, id="clear+observe"),
    ],
)
def test_agent_inbox_action_rejects_observe_on_enqueue_or_clear_operations(
    action: dict[str, Any],
) -> None:
    """`L07-R004` (second independent Rust review): `observe` names a return value to
    check against `expect`, but enqueue/clear operations have no pinned-Pi return value
    to observe -- only `claim`/`has_queued_messages` do. A prior fix made operation keys
    mutually exclusive but left `observe` attachable to any of them regardless."""
    schema = json.loads(AGENT_INBOX_SCHEMA.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(_agent_inbox_document(action)))
    assert errors, f"expected observe to be rejected on this operation: {action}"


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(
            {"claim": {"queue": "steering", "mode": "all"}, "observe": "x"}, id="claim+observe"
        ),
        pytest.param({"has_queued_messages": {}, "observe": "x"}, id="has_queued_messages+observe"),
    ],
)
def test_agent_inbox_action_accepts_observe_on_claim_or_pending_operations(
    action: dict[str, Any],
) -> None:
    """The positive counterpart: `observe` remains valid on the two operations that
    actually return something to check."""
    schema = json.loads(AGENT_INBOX_SCHEMA.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(_agent_inbox_document(action)))
    assert not errors, [error.message for error in errors]


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(
            {
                "steer": {"text": "A"},
                "claim": {"queue": "steering", "mode": "all"},
                "observe": "x",
            },
            id="steer+claim+observe",
        ),
        pytest.param(
            {
                "follow_up": {"text": "A"},
                "claim": {"queue": "steering", "mode": "all"},
                "observe": "x",
            },
            id="follow_up+claim+observe",
        ),
        pytest.param(
            {"clear": {"queue": "all"}, "has_queued_messages": {}, "observe": "x"},
            id="clear+has_queued_messages+observe",
        ),
        pytest.param(
            {"steer": {"text": "A"}, "claim": {"queue": "steering", "mode": "all"}},
            id="steer+claim-no-observe",
        ),
    ],
)
def test_agent_inbox_action_rejects_a_second_operation_alongside_claim_or_pending(
    action: dict[str, Any],
) -> None:
    """`L07-R004` (third independent Rust review): a `oneOf` keyed only on `required`
    presence does not by itself forbid a second operation key -- `{"steer": ...,
    "claim": ..., "observe": "x"}` matched the `claim` branch alone (that branch placed
    no restriction on any other key), so the uninspected `steer` key rode along
    unrejected even though exclusivity and the `observe` constraint were each
    individually enforced. Every branch must name every OTHER operation key in its own
    `not`, not just check its own required key and (for four of the six) `observe`."""
    schema = json.loads(AGENT_INBOX_SCHEMA.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(_agent_inbox_document(action)))
    assert errors, f"expected this action to be rejected: {action}"
