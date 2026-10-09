"""Tool schemas are request state, stored by hash like every other component."""

import json
from dataclasses import replace
from typing import Any

import pytest

from minion_agent.llm import GrammarConstrainedSampling, JsonSchemaConstrainedSampling, ToolSchema
from minion_agent.session import ArtifactStore, EventKind, SessionLog, assemble_system
from minion_agent.session.request_header import (
    reconstruct_header,
    reconstruct_tools,
    record_header,
)


def _schema(name: str = "echo") -> ToolSchema:
    return ToolSchema(
        name=name,
        description="repeat",
        parameters={"type": "object", "properties": {}},
    )


def _nested_schema(name: str) -> ToolSchema:
    """A schema with genuinely nested `parameters`, to prove the store round
    trip preserves structure and not just the top-level fields."""
    return ToolSchema(
        name=name,
        description=f"{name} description",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "enum": ["a", "b"]},
                "opts": {
                    "type": "object",
                    "properties": {"depth": {"type": "integer"}},
                },
            },
            "required": ["path"],
        },
    )


def test_a_header_without_tools_records_none() -> None:
    log, store = SessionLog("s1"), ArtifactStore()

    event = record_header(log, store, {"system_base": "be helpful"}, model="m")

    assert "tools" in event.data
    assert reconstruct_tools(event, store) == ()


def test_tool_schemas_round_trip_through_the_store() -> None:
    """Full structural equality, not just names: `ToolSchema` is a frozen
    dataclass, so `==` compares name, description, and parameters. Nested
    `parameters` (an object property, an array of allowed values, a mix of
    types) are the shape `as_json`'s recursive `_canonical` exists to
    normalize, and a round trip through the store is where a bug in that
    normalization -- or in `reconstruct_tools` -- would surface."""
    log, store = SessionLog("s1"), ArtifactStore()
    first, second = _nested_schema("echo"), _nested_schema("read")

    event = record_header(log, store, {"system_base": "s"}, model="m", tools=(first, second))

    assert reconstruct_tools(event, store) == (first, second)


def test_the_header_stores_a_reference_not_the_schemas() -> None:
    """The point of content addressing: a stable tool set costs one hash per
    step, not a re-snapshot of every schema."""
    log, store = SessionLog("s1"), ArtifactStore()

    event = record_header(log, store, {"system_base": "s"}, model="m", tools=(_schema(),))

    assert event.data["tools"].startswith("sha256:")


def test_an_unchanged_tool_set_addresses_to_the_same_reference() -> None:
    log, store = SessionLog("s1"), ArtifactStore()

    first = record_header(log, store, {"system_base": "s"}, model="m", tools=(_schema(),))
    second = record_header(log, store, {"system_base": "s"}, model="m", tools=(_schema(),))

    assert first.data["tools"] == second.data["tools"]
    # Strengthened over the brief: pin the actual content-addressing
    # behaviour, not just that two calls happen to agree. A `put` that
    # returned a constant reference for every call would pass the assertion
    # above without ever hashing anything.
    changed = record_header(
        log, store, {"system_base": "s"}, model="m", tools=(_schema(), _schema("read"))
    )
    assert changed.data["tools"] != first.data["tools"]
    assert len(store) == 3  # system_base "s" + one-tool payload + two-tool payload


def test_tools_do_not_leak_into_the_system_prompt() -> None:
    """They are request state, not prompt text. Joining them into the system
    string would change what the model reads."""
    log, store = SessionLog("s1"), ArtifactStore()

    event = record_header(log, store, {"system_base": "be helpful"}, model="m", tools=(_schema(),))

    header = reconstruct_header(event, store)
    assert assemble_system(header) == "be helpful"
    # Strengthened over the brief: `_assemble is assemble_system` is a
    # tautology about a re-export, true regardless of whether tools leak.
    # Pin the actual separation instead -- the reconstructed component
    # mapping used to build the system prompt must not contain a "tools" key
    # or any trace of the tool payload.
    assert "tools" not in header
    assert set(header) == {"system_base"}


# --- L03-D001: every model-facing field, constrained_sampling included, comes back -------------

_SAMPLING_STATES = {
    "absent": None,
    "false": False,
    "json-schema-prefer": JsonSchemaConstrainedSampling(strict="prefer"),
    "json-schema-require": JsonSchemaConstrainedSampling(strict="require"),
    "grammar-lark": GrammarConstrainedSampling(openai_lark="start: WORD"),
    "grammar-regex": GrammarConstrainedSampling(openai_regex="[a-z]+"),
    "grammar-both": GrammarConstrainedSampling(openai_lark="start: WORD", openai_regex="[a-z]+"),
    "grammar-neither": GrammarConstrainedSampling(),
}


@pytest.mark.parametrize("sampling", _SAMPLING_STATES.values(), ids=_SAMPLING_STATES.keys())
def test_every_sampling_state_round_trips_through_the_store(sampling: Any) -> None:
    """Complete equality, both as values and as the model-facing JSON: the decoder is the
    inverse of `as_json` for each of the eight certified states."""
    log, store = SessionLog("s1"), ArtifactStore()
    schema = replace(_nested_schema("echo"), constrained_sampling=sampling)

    event = record_header(log, store, {"system_base": "s"}, model="m", tools=(schema,))

    (rebuilt,) = reconstruct_tools(event, store)
    assert rebuilt == schema
    assert rebuilt.as_json() == schema.as_json()


def test_false_and_absent_stay_distinct_after_reconstruction() -> None:
    log, store = SessionLog("s1"), ArtifactStore()
    absent, disabled = _schema("a"), replace(_schema("b"), constrained_sampling=False)

    event = record_header(log, store, {"system_base": "s"}, model="m", tools=(absent, disabled))

    rebuilt = reconstruct_tools(event, store)
    assert [schema.constrained_sampling for schema in rebuilt] == [None, False]


def test_a_header_stored_without_the_field_reconstructs_it_as_absent() -> None:
    """Headers recorded before Layer 05 added `constrained_sampling` (de2a977..d9054fe) store
    tool entries with no such member; they stay readable."""
    log, store = SessionLog("s1"), ArtifactStore()
    payload = json.dumps(
        [
            {
                "name": "echo",
                "description": "repeat",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        sort_keys=True,
    )
    event = log.append(
        EventKind.REQUEST_HEADER, {"model": "m", "components": {}, "tools": store.put(payload)}
    )

    assert reconstruct_tools(event, store) == (_schema(),)


_MALFORMED = {
    "true": True,
    "string": "json_schema",
    "zero": 0,
    "list": [],
    "unknown-type": {"type": "regex"},
    "json-schema-missing-strict": {"type": "json_schema"},
    "json-schema-bad-strict": {"type": "json_schema", "strict": "maybe"},
    "json-schema-extra-member": {"type": "json_schema", "strict": "prefer", "extra": 1},
    "grammar-missing-variants": {"type": "grammar"},
    "grammar-variants-not-object": {"type": "grammar", "variants": []},
    "grammar-unknown-format": {"type": "grammar", "variants": {"openai_cfg": "x"}},
    "grammar-non-string-format": {"type": "grammar", "variants": {"openai_lark": 1}},
    "grammar-extra-member": {"type": "grammar", "variants": {}, "extra": 1},
}


@pytest.mark.parametrize("value", _MALFORMED.values(), ids=_MALFORMED.keys())
def test_a_stored_value_outside_the_four_states_fails_reconstruction(value: Any) -> None:
    """Never read back as absent: that would be the silent loss L03-D001 removes."""
    log, store = SessionLog("s1"), ArtifactStore()
    payload = json.dumps([{**_schema().as_json(), "constrained_sampling": value}], sort_keys=True)
    event = log.append(
        EventKind.REQUEST_HEADER, {"model": "m", "components": {}, "tools": store.put(payload)}
    )

    with pytest.raises(ValueError, match="not a certified state"):
        reconstruct_tools(event, store)


def test_the_stored_form_and_its_hash_are_unchanged() -> None:
    """L03-D001 changes reading only. Pinned against the pre-change `main` (b355bf92)."""
    log, store = SessionLog("s1"), ArtifactStore()
    schema = ToolSchema(
        name="echo",
        description="repeat",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        constrained_sampling=False,
    )

    event = record_header(log, store, {"system_base": "s"}, model="m", tools=(schema,))

    assert event.data["tools"] == (
        "sha256:a874be3af7c729fcc2097f1efe0950375584a74827ee9024a45db116851555e8"
    )
    assert store.get(event.data["tools"]).decode("utf-8") == (
        '[{"constrained_sampling": false, "description": "repeat", "name": "echo", '
        '"parameters": {"properties": {"text": {"type": "string"}}, "type": "object"}}]'
    )
