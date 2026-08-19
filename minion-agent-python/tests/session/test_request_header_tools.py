"""Tool schemas are request state, stored by hash like every other component."""

from minion_agent.llm import ToolSchema
from minion_agent.session import ArtifactStore, SessionLog, assemble_system
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
