"""Request state is reconstructable from content-addressed components."""

from minion_agent.session.artifacts import ArtifactStore
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog
from minion_agent.session.request_header import (
    assemble_system,
    reconstruct_header,
    record_header,
)

BOOTSTRAP = "a very large stable bootstrap block " * 100


def test_recording_a_header_logs_references_not_content() -> None:
    log, store = SessionLog("s1"), ArtifactStore()

    event = record_header(log, store, {"system_base": BOOTSTRAP}, model="mock-1")

    assert event.kind is EventKind.REQUEST_HEADER
    assert event.data["components"]["system_base"].startswith("sha256:")
    assert BOOTSTRAP not in str(event.data)


def test_a_stable_component_is_stored_once_across_many_steps() -> None:
    """The motivating case: a 15k block must not be re-snapshotted per step."""
    log, store = SessionLog("s1"), ArtifactStore()

    for step in range(10):
        record_header(
            log,
            store,
            {"system_base": BOOTSTRAP, "memory": f"recall for step {step}"},
            model="mock-1",
        )

    # One bootstrap plus ten distinct memory blocks, plus one shared
    # no-tools marker (every header records a tools reference, even when
    # `tools` is the default empty tuple -- see Task 14).
    assert len(store) == 12
    assert len(log) == 10


def test_a_header_reconstructs_exactly() -> None:
    log, store = SessionLog("s1"), ArtifactStore()
    components = {"system_base": BOOTSTRAP, "memory": "recalled"}

    event = record_header(log, store, components, model="mock-1")

    assert reconstruct_header(event, store) == components


def test_reconstruction_matches_what_was_dispatched() -> None:
    """The property the invariant checks: the model saw what the log says."""
    log, store = SessionLog("s1"), ArtifactStore()
    components = {"system_base": "you are helpful", "memory": "user likes tea"}
    dispatched = assemble_system(components)

    event = record_header(log, store, components, model="mock-1")

    assert assemble_system(reconstruct_header(event, store)) == dispatched


def test_component_order_is_stable_regardless_of_insertion_order() -> None:
    """Byte-for-byte agreement requires a canonical order, not dict order."""
    one = assemble_system({"memory": "m", "system_base": "s"})
    two = assemble_system({"system_base": "s", "memory": "m"})

    assert one == two


def test_the_model_is_recorded_alongside_the_components() -> None:
    log, store = SessionLog("s1"), ArtifactStore()

    event = record_header(log, store, {"system_base": "s"}, model="mock-1")

    assert event.data["model"] == "mock-1"


def test_an_empty_composition_assembles_to_nothing() -> None:
    assert assemble_system({}) == ""
