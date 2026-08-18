"""Content-addressed storage: one copy per distinct content, never deleted."""

import pytest

from minion_agent.session.artifacts import ArtifactStore, MissingArtifactError


def test_put_returns_a_content_address() -> None:
    store = ArtifactStore()

    ref = store.put("hello")

    assert ref.startswith("sha256:")
    assert store.get(ref) == b"hello"


def test_identical_content_yields_one_copy() -> None:
    """The whole point: a stable 15k prompt block is stored once."""
    store = ArtifactStore()

    first = store.put("a large stable block")
    second = store.put("a large stable block")

    assert first == second
    assert len(store) == 1


def test_different_content_yields_different_references() -> None:
    store = ArtifactStore()

    assert store.put("a") != store.put("b")
    assert len(store) == 2


def test_text_and_its_utf8_bytes_are_the_same_artifact() -> None:
    store = ArtifactStore()

    assert store.put("hello") == store.put(b"hello")


def test_has_reports_membership() -> None:
    store = ArtifactStore()
    ref = store.put("hello")

    assert store.has(ref)
    assert not store.has("sha256:" + "0" * 64)


def test_missing_artifacts_raise() -> None:
    store = ArtifactStore()

    with pytest.raises(MissingArtifactError, match="sha256:"):
        store.get("sha256:" + "0" * 64)


def test_the_store_has_no_delete() -> None:
    """Artifacts holding model-visible content inherit the log's never-delete
    discipline (design spec section 5), so removal is not part of the API."""
    assert not hasattr(ArtifactStore, "delete")
    assert not hasattr(ArtifactStore, "remove")
