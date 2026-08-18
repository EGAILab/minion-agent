"""Eligibility inherits down the scope chain; composition is the caller's."""

from minion_agent.runtime.scope import ScopeKey
from minion_agent.runtime.scoped_registry import ScopedRegistry

DEFINITION = ScopeKey("definition")
INSTANCE = ScopeKey("instance", parent=DEFINITION)
TURN = ScopeKey("turn", parent=INSTANCE)
OTHER = ScopeKey("other", parent=DEFINITION)


def test_untagged_entries_are_visible_everywhere() -> None:
    registry: ScopedRegistry[str] = ScopedRegistry()
    registry.add(None, "global", "g")

    assert registry.visible_from(TURN) == [("global", "g")]
    assert registry.visible_from(None) == [("global", "g")]


def test_descendant_sees_ancestor_entries_nearest_first() -> None:
    registry: ScopedRegistry[str] = ScopedRegistry()
    registry.add(None, "u", "untagged")
    registry.add(DEFINITION, "d", "definition")
    registry.add(INSTANCE, "i", "instance")
    registry.add(TURN, "t", "turn")

    assert registry.visible_from(TURN) == [
        ("t", "turn"),
        ("i", "instance"),
        ("d", "definition"),
        ("u", "untagged"),
    ]


def test_ancestor_never_sees_descendant_entries() -> None:
    registry: ScopedRegistry[str] = ScopedRegistry()
    registry.add(TURN, "t", "turn")

    assert registry.visible_from(DEFINITION) == []
    assert registry.visible_from(None) == []


def test_siblings_are_invisible_to_each_other() -> None:
    registry: ScopedRegistry[str] = ScopedRegistry()
    registry.add(INSTANCE, "i", "instance")
    registry.add(OTHER, "o", "other")

    assert registry.visible_from(INSTANCE) == [("i", "instance")]
    assert registry.visible_from(OTHER) == [("o", "other")]


def test_same_name_at_two_depths_is_returned_twice_nearest_first() -> None:
    """The helper does not shadow. A keyed registry takes the first; an
    additive one takes both. That choice is not the runtime's to make."""
    registry: ScopedRegistry[str] = ScopedRegistry()
    registry.add(DEFINITION, "bash", "definition-bash")
    registry.add(TURN, "bash", "turn-bash")

    assert registry.visible_from(TURN) == [
        ("bash", "turn-bash"),
        ("bash", "definition-bash"),
    ]


def test_remove_handle_withdraws_one_entry() -> None:
    registry: ScopedRegistry[str] = ScopedRegistry()
    remove = registry.add(INSTANCE, "i", "instance")
    registry.add(INSTANCE, "j", "other")

    remove()

    assert registry.visible_from(INSTANCE) == [("j", "other")]


def test_insertion_order_is_preserved_within_one_scope() -> None:
    registry: ScopedRegistry[str] = ScopedRegistry()
    registry.add(INSTANCE, "a", "first")
    registry.add(INSTANCE, "b", "second")

    assert registry.visible_from(INSTANCE) == [("a", "first"), ("b", "second")]


def test_len_counts_live_entries() -> None:
    registry: ScopedRegistry[str] = ScopedRegistry()
    registry.add(None, "a", "one")
    remove = registry.add(TURN, "b", "two")

    assert len(registry) == 2

    remove()

    assert len(registry) == 1
