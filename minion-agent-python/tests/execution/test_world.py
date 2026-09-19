"""`EXEC-006`: execution-world identity, `compatible()` (equality-only, `L12-R013`), and
`validate()`'s ordered `ExecutionWorldError` payload (`L12-R019`)."""

from minion_agent.execution.result import Err, Ok
from minion_agent.execution.world import ExecutionWorldIdentity, compatible, validate


def test_equal_identities_are_compatible() -> None:
    a = ExecutionWorldIdentity("local")
    b = ExecutionWorldIdentity("local")
    assert compatible(a, b) is True


def test_unequal_identities_are_incompatible() -> None:
    a = ExecutionWorldIdentity("local")
    b = ExecutionWorldIdentity("remote")
    assert compatible(a, b) is False


def test_compatibility_is_symmetric() -> None:
    """`L12-R013`: equality-only is inherently symmetric, by construction."""
    a = ExecutionWorldIdentity("x")
    b = ExecutionWorldIdentity("y")
    assert compatible(a, b) == compatible(b, a)


def test_validate_ok_when_all_equal() -> None:
    identity = ExecutionWorldIdentity("local")
    result = validate([("fs", identity), ("shell", identity)])
    assert result == Ok(None)


def test_validate_err_names_both_providers() -> None:
    a = ExecutionWorldIdentity("a")
    b = ExecutionWorldIdentity("b")
    result = validate([("fs", a), ("shell", b)])
    assert isinstance(result, Err)
    assert len(result.error.incompatible_pairs) == 1
    pair = result.error.incompatible_pairs[0]
    assert pair.left == "fs"
    assert pair.right == "shell"


def test_validate_pairs_are_ordered_by_input_index() -> None:
    """`L12-R019`: `EXECUTIONWORLDERROR HAS A CONCRETE, ORDERED PAYLOAD` witness -- fs/shell and
    fs/subprocess incompatible, shell/subprocess compatible, in exact input-index `i < j` order."""
    a = ExecutionWorldIdentity("a")
    b = ExecutionWorldIdentity("b")
    result = validate([("fs", a), ("shell", b), ("subprocess", b)])
    assert isinstance(result, Err)
    pairs = result.error.incompatible_pairs
    assert len(pairs) == 2
    assert (pairs[0].left, pairs[0].right) == ("fs", "shell")
    assert (pairs[1].left, pairs[1].right) == ("fs", "subprocess")


def test_validate_order_independence() -> None:
    """`EXECUTION-WORLD COMPATIBILITY IS SYMMETRIC AND ORDER-INDEPENDENT` witness: both orderings
    of two unequal identities produce equivalent (both-incompatible) outcomes."""
    a = ExecutionWorldIdentity("a")
    b = ExecutionWorldIdentity("b")
    forward = validate([("a", a), ("b", b)])
    backward = validate([("b", b), ("a", a)])
    assert isinstance(forward, Err)
    assert isinstance(backward, Err)
    assert {p.left for p in forward.error.incompatible_pairs} | {
        p.right for p in forward.error.incompatible_pairs
    } == {"a", "b"}
    assert {p.left for p in backward.error.incompatible_pairs} | {
        p.right for p in backward.error.incompatible_pairs
    } == {"a", "b"}


def test_validate_with_zero_or_one_provider_is_always_ok() -> None:
    assert validate([]) == Ok(None)
    assert validate([("fs", ExecutionWorldIdentity("x"))]) == Ok(None)


def test_providers_not_passed_are_never_implicated() -> None:
    """Mixed worlds remain legal until a same-resource consumer validates them together -- a
    provider never passed to this `validate()` call (however incompatible it might be with one
    that WAS passed) cannot affect the outcome."""
    result = validate([("fs", ExecutionWorldIdentity("a"))])
    assert result == Ok(None)
