"""`RunAbortController`/`RunSignal`: a poll-based, cooperative cancellation flag (Layer 09),
split into a private mutator and a read-only view (`L09-R004`)."""

import pytest

from minion_agent.runtime import RunAbortController, RunSignal


def test_a_fresh_controller_is_not_aborted() -> None:
    assert RunAbortController().signal.aborted is False


def test_abort_sets_the_signal_aborted() -> None:
    controller = RunAbortController()
    controller.abort()
    assert controller.signal.aborted is True


def test_abort_is_idempotent() -> None:
    """Matches pinned Pi's own `AbortController.abort()`: no "already aborted" error state."""
    controller = RunAbortController()
    controller.abort()
    controller.abort()
    assert controller.signal.aborted is True


def test_two_controllers_are_independent() -> None:
    """A NEW controller per run (matching pinned Pi's own `new AbortController()` per run) --
    one run's abort must never leak into another's."""
    a = RunAbortController()
    b = RunAbortController()
    a.abort()
    assert a.signal.aborted is True
    assert b.signal.aborted is False


def test_the_signal_is_the_same_object_across_accesses() -> None:
    """`L09-R004`'s own "stable per-run identity" requirement: `.signal` is created once per
    controller, not per access."""
    controller = RunAbortController()
    assert controller.signal is controller.signal


def test_the_read_only_signal_has_no_abort_method() -> None:
    """`L09-R004`: a consumer holding only the `RunSignal` view cannot itself trigger
    cancellation -- matching pinned Pi's own `AbortSignal`, which has no `.abort()` at all."""
    controller = RunAbortController()
    assert not hasattr(controller.signal, "abort")


def test_run_signal_cannot_be_constructed_without_a_controller() -> None:
    """`RunSignal` is never constructed directly by application code -- only obtained via
    `RunAbortController.signal`/`AgentInstance.signal`."""
    with pytest.raises(TypeError):
        RunSignal()  # type: ignore[call-arg]
