"""`RunSignal`: a poll-based, cooperative cancellation flag (Layer 09)."""

from minion_agent.runtime import RunSignal


def test_a_fresh_signal_is_not_aborted() -> None:
    assert RunSignal().aborted is False


def test_abort_sets_aborted() -> None:
    signal = RunSignal()
    signal.abort()
    assert signal.aborted is True


def test_abort_is_idempotent() -> None:
    """Matches pinned Pi's own `AbortController.abort()`: no "already aborted" error state."""
    signal = RunSignal()
    signal.abort()
    signal.abort()
    assert signal.aborted is True


def test_two_signals_are_independent() -> None:
    """A NEW signal per run (matching pinned Pi's own `new AbortController()` per run) -- one
    run's abort must never leak into another's."""
    a = RunSignal()
    b = RunSignal()
    a.abort()
    assert a.aborted is True
    assert b.aborted is False
