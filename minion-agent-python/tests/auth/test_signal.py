"""`CombinedSignal`: the additive, auth-owned signal composition `refresh_if_expiring` uses
(`L11-R002`; Pi `AbortSignal.any([signal, AbortSignal.timeout(ms)])`)."""

from minion_agent.auth.signal import CombinedSignal
from minion_agent.runtime.signal import RunAbortController


def test_not_aborted_before_the_budget_elapses_with_no_caller_signal() -> None:
    times = iter([0.0, 5.0])
    signal = CombinedSignal(None, 15.0, now=lambda: next(times))  # deadline set at t=0 -> 15.0
    assert signal.aborted is False  # checked at t=5.0


def test_aborted_once_the_budget_elapses_with_no_caller_signal() -> None:
    times = iter([0.0, 15.0])
    signal = CombinedSignal(None, 15.0, now=lambda: next(times))
    assert signal.aborted is True  # checked at t=15.0, exactly at the deadline


def test_aborted_immediately_when_the_caller_signal_is_already_aborted() -> None:
    controller = RunAbortController()
    controller.abort()
    signal = CombinedSignal(controller.signal, 15.0, now=lambda: 0.0)
    assert signal.aborted is True


def test_not_aborted_when_neither_the_caller_signal_nor_the_budget_has_fired() -> None:
    controller = RunAbortController()
    signal = CombinedSignal(controller.signal, 15.0, now=lambda: 1.0)
    assert signal.aborted is False


def test_aborted_when_the_caller_signal_fires_before_the_budget_elapses() -> None:
    controller = RunAbortController()
    signal = CombinedSignal(controller.signal, 15.0, now=lambda: 1.0)
    controller.abort()
    assert signal.aborted is True  # an OR: either side firing aborts the combination
