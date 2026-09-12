"""RFC 8628 device-code poll state machine (`PROV-010`) -- fake transport, fake clock, no real
waiting, no live network.
"""

import pytest

from minion_agent.auth.device_code import (
    DeviceFlowCancelled,
    DeviceFlowFailed,
    DeviceFlowTimedOut,
    DevicePollComplete,
    DevicePollFailed,
    DevicePollPending,
    DevicePollSlowDown,
    abortable_sleep,
    poll_device_code_flow,
)
from minion_agent.runtime.signal import RunAbortController


class FakeClock:
    """A monotonic-shaped fake clock, advanced only by the fake `sleep` it is paired with."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def now(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


class ScriptedPoll:
    """A `poll` callable driven by a fixed, ordered sequence of results."""

    def __init__(self, results: list) -> None:  # type: ignore[type-arg]
        self._results = list(results)
        self.call_count = 0

    async def __call__(self):  # type: ignore[no-untyped-def]
        self.call_count += 1
        return self._results.pop(0)


async def test_immediate_success() -> None:
    poll = ScriptedPoll([DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(poll, sleep=clock.sleep, now=clock.now)

    assert result == "token"
    assert poll.call_count == 1


async def test_pending_then_success() -> None:
    poll = ScriptedPoll([DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(poll, interval_seconds=5, sleep=clock.sleep, now=clock.now)

    assert result == "token"
    assert poll.call_count == 2
    assert clock.elapsed == pytest.approx(5.0)


async def test_pending_pending_then_success() -> None:
    poll = ScriptedPoll([DevicePollPending(), DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(poll, interval_seconds=5, sleep=clock.sleep, now=clock.now)

    assert result == "token"
    assert poll.call_count == 3
    assert clock.elapsed == pytest.approx(10.0)


async def test_slow_down_then_success_uses_server_provided_interval() -> None:
    poll = ScriptedPoll([DevicePollSlowDown(interval_seconds=9), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(poll, interval_seconds=5, sleep=clock.sleep, now=clock.now)

    assert result == "token"
    assert clock.elapsed == pytest.approx(9.0)  # the server's own interval, not 5 + 5


async def test_slow_down_without_a_server_interval_increments_by_five_seconds() -> None:
    """RFC 8628 section 3.5's own fallback increment."""
    poll = ScriptedPoll([DevicePollSlowDown(), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(poll, interval_seconds=5, sleep=clock.sleep, now=clock.now)

    assert result == "token"
    assert clock.elapsed == pytest.approx(10.0)  # 5 + SLOW_DOWN_INCREMENT_SECONDS (5)


async def test_slow_down_with_a_non_finite_server_interval_falls_back_to_the_fixed_increment() -> (
    None
):
    """`L11-R004`: Pi's own guard requires the server-provided interval to be finite AND
    positive (`Number.isFinite(...) && ... > 0`) before trusting it -- an infinite value must be
    treated exactly like an absent one (the fixed +5s increment), never scheduled as a real sleep
    duration."""
    poll = ScriptedPoll(
        [DevicePollSlowDown(interval_seconds=float("inf")), DevicePollComplete("token")]
    )
    clock = FakeClock()

    result = await poll_device_code_flow(poll, interval_seconds=2, sleep=clock.sleep, now=clock.now)

    assert result == "token"
    assert clock.elapsed == pytest.approx(7.0)  # 2 + SLOW_DOWN_INCREMENT_SECONDS (5), not inf


async def test_pending_then_slow_down_then_success() -> None:
    poll = ScriptedPoll([DevicePollPending(), DevicePollSlowDown(), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(poll, interval_seconds=5, sleep=clock.sleep, now=clock.now)

    assert result == "token"
    assert poll.call_count == 3
    # First sleep at 5s (pending), second (after slow_down) at 5+5=10s.
    assert clock.elapsed == pytest.approx(15.0)


async def test_terminal_protocol_error_stops_immediately() -> None:
    poll = ScriptedPoll([DevicePollFailed("access_denied")])
    clock = FakeClock()

    with pytest.raises(DeviceFlowFailed, match="access_denied"):
        await poll_device_code_flow(poll, sleep=clock.sleep, now=clock.now)

    assert poll.call_count == 1


async def test_expiry_without_slow_down_raises_the_plain_timeout_message() -> None:
    poll = ScriptedPoll([DevicePollPending()] * 100)
    clock = FakeClock()

    with pytest.raises(DeviceFlowTimedOut, match=r"Device flow timed out$"):
        await poll_device_code_flow(
            poll, interval_seconds=5, expires_in_seconds=12, sleep=clock.sleep, now=clock.now
        )


async def test_expiry_after_slow_down_raises_the_clock_drift_message() -> None:
    poll = ScriptedPoll([DevicePollSlowDown()] + [DevicePollPending()] * 100)
    clock = FakeClock()

    with pytest.raises(DeviceFlowTimedOut, match="clock drift"):
        await poll_device_code_flow(
            poll, interval_seconds=5, expires_in_seconds=12, sleep=clock.sleep, now=clock.now
        )


async def test_deadline_reached_between_a_poll_and_its_own_tail_sleep_breaks_immediately() -> None:
    """With a REAL clock, `poll()` itself can take enough wall-clock time to cross the deadline
    between the loop-top check and the tail-remaining check -- the loop must break out immediately
    rather than attempting a non-positive sleep, not merely rely on the next loop-top check (which
    a fake, poll()-time-free clock like `FakeClock` above can never exercise)."""
    times = iter([0.0, 0.0, 5.0])

    def now() -> float:
        return next(times)

    poll = ScriptedPoll([DevicePollPending()])

    async def sleep(_seconds: float) -> None:
        raise AssertionError("must break before attempting a non-positive sleep")

    with pytest.raises(DeviceFlowTimedOut):
        await poll_device_code_flow(
            poll, interval_seconds=5, expires_in_seconds=5, sleep=sleep, now=now
        )

    assert poll.call_count == 1


async def test_abort_before_the_first_poll_is_detected_immediately() -> None:
    controller = RunAbortController()
    controller.abort()
    poll = ScriptedPoll([DevicePollComplete("token")])
    clock = FakeClock()

    with pytest.raises(DeviceFlowCancelled):
        await poll_device_code_flow(
            poll, signal=controller.signal, sleep=clock.sleep, now=clock.now
        )

    assert poll.call_count == 0


async def test_abort_during_sleep_between_polls_stops_before_the_next_poll() -> None:
    controller = RunAbortController()

    async def aborting_sleep(seconds: float) -> None:
        clock.elapsed += seconds
        controller.abort()

    poll = ScriptedPoll([DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    with pytest.raises(DeviceFlowCancelled):
        await poll_device_code_flow(
            poll, interval_seconds=5, signal=controller.signal, sleep=aborting_sleep, now=clock.now
        )

    assert poll.call_count == 1  # the second poll never ran


async def test_abort_during_an_in_flight_poll_is_caught_at_the_next_iteration() -> None:
    """The poller does not race an in-flight `poll()` call against the signal (matching Pi's own
    behavior exactly -- neither implementation wraps the transport call itself in cancellation);
    an abort raised BY `poll()`'s own caller-provided body still lets that one attempt finish, and
    is only observed at the next loop-top check."""
    controller = RunAbortController()
    observed_after_poll_returns = False

    async def poll():  # type: ignore[no-untyped-def]
        nonlocal observed_after_poll_returns
        controller.abort()
        observed_after_poll_returns = True
        return DevicePollPending()

    clock = FakeClock()

    with pytest.raises(DeviceFlowCancelled):
        await poll_device_code_flow(
            poll, interval_seconds=5, signal=controller.signal, sleep=clock.sleep, now=clock.now
        )

    assert observed_after_poll_returns  # the in-flight poll() ran to completion


async def test_wait_before_first_poll_sleeps_once_before_the_first_attempt() -> None:
    poll = ScriptedPoll([DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(
        poll, interval_seconds=5, wait_before_first_poll=True, sleep=clock.sleep, now=clock.now
    )

    assert result == "token"
    assert clock.elapsed == pytest.approx(5.0)
    assert poll.call_count == 1


async def test_interval_is_clamped_to_the_minimum() -> None:
    poll = ScriptedPoll([DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    await poll_device_code_flow(poll, interval_seconds=0.001, sleep=clock.sleep, now=clock.now)

    assert clock.elapsed == pytest.approx(1.0)  # MINIMUM_INTERVAL_SECONDS, not the requested 0.001


async def test_default_interval_is_five_seconds_when_none_is_given() -> None:
    poll = ScriptedPoll([DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    await poll_device_code_flow(poll, sleep=clock.sleep, now=clock.now)

    assert clock.elapsed == pytest.approx(5.0)


async def test_fractional_initial_interval_is_floored_to_whole_milliseconds() -> None:
    """`L11-R012`: Pi floors the caller's own initial interval to whole milliseconds
    (`Math.floor(seconds * 1000)`) before scheduling -- `1.2349` seconds must schedule exactly
    `1.234`, not the raw fractional value (whose own float representation would otherwise drift,
    e.g. `1.2348999999999997`, an observable cross-language mismatch)."""
    poll = ScriptedPoll([DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    await poll_device_code_flow(poll, interval_seconds=1.2349, sleep=clock.sleep, now=clock.now)

    assert clock.elapsed == pytest.approx(1.234, abs=1e-9)


async def test_fractional_server_slow_down_interval_is_floored_to_whole_milliseconds() -> None:
    """`L11-R012`: the same whole-millisecond floor applies to a finite, positive server-provided
    `slow_down` interval, not only the caller's own initial one."""
    poll = ScriptedPoll([DevicePollSlowDown(interval_seconds=1.2349), DevicePollComplete("token")])
    clock = FakeClock()

    await poll_device_code_flow(poll, interval_seconds=5, sleep=clock.sleep, now=clock.now)

    assert clock.elapsed == pytest.approx(1.234, abs=1e-9)


async def test_initial_interval_infinity_does_not_throw_when_the_first_poll_completes() -> None:
    """`L11-R014`: pinned Pi's own INITIAL interval option carries no `Number.isFinite` guard
    (unlike the `slow_down` server value, `L11-R004`) -- JS's own `Math.floor`/`Math.max` never
    raise for `Infinity`, they just propagate it arithmetically. An immediately-successful first
    poll must return the completed value without ever attempting to schedule a sleep, regardless
    of how nonsensical the configured initial interval is -- a naive Python port that floors
    unconditionally at setup time would incorrectly crash before `poll` is ever even called."""
    poll = ScriptedPoll([DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(
        poll, interval_seconds=float("inf"), sleep=clock.sleep, now=clock.now
    )

    assert result == "token"
    assert poll.call_count == 1


async def test_initial_interval_nan_does_not_throw_when_the_first_poll_completes() -> None:
    """`L11-R014`: the same non-throwing setup guarantee for `NaN`, Pi's other special numeric
    value `Math.floor`/`Math.max` propagate without raising."""
    poll = ScriptedPoll([DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(
        poll, interval_seconds=float("nan"), sleep=clock.sleep, now=clock.now
    )

    assert result == "token"
    assert poll.call_count == 1


async def test_abortable_sleep_raises_immediately_if_already_aborted() -> None:
    controller = RunAbortController()
    controller.abort()

    with pytest.raises(DeviceFlowCancelled):
        await abortable_sleep(10.0, controller.signal, sleep=lambda _s: _noop())


async def test_abortable_sleep_polls_the_signal_between_short_steps() -> None:
    controller = RunAbortController()
    steps: list[float] = []

    async def counting_sleep(seconds: float) -> None:
        steps.append(seconds)
        if len(steps) == 3:
            controller.abort()

    with pytest.raises(DeviceFlowCancelled):
        await abortable_sleep(
            1.0, controller.signal, sleep=counting_sleep, poll_interval_seconds=0.1
        )

    assert len(steps) == 3
    assert all(step == pytest.approx(0.1) for step in steps)


async def test_abortable_sleep_completes_normally_without_a_signal() -> None:
    total = 0.0

    async def track_sleep(seconds: float) -> None:
        nonlocal total
        total += seconds

    await abortable_sleep(0.25, None, sleep=track_sleep, poll_interval_seconds=0.1)

    assert total == pytest.approx(0.25)


async def _noop() -> None:
    pass
