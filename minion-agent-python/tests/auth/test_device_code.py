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
    """A monotonic-shaped fake clock, advanced only by the fake `sleep` it is paired with.

    Rounds `elapsed` to nanosecond precision after every increment (`L11-R021`): a REAL monotonic
    clock is read directly, with no accumulated summation error -- but this fake one advances by
    repeatedly summing whatever durations `abortable_sleep`'s own signal-polling slicing loop
    requests, and IEEE-754 float addition does not sum many small increments (e.g. one hundred
    `0.05`s) back to an exact whole number. That drift is an artifact of THIS test double, never of
    production code or of Pi's own real timing -- production's own scheduling arithmetic
    (`_floor_to_whole_milliseconds`) must stay exactly tolerance-free for every input, including a
    genuine caller-supplied value that happens to reach it through a deadline computation
    (`L11-R021`'s own finding), so any drift correction belongs here, in the fake clock the drift
    actually comes from, not in that shared production arithmetic."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def now(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.elapsed = round(self.elapsed + seconds, 9)


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


async def test_wait_before_first_poll_exact_expiry_remainder_truncates_like_node() -> None:
    """`L11-R021`: with a SHORT `expires_in_seconds` (`0.0019999995`s = 1.9999995 ms) and
    `wait_before_first_poll=True`, `deadline - now()` on this FIRST computation -- before any sleep
    has happened, so no clock-arithmetic drift exists yet -- is EXACTLY the caller's own supplied
    expiry value: a genuine public input, not an internally-drifted one. Pinned Pi's real
    `setTimeout` (`Math.trunc`) truncates it DOWN to `1` ms; two earlier revisions each applied a
    tolerance to this same `remaining` computation that instead rounded it UP to `2` ms, silently
    changing a real caller-supplied expiry. This must schedule exactly `0.001` seconds, not
    `0.002`."""
    poll = ScriptedPoll([DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(
        poll,
        expires_in_seconds=0.0019999995,
        wait_before_first_poll=True,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert result == "token"
    assert clock.elapsed == pytest.approx(0.001)
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


async def test_initial_interval_nan_used_for_a_pending_response_clamps_to_pi_magnitude() -> None:
    """`L11-R014` (§11.8 convergence, revision 2): once a first `pending` response means the
    initial interval is actually USED to schedule a sleep (not merely set up and discarded by an
    immediate `complete`), a non-finite interval must not prevent progress to the next poll --
    AND must clamp to Pi's own real observable magnitude, not an independently-chosen, three-
    orders-of-magnitude-larger value. Pi hands a non-finite delay straight to Node's own
    `setTimeout`, which clamps any out-of-range delay to ONE MILLISECOND (`Number.isFinite`-free
    numeric bounds check); a revision that instead used `MINIMUM_INTERVAL_SECONDS` (one full
    second) was an unapproved observable departure from that real magnitude, corrected here."""
    poll = ScriptedPoll([DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(
        poll, interval_seconds=float("nan"), sleep=clock.sleep, now=clock.now
    )

    assert result == "token"
    assert poll.call_count == 2
    assert clock.elapsed == pytest.approx(0.001)


async def test_initial_interval_infinity_used_for_a_pending_response_clamps_to_pi_magnitude() -> (
    None
):
    """`L11-R014` (§11.8 convergence, revision 2): the same Pi-magnitude clamp for `Infinity` -- an
    earlier revision let this loop forever slicing an ever-infinite remaining duration, never
    reaching the second poll at all; a later revision fixed the hang but clamped to a value three
    orders of magnitude larger than Pi's own real one-millisecond host-timer clamp."""
    poll = ScriptedPoll([DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(
        poll, interval_seconds=float("inf"), sleep=clock.sleep, now=clock.now
    )

    assert result == "token"
    assert poll.call_count == 2
    assert clock.elapsed == pytest.approx(0.001)


async def test_negative_infinity_used_for_a_pending_response_clamps_to_the_rfc_floor() -> None:
    """`L11-R016`: pinned Pi's own `Math.max(MINIMUM_INTERVAL_MS, Math.floor(-Infinity * 1000))`
    resolves ORDINARILY to `MINIMUM_INTERVAL_MS` (one second) -- negative `Infinity` is a valid,
    comparable number that simply LOSES every `Math.max` comparison against a finite value, so it
    NEVER reaches `setTimeout` as an "invalid delay" the way `NaN`/POSITIVE `Infinity` do. A
    candidate that treats every non-finite value alike (clamping negative `Infinity` to the SAME
    one-millisecond `NON_FINITE_INTERVAL_FALLBACK_SECONDS` as `NaN`/positive `Infinity`) diverges
    from Pi by three orders of magnitude in the OPPOSITE direction from `L11-R014`'s own original
    mistake -- Pi actually waits the FULL one-second RFC-8628 floor here, not one millisecond."""
    poll = ScriptedPoll([DevicePollPending(), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(
        poll, interval_seconds=float("-inf"), sleep=clock.sleep, now=clock.now
    )

    assert result == "token"
    assert poll.call_count == 2
    assert clock.elapsed == pytest.approx(1.0)


async def test_nan_initial_interval_stays_non_finite_through_a_slow_down_fallback_increment() -> (
    None
):
    """`L11-R014` (§11.8 convergence, revision 2): a `slow_down` response with no server-provided
    interval increments the CURRENT interval by the fixed fallback amount
    (`interval + SLOW_DOWN_INCREMENT_SECONDS`) -- if that current interval is still an unclamped
    `NaN` (because the caller's own initial interval was `NaN` and no sleep has happened yet), the
    incremented result (`NaN + 5.0 == NaN`) must ALSO stay non-finite rather than being silently
    neutralized by `max()`'s own order-dependent `NaN` comparison, so the SAME `abortable_sleep`
    clamp point still gets a chance to normalize it deterministically."""
    poll = ScriptedPoll([DevicePollSlowDown(), DevicePollComplete("token")])
    clock = FakeClock()

    result = await poll_device_code_flow(
        poll, interval_seconds=float("nan"), sleep=clock.sleep, now=clock.now
    )

    assert result == "token"
    assert poll.call_count == 2
    assert clock.elapsed == pytest.approx(0.001)


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


async def test_abortable_sleep_direct_call_with_negative_infinity_clamps_to_pi_magnitude() -> None:
    """`L11-R017`: pinned Pi's own EXPORTED `abortableSleep` has NO preceding normalization of its
    own -- it hands `ms` straight to `setTimeout`, which clamps ANY invalid delay (including
    NEGATIVE `Infinity`, not just `NaN`/positive `Infinity`) to Node's documented one-millisecond
    floor. This is DIFFERENT from `test_negative_infinity_used_for_a_pending_response_clamps_to_
    the_rfc_floor` above, which calls negative `Infinity` THROUGH `poll_device_code_flow`'s own
    upstream `Math.max`-analog normalization -- that normalization resolves negative `Infinity` to
    the ordinary one-second RFC floor BEFORE it would ever reach `abortable_sleep`. Called
    directly, with no such normalization in front of it, negative `Infinity` must clamp to the
    SAME one-millisecond magnitude as `NaN`/positive `Infinity`, not perform zero sleep at all (the
    pre-fix bug: `while remaining > 0` with `remaining=-inf` is immediately `False`, so no sleep
    call happens)."""
    total = 0.0
    calls = 0

    async def track_sleep(seconds: float) -> None:
        nonlocal total, calls
        total += seconds
        calls += 1

    await abortable_sleep(float("-inf"), None, sleep=track_sleep, poll_interval_seconds=0.1)

    assert calls >= 1
    assert total == pytest.approx(0.001)


async def test_abortable_sleep_direct_call_with_zero_clamps_to_pi_magnitude() -> None:
    """`L11-R017` (explicitly requested characterization): a raw ZERO delay passed directly to
    `abortable_sleep` is also outside Node's documented `setTimeout` valid range (`[1,
    2147483647]` milliseconds) and must clamp to the same one-millisecond magnitude, not complete
    with zero sleep calls."""
    total = 0.0
    calls = 0

    async def track_sleep(seconds: float) -> None:
        nonlocal total, calls
        total += seconds
        calls += 1

    await abortable_sleep(0.0, None, sleep=track_sleep, poll_interval_seconds=0.1)

    assert calls >= 1
    assert total == pytest.approx(0.001)


async def test_abortable_sleep_direct_call_with_negative_finite_clamps_to_pi_magnitude() -> None:
    """`L11-R017` (explicitly requested characterization): a raw negative FINITE delay passed
    directly to `abortable_sleep` is likewise outside Node's valid `setTimeout` range and must
    clamp to the same one-millisecond magnitude, not complete with zero sleep calls."""
    total = 0.0
    calls = 0

    async def track_sleep(seconds: float) -> None:
        nonlocal total, calls
        total += seconds
        calls += 1

    await abortable_sleep(-5.0, None, sleep=track_sleep, poll_interval_seconds=0.1)

    assert calls >= 1
    assert total == pytest.approx(0.001)


async def test_abortable_sleep_direct_call_with_fractional_millisecond_truncates_like_node() -> (
    None
):
    """`L11-R019`: `0.0019` seconds (1.9 ms) is a VALID delay -- neither `NaN`/`Infinity` nor
    outside `[1, 2147483647]` ms, so `_needs_setimeout_clamp` does not touch it -- but Node's real
    `setTimeout` internally truncates ANY accepted delay to a whole integer millisecond count
    before scheduling it, independently of that invalid-range clamp. `abortable_sleep` must
    therefore schedule exactly `0.001` seconds (1 ms), not the raw `0.0019`."""
    total = 0.0
    calls = 0

    async def track_sleep(seconds: float) -> None:
        nonlocal total, calls
        total += seconds
        calls += 1

    await abortable_sleep(0.0019, None, sleep=track_sleep, poll_interval_seconds=0.1)

    assert calls >= 1
    assert total == pytest.approx(0.001)


async def test_abortable_sleep_direct_call_near_millisecond_boundary_truncates_exactly() -> None:
    """`L11-R020`: `1.9999995` ms is a GENUINE public delay, deliberately chosen a hair below the
    `2` ms boundary -- Pi's real `setTimeout` (`Math.trunc`, never rounds) truncates it DOWN to
    `1` ms, exactly like any other value in `[1, 2)` ms. An earlier revision's epsilon-tolerant
    `_floor_to_whole_milliseconds` incorrectly rounded this UP to `2` ms, conflating a genuine
    near-boundary public input with the unrelated internal floating-point drift that only
    `poll_device_code_flow`'s own deadline remainder can exhibit. `abortable_sleep`'s own
    truncation must stay EXACT for a direct-call delay: this witness pins that permanently."""
    total = 0.0
    calls = 0

    async def track_sleep(seconds: float) -> None:
        nonlocal total, calls
        total += seconds
        calls += 1

    await abortable_sleep(0.0019999995, None, sleep=track_sleep, poll_interval_seconds=0.1)

    assert calls >= 1
    assert total == pytest.approx(0.001)


async def _noop() -> None:
    pass
