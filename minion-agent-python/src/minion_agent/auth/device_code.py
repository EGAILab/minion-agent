"""RFC 8628 device-authorization polling, provider-neutral and transport/timing-injectable
(`PROV-010`; Pi `pollOAuthDeviceCodeFlow`/`abortableSleep`, `auth/oauth/device-code.ts`).

This module owns the ENTIRE interval/backoff/deadline/cancellation state machine. A caller's own
`poll` callable is the only transport seam: it reports what ONE attempt found, nothing more --
never a retry decision, never a sleep, never a deadline check. Keeping those semantics here (not
duplicated into a canonical runner) is what makes the canonical evidence for this module a genuine
cross-language proof rather than a runner-side reimplementation.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..runtime.signal import RunSignal

MINIMUM_INTERVAL_SECONDS = 1.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
"""RFC 8628 section 3.2: if the authorization server omits `interval`, the client must use 5
seconds."""
SLOW_DOWN_INCREMENT_SECONDS = 5.0
"""RFC 8628 section 3.5: a `slow_down` response means the polling interval must increase by 5
seconds, UNLESS the server's own response names a new interval directly (Pi's own comment,
`device-code.ts`: trusting only a client-tracked increment risks polling early forever under
WSL/VM clock drift, so a server-provided interval is preferred when present)."""

NON_FINITE_INTERVAL_FALLBACK_SECONDS = 0.001
"""`L11-R014` (§11.8 convergence, revision 2): the delay `NaN` or POSITIVE `Infinity` clamps to
when actually USED to schedule a sleep. NOT `MINIMUM_INTERVAL_SECONDS` -- that constant is RFC
8628's own "never poll faster than this" floor for ORDINARY, finite intervals, a different concept
entirely. This value instead matches pinned Pi's own OBSERVABLE magnitude as closely as Python
reasonably can: Pi hands an invalid delay straight through to the host `setTimeout`, and Node's own
documented contract ("If delay is larger than 2147483647 or less than 1, the delay will be set to
1") clamps ANY out-of-range delay -- `NaN` fails both bounds (every comparison against `NaN` is
`false`), positive `Infinity` fails the upper bound -- to exactly ONE MILLISECOND, independently
confirmed live against a real Node process by an earlier review. A revision of this constant
previously chose `MINIMUM_INTERVAL_SECONDS` (one full second) instead, three orders of magnitude
larger than Pi's own real value, which is an unapproved observable departure this revision
corrects -- not merely "good enough progress," but the actual pinned magnitude, disclosed as
approximate only in that Python's own scheduler cannot guarantee sub-millisecond precision, the
same kind of immaterial mechanism difference `abortable_sleep`'s own 50 ms signal-polling slice
already discloses elsewhere in this module.

Deliberately EXCLUDES negative `Infinity` (`L11-R016`): pinned Pi's own `Math.max(MINIMUM_INTERVAL_
MS, Math.floor(-Infinity * 1000))` resolves ORDINARILY to `MINIMUM_INTERVAL_MS` -- negative
`Infinity` is a valid, comparable number that simply LOSES every `Math.max` comparison against a
finite value, so it never reaches `setTimeout` as an "invalid delay" at all; the host-timer clamp
this constant models is never even consulted for it. See `_needs_host_timer_clamp`'s own docstring
for the exact predicate that keeps negative `Infinity` on the ordinary `max()` path instead."""


def _needs_host_timer_clamp(seconds: float) -> bool:
    """True only for `NaN` or POSITIVE `Infinity` (`L11-R016`) -- the two values that fail Node's
    own `setTimeout` numeric bounds check and get clamped to its documented one-millisecond
    minimum (`NON_FINITE_INTERVAL_FALLBACK_SECONDS`'s own docstring). Negative `Infinity` is
    EXPLICITLY EXCLUDED: it is a valid, comparable number Python's own ordinary `max()` already
    resolves CORRECTLY against `MINIMUM_INTERVAL_SECONDS` (unlike `NaN`, which Python's `max()`
    mishandles via its own order-dependent comparison, `L11-R014`/`L11-R015`) -- routing negative
    `Infinity` through this predicate's own `True` branch would incorrectly apply the host-timer
    clamp to a value that never actually needs it, an earlier revision's own exact mistake (`L11-
    R016`: pinned Pi's `Math.max(1000, -Infinity)` already yields plain `1000`, never reaching
    `setTimeout` as an invalid delay at all)."""
    return math.isnan(seconds) or seconds == math.inf


CANCEL_MESSAGE = "Login cancelled"
TIMEOUT_MESSAGE = "Device flow timed out"
SLOW_DOWN_TIMEOUT_MESSAGE = (
    "Device flow timed out after one or more slow_down responses. This is often caused by clock "
    "drift in WSL or VM environments. Please sync or restart the VM clock and try again."
)


def _floor_to_whole_milliseconds(seconds: float) -> float:
    """Pi floors a caller/server-provided interval to whole MILLISECONDS before scheduling it
    (`L11-R012`; Pi `Math.floor(seconds * 1000)`, both for the caller's own initial interval and a
    finite/positive server-provided `slow_down` interval) -- a fractional-second interval (e.g.
    `1.2349`) must schedule exactly `1.234`, not the raw fractional value. This module's own public
    API stays seconds-based; this helper is the one place the millisecond floor is applied.

    Non-finite input passes through UNCHANGED (`L11-R014`, setup half): pinned Pi's own INITIAL
    interval option (unlike the `slow_down` server value, `PROV-004`) carries no `Number.isFinite`
    guard at all, and JS's own `Math.floor`/`Math.max` never raise for `Infinity`/`NaN` -- they
    simply propagate the special value arithmetically (`Math.floor(Infinity) === Infinity`,
    `Math.floor(NaN) === NaN`). Python's `math.floor` raises `OverflowError`/`ValueError` for
    exactly these inputs, which a naive port would incorrectly turn into a setup-time crash even
    when the flow never ends up sleeping at all -- e.g. an immediately-successful first poll
    returns before this interval is ever used. This early-return is what makes this helper a
    faithful, non-throwing port of Pi's own PURE-ARITHMETIC layer, deliberately mirroring it
    exactly rather than clamping here -- clamping a non-finite value that IS actually used to
    schedule a sleep is a SEPARATE concern, handled where the sleep is actually scheduled
    (`abortable_sleep`'s own docstring), matching Pi's own separate host-timer clamping boundary,
    not this pure-math one."""
    if not math.isfinite(seconds):
        return seconds
    return math.floor(seconds * 1000) / 1000


@dataclass(frozen=True, slots=True)
class DevicePollPending:
    """The server has not yet authorized the device. Keep polling at the current interval."""


@dataclass(frozen=True, slots=True)
class DevicePollSlowDown:
    """RFC 8628 section 3.5: increase the polling interval before the next attempt.
    `interval_seconds`, when the server provides one, is preferred over the fixed increment --
    but only when it is finite and positive (`L11-R004`; Pi `device-code.ts`'s own
    `Number.isFinite(result.intervalSeconds) && result.intervalSeconds > 0` guard). A non-finite
    value (e.g. `float("inf")`) is treated exactly like an absent one: the fixed +5s increment
    applies instead, never scheduling a non-finite or non-positive sleep. A finite, positive value
    is FLOORED to whole milliseconds before scheduling (`L11-R012`; see
    `_floor_to_whole_milliseconds`'s own docstring), the same as the caller's own initial
    interval."""

    interval_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class DevicePollFailed:
    """A terminal protocol error. Polling stops immediately; this is not a retryable outcome."""

    message: str


@dataclass(frozen=True, slots=True)
class DevicePollComplete[T]:
    """The device is authorized. `value` is `poll`'s own successful result, returned unchanged by
    `poll_device_code_flow`."""

    value: T


type DevicePollResult[T] = (
    DevicePollPending | DevicePollSlowDown | DevicePollFailed | DevicePollComplete[T]
)
"""Every outcome one poll attempt may report (Pi `OAuthDeviceCodePollResult`)."""


class DeviceFlowError(Exception):
    """Base for `poll_device_code_flow`'s own terminal (non-success) outcomes."""


class DeviceFlowCancelled(DeviceFlowError):
    """The caller's own `RunSignal` was observed aborted -- either already aborted when checked,
    or aborted during a sleep between attempts."""


class DeviceFlowTimedOut(DeviceFlowError):
    """The deadline passed with no successful poll. The message distinguishes whether one or more
    `slow_down` responses were seen (Pi's own two-message split, `device-code.ts`), since that
    combination is a specific, actionable symptom (clock drift), not a generic timeout."""


class DeviceFlowFailed(DeviceFlowError):
    """`poll` itself reported a terminal protocol error (`DevicePollFailed`)."""


async def abortable_sleep(
    seconds: float,
    signal: RunSignal | None,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    poll_interval_seconds: float = 0.05,
) -> None:
    """Sleep for `seconds`, checking `signal` at `poll_interval_seconds` boundaries.

    `RunSignal` is poll-based BY CERTIFIED DESIGN (Layer 09) -- it has no push/event mechanism the
    way Pi's own `AbortSignal.addEventListener("abort", ...)` does, and this module does not add
    one. Slicing a long sleep into short steps and checking `signal.aborted` between them is how a
    poll-only signal gets checked "promptly enough" without redesigning `RunSignal` itself; it is
    a deliberate, disclosed difference from Pi's own instant-interrupt behavior, immaterial to any
    already-certified Layer-09 semantic.

    Raises `DeviceFlowCancelled` immediately if `signal` is already aborted, or as soon as a
    step boundary observes it. `sleep` is injectable so tests never wait in real time.

    A non-finite `seconds` (`NaN`/`Infinity`) is clamped to
    `NON_FINITE_INTERVAL_FALLBACK_SECONDS` (`L11-R014`, resolved by §11.8 convergence agreement
    revision 2, `11-auth-foundation-r014-convergence-agreement-v2.md`): this is the Python analog
    of Pi's own HOST-timer boundary (`setTimeout` clamps an out-of-range delay -- including
    `NaN`/`Infinity`, both of which fail its own numeric bounds check -- to its own documented
    one-millisecond minimum, per Node's own `setTimeout` contract), not something derivable from
    the pure `Math.floor`/`Math.max` arithmetic upstream (which never throws and faithfully
    propagates these special values, matching `_floor_to_whole_milliseconds`'s own pass-through
    contract for the NOT-yet-scheduled case). See `NON_FINITE_INTERVAL_FALLBACK_SECONDS`'s own
    docstring for why this is Pi's actual observed magnitude, not an independently-chosen
    "good enough" value -- a first convergence revision used `MINIMUM_INTERVAL_SECONDS` (one full
    second) instead, which an independent review correctly identified as an unapproved, three-
    orders-of-magnitude-larger observable departure from Pi's own real behavior. Negative
    `Infinity` is deliberately NOT clamped here (`L11-R016`; see `_needs_host_timer_clamp`'s own
    docstring) -- it falls through to the ordinary `remaining > 0` check below, which already
    treats any non-positive duration as "no wait needed," matching Pi's own arithmetic-layer
    resolution of negative `Infinity` to a plain, ordinary, non-special value."""
    if _needs_host_timer_clamp(seconds):
        seconds = NON_FINITE_INTERVAL_FALLBACK_SECONDS
    if signal is not None and signal.aborted:
        raise DeviceFlowCancelled(CANCEL_MESSAGE)
    remaining = seconds
    while remaining > 0:
        step = min(poll_interval_seconds, remaining)
        await sleep(step)
        remaining -= step
        if signal is not None and signal.aborted:
            raise DeviceFlowCancelled(CANCEL_MESSAGE)


async def poll_device_code_flow[T](
    poll: Callable[[], Awaitable[DevicePollResult[T]]],
    *,
    interval_seconds: float | None = None,
    expires_in_seconds: float | None = None,
    wait_before_first_poll: bool = False,
    signal: RunSignal | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now: Callable[[], float] = time.monotonic,
) -> T:
    """Drive one RFC 8628 device-authorization poll loop to completion (Pi
    `pollOAuthDeviceCodeFlow`).

    `poll` is the only transport seam -- this function owns every interval/backoff/deadline/
    cancellation decision. `sleep` and `now` are injectable so tests run with no real waiting and
    a fully controlled clock; `now` uses a MONOTONIC clock by default (Pi's own `Date.now()` is
    wall-clock -- an immaterial, deliberate difference for a pure elapsed-time deadline, since
    nothing in this contract depends on wall-clock jumps).

    `expires_in_seconds=None` means no deadline (matches Pi's own `Number.POSITIVE_INFINITY`
    fallback). Raises `DeviceFlowCancelled`, `DeviceFlowFailed`, or `DeviceFlowTimedOut` on every
    non-success outcome; returns `poll`'s own `DevicePollComplete.value` on success.
    """
    deadline = now() + expires_in_seconds if expires_in_seconds is not None else float("inf")
    _initial_floored = _floor_to_whole_milliseconds(
        interval_seconds if interval_seconds is not None else DEFAULT_POLL_INTERVAL_SECONDS
    )
    # `NaN`/positive `Infinity` are left UNCHANGED here (`L11-R014`) -- Python's own two-argument
    # `max()` would silently neutralize a NaN operand via its own order-dependent comparison
    # (`max(MINIMUM_INTERVAL_SECONDS, nan)` returns `MINIMUM_INTERVAL_SECONDS`, since `nan >
    # MINIMUM_INTERVAL_SECONDS` is `False`), which would incorrectly bypass `abortable_sleep`'s own
    # explicit host-timer clamp below before it ever runs. Deferring to that single clamp point
    # keeps NaN and positive Infinity on the exact same path, rather than one being neutralized
    # here and the other reaching `abortable_sleep` still non-finite. NEGATIVE Infinity is
    # deliberately EXCLUDED from this deferral (`L11-R016`) -- it takes the ordinary `max()` branch
    # below, exactly like any other finite-but-too-small value, since Python's `max()` already
    # resolves it correctly (unlike `NaN`) and Pi's own `Math.max` does the exact same thing.
    interval = (
        _initial_floored
        if _needs_host_timer_clamp(_initial_floored)
        else max(MINIMUM_INTERVAL_SECONDS, _initial_floored)
    )
    slow_down_responses = 0

    if wait_before_first_poll:
        remaining = deadline - now()
        if remaining > 0:
            await abortable_sleep(min(interval, remaining), signal, sleep=sleep)

    while now() < deadline:
        if signal is not None and signal.aborted:
            raise DeviceFlowCancelled(CANCEL_MESSAGE)

        result = await poll()

        if isinstance(result, DevicePollComplete):
            return result.value
        if isinstance(result, DevicePollFailed):
            raise DeviceFlowFailed(result.message)
        if isinstance(result, DevicePollSlowDown):
            slow_down_responses += 1
            server_interval = result.interval_seconds
            if (
                server_interval is not None
                and math.isfinite(server_interval)
                and server_interval > 0
            ):
                interval = max(
                    MINIMUM_INTERVAL_SECONDS, _floor_to_whole_milliseconds(server_interval)
                )
            else:
                # `interval + SLOW_DOWN_INCREMENT_SECONDS` stays NaN/positive-Infinity if `interval`
                # itself still is (an unclamped initial interval deferred by the SAME predicate
                # above, `L11-R014`/`L11-R016`) -- the SAME `max()`-neutralization hazard as the
                # initial-interval computation above applies here too, so it gets the same explicit
                # host-timer-clamp check rather than relying on `max()`'s own order-dependent NaN
                # comparison. `interval` is never negative-Infinity by this point (already resolved
                # ordinarily at setup), but the SAME predicate is used here for correctness-by-
                # construction rather than relying on that invariant holding forever.
                _incremented = interval + SLOW_DOWN_INCREMENT_SECONDS
                interval = (
                    _incremented
                    if _needs_host_timer_clamp(_incremented)
                    else max(MINIMUM_INTERVAL_SECONDS, _incremented)
                )
        # DevicePollPending (or a handled slow_down above): fall through to sleep-and-retry.

        remaining = deadline - now()
        if remaining <= 0:
            break
        await abortable_sleep(min(interval, remaining), signal, sleep=sleep)

    message = SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE
    raise DeviceFlowTimedOut(message)
