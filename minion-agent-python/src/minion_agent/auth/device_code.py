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
"""The delay `abortable_sleep` clamps an INVALID delay to (`L11-R014`/`L11-R017`, §11.8
convergence). NOT `MINIMUM_INTERVAL_SECONDS` -- that constant is RFC 8628's own "never poll faster
than this" floor for ORDINARY, finite intervals, a different concept entirely. This value instead
matches pinned Pi's own OBSERVABLE magnitude as closely as Python reasonably can: Pi's own EXPORTED
`abortableSleep` hands its `ms` argument straight to `setTimeout` with NO preceding normalization,
and Node's own documented contract ("If delay is larger than 2147483647 or less than 1, the delay
will be set to 1") clamps ANY delay outside `[1, 2147483647]` milliseconds -- `NaN` (every
comparison against `NaN` is `false`, so it fails the range check both ways), positive `Infinity`,
NEGATIVE `Infinity`, zero, and any other negative or sub-millisecond value -- to exactly ONE
MILLISECOND, independently confirmed live against a real Node process by an earlier review. This
is a full model of that ENTIRE bounds check (`_needs_setimeout_clamp`'s own predicate), not merely
the `NaN`/positive-`Infinity` subset an earlier revision handled.

TWO SEPARATE BOUNDARIES, TWO DIFFERENT RULES (`L11-R017`): pinned Pi's own `pollOAuthDeviceCodeFlow`
NORMALIZES its own initial interval via `Math.max(MINIMUM_INTERVAL_MS, Math.floor(...))` BEFORE
ever calling `abortableSleep` -- ordinary `Math.max` already resolves negative `Infinity` to
`MINIMUM_INTERVAL_MS` correctly (it is a valid, comparable number that simply loses that
comparison), so negative `Infinity` NEVER reaches `abortableSleep`/`setTimeout` as an invalid delay
through THAT path (`poll_device_code_flow`'s own interval computation, which checks only
`math.isnan` before deferring, not this predicate). But `abortableSleep` is ALSO independently
EXPORTED and callable directly, bypassing that normalization entirely -- called that way, a raw
negative `Infinity` (or any other out-of-range value) DOES reach the host timer as invalid and gets
clamped, exactly like `NaN`/positive `Infinity` do. `abortable_sleep`, this module's own direct
analog of that exported function, must model that DIRECT-CALL contract in full; a caller going
through `poll_device_code_flow` never actually exercises this predicate for negative `Infinity`,
since the poll loop's own upstream normalization has already turned it into an ordinary, valid
delay by the time `abortable_sleep` is called."""


def _needs_setimeout_clamp(seconds: float) -> bool:
    """Models Node's own documented `setTimeout` delay-bounds check in full (`L11-R017`): a delay
    is invalid -- and gets clamped to `NON_FINITE_INTERVAL_FALLBACK_SECONDS` -- unless it is a
    NUMBER in the INCLUSIVE range `[1, 2147483647]` milliseconds. `NaN` fails both comparisons (any
    comparison against `NaN` is `false` in both Python and JS, so this predicate needs no explicit
    `math.isnan` check -- the ordinary chained comparison already produces the correct answer for
    it); positive `Infinity` fails the upper bound; NEGATIVE `Infinity`, zero, any negative number,
    and any positive sub-millisecond number all fail the lower bound. This is `abortable_sleep`'s
    OWN predicate, modeling Pi's exported `abortableSleep`/`setTimeout` boundary directly -- it is
    deliberately NOT used by `poll_device_code_flow`'s own upstream interval computation, which
    models a DIFFERENT boundary (Pi's own pure `Math.max` arithmetic, which already resolves
    negative `Infinity` correctly on its own, see `NON_FINITE_INTERVAL_FALLBACK_SECONDS`'s own
    docstring for why these are two separate rules)."""
    milliseconds = seconds * 1000
    return not (1 <= milliseconds <= 2_147_483_647)


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
    not this pure-math one.

    EXACT truncation, no tolerance of any kind (`L11-R020`/`L11-R021`): this helper is used for
    every delay this module truncates, without exception -- the caller's own initial interval and a
    server-provided `slow_down` interval (`L11-R012`), `abortable_sleep`'s own direct-call
    truncation of an already-valid delay (`L11-R019`), AND `poll_device_code_flow`'s own
    deadline-capped `remaining` value passed into `abortable_sleep` (`L11-R021`). Two earlier
    revisions each tried adding a small epsilon tolerance to compensate for floating-point
    SUMMATION drift this module's own test doubles can introduce -- first directly here (`L11-
    R019`), then narrowed to a separately-named helper applied only to `remaining` (`L11-R020`) --
    and BOTH were wrong, for the same underlying reason: `remaining = deadline - now()` is not
    reliably "internal" or "drift-affected" -- on a FIRST computation (e.g. `wait_before_first_poll`
    with a short `expires_in_seconds` and no sleep yet performed), it is EXACTLY the caller's own
    supplied expiry value, arithmetic-identical to a genuine public input. Pinned Pi's real
    `setTimeout` truncates `1.9999995` ms down to `1` ms (`Math.trunc` never rounds); an epsilon
    applied to `remaining` in that exact scenario instead rounded it UP to `2` ms -- confirming no
    tolerance belongs ANYWHERE in this module's own production scheduling arithmetic, regardless of
    how narrowly it is scoped. The floating-point-summation drift these two revisions were actually
    trying to compensate for is a property of a CLOCK TEST DOUBLE that advances via many small
    repeated additions (`FakeClock`/`_InstantClock`'s own `sleep`, driven by `abortable_sleep`'s own
    signal-polling slicing loop) -- it does not occur with a REAL monotonic clock, which is read
    directly rather than accumulated by summing past sleep durations. The correct fix lives entirely
    in those test doubles (rounding their own accumulated `elapsed` value after each increment),
    never in this module's own production arithmetic, which must stay exact for every input with no
    exception, matching Pi with zero tolerance."""
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

    This function is the Python analog of Pi's own EXPORTED `abortableSleep`, which hands its `ms`
    argument straight to `setTimeout` with no preceding normalization (`L11-R017`) -- so an INVALID
    `seconds` (per `_needs_setimeout_clamp`'s own full Node-`setTimeout`-bounds predicate: `NaN`,
    positive `Infinity`, NEGATIVE `Infinity`, zero, any negative number, or any positive
    sub-millisecond number) clamps to `NON_FINITE_INTERVAL_FALLBACK_SECONDS`, matching Node's own
    documented one-millisecond minimum. This is DIFFERENT from -- and independent of --
    `poll_device_code_flow`'s own upstream interval computation, which models Pi's SEPARATE
    `Math.max`-based normalization boundary and already resolves negative `Infinity` to an ordinary
    valid delay before ever calling this function (see `NON_FINITE_INTERVAL_FALLBACK_SECONDS`'s own
    docstring for the full two-boundary explanation). A caller invoking THIS function directly with
    a raw negative `Infinity` (or any other invalid delay) -- bypassing the poll loop's own
    normalization entirely -- still gets the SAME one-millisecond clamp `NaN`/positive `Infinity`
    already receive, matching Pi's own exported function exactly.

    A VALID delay (one `_needs_setimeout_clamp` does not reject) is still not scheduled at its own
    exact fractional-millisecond value (`L11-R019`): Node's real `setTimeout` internally truncates
    ANY accepted delay to a whole integer millisecond count before scheduling it, independently of
    the documented invalid-range clamp above -- a delay of `0.0019` seconds (1.9 ms), called
    directly, is neither `NaN`/`Infinity` nor outside `[1, 2147483647]` ms, so it is NOT touched by
    `_needs_setimeout_clamp`, yet Node still truncates it to exactly 1 ms, not 1.9 ms. This is a
    THIRD, separate rule from both the poll loop's own explicit `Math.floor` (`L11-R012`, applied to
    the interval BEFORE `Math.max`, as part of Pi's own visible arithmetic) and this function's own
    invalid-delay clamp above: it is `setTimeout`'s own internal behavior on an already-VALID delay,
    so it applies here unconditionally, even to a delay the poll loop's own upstream flooring never
    touched (a raw fractional-millisecond value passed directly to this exported function). Reusing
    `_floor_to_whole_milliseconds` is safe here specifically because this branch is only reached for
    an already-finite `seconds` (the non-finite/out-of-range cases all take the clamp branch above),
    and flooring an already-whole-millisecond value (as every poll-loop-sourced call already is, per
    `L11-R012`) is a no-op, so no double-application hazard exists for that caller."""
    if _needs_setimeout_clamp(seconds):
        seconds = NON_FINITE_INTERVAL_FALLBACK_SECONDS
    else:
        seconds = _floor_to_whole_milliseconds(seconds)
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
    # Only `NaN` is left UNCHANGED here (`L11-R014`/`L11-R017`) -- Python's own two-argument `max()`
    # would silently neutralize a NaN operand via its own order-dependent comparison
    # (`max(MINIMUM_INTERVAL_SECONDS, nan)` returns `MINIMUM_INTERVAL_SECONDS`, since `nan >
    # MINIMUM_INTERVAL_SECONDS` is `False`), which would incorrectly bypass `abortable_sleep`'s own
    # explicit `_needs_setimeout_clamp` check before it ever runs. Positive AND negative `Infinity`
    # both need NO special-casing at all: Python's ordinary `max()` already resolves both correctly
    # against `MINIMUM_INTERVAL_SECONDS` (`max(x, +inf) == +inf`; `max(x, -inf) == x`), matching
    # Pi's own `Math.max` exactly -- `+inf` naturally survives this branch unchanged (to be clamped
    # later, when actually used, by `abortable_sleep`'s own predicate), and `-inf` naturally
    # resolves to the ordinary RFC-8628 floor immediately, exactly like any other too-small value.
    interval = (
        _initial_floored
        if math.isnan(_initial_floored)
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
                # `interval + SLOW_DOWN_INCREMENT_SECONDS` stays `NaN` if `interval` itself still is
                # (an un-normalized initial interval deferred by the SAME `math.isnan` check above,
                # `L11-R014`/`L11-R017`) -- only `NaN` needs excluding from the ordinary `max()`
                # branch here, for the SAME reason as the initial-interval computation above:
                # Python's own `max()` would otherwise silently neutralize it via an order-dependent
                # comparison. Positive/negative `Infinity` both need no special-casing (Python's
                # ordinary `max()` already resolves both correctly against
                # `MINIMUM_INTERVAL_SECONDS`, matching Pi's own `Math.max` exactly) -- this call
                # site models the SAME poll-loop-normalization boundary as the initial-interval
                # setup above, NOT `abortable_sleep`'s own separate `_needs_setimeout_clamp`
                # boundary (see `NON_FINITE_INTERVAL_FALLBACK_SECONDS`'s own docstring for why
                # these are two different rules).
                _incremented = interval + SLOW_DOWN_INCREMENT_SECONDS
                interval = (
                    _incremented
                    if math.isnan(_incremented)
                    else max(MINIMUM_INTERVAL_SECONDS, _incremented)
                )
        # DevicePollPending (or a handled slow_down above): fall through to sleep-and-retry.

        remaining = deadline - now()
        if remaining <= 0:
            break
        await abortable_sleep(min(interval, remaining), signal, sleep=sleep)

    message = SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE
    raise DeviceFlowTimedOut(message)
