"""Executes `conformance/agent/*.yaml` auth-device-code (Layer 11 Pass 1, `PROV-010`) scenarios.

Drives the real `poll_device_code_flow` state machine directly through a scripted `poll`
sequence -- this module implements no interval/backoff/deadline logic itself; that is
`poll_device_code_flow`'s own job (see its own module docstring for why keeping that here, not
duplicated into this runner, is what makes the canonical evidence a genuine cross-language proof).

Real-time precision and cancellation are deliberately NOT exercised through this canonical shape
(Pass-1 `IMPLEMENTATION`/`CANONICAL EVIDENCE` decision): a fake, instantly-advancing clock stands
in for `time.monotonic()`/`asyncio.sleep`, so what is actually under test is the deterministic
SEQUENCE-OF-SCRIPTED-POLL-OUTCOMES to FINAL-RESULT mapping -- the same property `FakeClock` proves
in `tests/auth/test_device_code.py`, replayed here through the language-neutral scenario format.
`RunSignal`/abort behavior stays Python-only unit-test evidence for the same reason.
"""

from __future__ import annotations

from typing import Any

from minion_agent.auth.device_code import (
    DeviceFlowFailed,
    DeviceFlowTimedOut,
    DevicePollComplete,
    DevicePollFailed,
    DevicePollPending,
    DevicePollResult,
    DevicePollSlowDown,
    poll_device_code_flow,
)


class _InstantClock:
    """A monotonic-shaped fake clock advanced only by the amount its own `sleep` is asked to
    wait -- no real wall-clock time passes. Identical in kind to `test_device_code.py`'s own
    `FakeClock`; kept as a separate class here since a conformance runner must not import test
    helpers from `tests/auth/`."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def now(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


def _build_outcome(entry: dict[str, Any]) -> DevicePollResult[str]:
    if "pending" in entry:
        return DevicePollPending()
    if "slow_down" in entry:
        return DevicePollSlowDown(interval_seconds=entry["slow_down"].get("interval_seconds"))
    if "failed" in entry:
        return DevicePollFailed(entry["failed"]["message"])
    if "complete" in entry:
        return DevicePollComplete(entry["complete"]["value"])
    raise ValueError(f"unrecognized poll_sequence entry: {entry!r}")  # pragma: no cover


async def run_auth_device_code_scenario(document: dict[str, Any]) -> dict[str, Any]:
    """Run one `auth_device_code` scenario and return `{poll_count, complete|error}`,
    matching this family's own `expect` shape exactly."""
    spec_doc = document["auth_device_code"]
    outcomes = [_build_outcome(entry) for entry in spec_doc["poll_sequence"]]
    call_count = 0

    async def poll() -> DevicePollResult[str]:
        nonlocal call_count
        result = outcomes[call_count]
        call_count += 1
        return result

    clock = _InstantClock()
    kwargs: dict[str, Any] = {"sleep": clock.sleep, "now": clock.now}
    if "interval_seconds" in spec_doc:
        kwargs["interval_seconds"] = spec_doc["interval_seconds"]
    if "expires_in_seconds" in spec_doc:
        kwargs["expires_in_seconds"] = spec_doc["expires_in_seconds"]

    try:
        value = await poll_device_code_flow(poll, **kwargs)
    except DeviceFlowTimedOut as error:
        return {"poll_count": call_count, "error": {"type": "timed_out", "message": str(error)}}
    except DeviceFlowFailed as error:
        return {"poll_count": call_count, "error": {"type": "failed", "message": str(error)}}

    return {"poll_count": call_count, "complete": {"value": value}}
