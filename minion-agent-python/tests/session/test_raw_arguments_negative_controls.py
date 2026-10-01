"""L0206-D002 (`AI-003` raw `ToolCall.arguments` value domain) negative controls -- the Owner
decision's section 11 list (`minion-agent#99` comment 5926162818). Each is a single-point mutant
of the real certified seam that canonical `raw_arguments` cases must kill (known-bad -> FAIL,
candidate -> PASS)."""

from __future__ import annotations

import inspect
import json
import math
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from minion_agent.session import derive as derive_module
from minion_agent.session import log as log_module
from minion_agent.tools import execute as execute_module

from ..conformance import raw_arguments_runner as runner
from ..conformance.test_raw_arguments_conformance import CASES

LONE = ("string/lone-high-only", "string/lone-low-only", "string/low-then-high")
SURROGATE_KEYS = ("keys/surrogate-keys",)
NON_FINITE = ("number/overflow-positive", "number/overflow-negative")
NEGATIVE_ZERO = ("number/negative-zero",)


def _cases(*ids: str) -> list[dict[str, Any]]:
    selected = [case for case in CASES if case["id"] in set(ids)]
    assert len(selected) == len(ids)
    return selected


async def _killed(*ids: str) -> list[str]:
    killed = []
    for case in _cases(*ids):
        try:
            runner.check(case, await runner.run_case(case))
        except AssertionError:
            killed.append(case["id"])
    return sorted(killed)


def _is_unpaired(ch: str) -> bool:
    return 0xD800 <= ord(ch) <= 0xDFFF


def _map(value: Any, strings: Callable[[str], Any], numbers: Callable[[Any], Any]) -> Any:
    if isinstance(value, dict):
        return {strings(k): _map(v, strings, numbers) for k, v in value.items()}
    if isinstance(value, list):
        return [_map(v, strings, numbers) for v in value]
    if isinstance(value, str):
        return strings(value)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return numbers(value)
    return value


def _replace(s: str) -> str:
    return "".join("�" if _is_unpaired(ch) else ch for ch in s)


def _json_projection(value: Any) -> Any:
    """Pi's own persisted-file projection (JSON.stringify): -0 -> 0, +/-Infinity -> null."""

    def number(v: Any) -> Any:
        if isinstance(v, float) and math.isinf(v):
            return None
        if isinstance(v, float) and v == 0:
            return 0
        return v

    return _map(value, lambda s: s, number)


async def test_every_witness_passes_unmutated() -> None:
    assert await _killed(*LONE, *SURROGATE_KEYS, *NON_FINITE, *NEGATIVE_ZERO) == []


def _arguments_mutant(transform: Callable[[Any], Any]) -> Callable[..., Any]:
    """Wrap `encode_message` (session encode) or `decode_message` (session decode) so the
    tool call's arguments pass through `transform`."""

    def wrap(original: Callable[..., Any], encode: bool) -> Callable[..., Any]:
        def mutant(message: Any) -> Any:
            data = original(message)
            if encode:
                for block in data.get("content", []):
                    if isinstance(block, dict) and "arguments" in block:
                        block["arguments"] = transform(block["arguments"])
                return data
            for block in data.content:
                if hasattr(block, "arguments"):
                    object.__setattr__(block, "arguments", transform(block.arguments))
            return data

        return mutant

    return wrap


def _reject_unpaired(value: Any) -> Any:
    def strings(s: str) -> str:
        if any(_is_unpaired(ch) for ch in s):
            raise log_module.NotJsonSafeError("lone surrogate")
        return s

    return _map(value, strings, lambda n: n)


@pytest.mark.parametrize(
    ("name", "transform", "encode", "kills"),
    [
        ("rejects a raw lone surrogate", _reject_unpaired, True, (*LONE, *SURROGATE_KEYS)),
        (
            "replaces it with U+FFFD at decode",
            lambda v: _map(v, _replace, lambda n: n),
            False,
            (*LONE, *SURROGATE_KEYS),
        ),
        (
            "session decode normalizes (lossy UTF-8 round trip)",
            lambda v: _map(v, lambda s: s.encode("utf-8", "replace").decode(), lambda n: n),
            False,
            (*LONE, *SURROGATE_KEYS),
        ),
        (
            "correct until persistence, lost on replay (JSON projection)",
            _json_projection,
            True,
            (*NON_FINITE, *NEGATIVE_ZERO),
        ),
    ],
)
async def test_session_boundary_mutants_are_killed(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    transform: Callable[[Any], Any],
    encode: bool,
    kills: tuple[str, ...],
) -> None:
    del name
    wrap = _arguments_mutant(transform)
    if encode:
        monkeypatch.setattr(runner, "encode_message", wrap(derive_module.encode_message, True))
    else:
        monkeypatch.setattr(runner, "decode_message", wrap(derive_module.decode_message, False))
    assert await _killed(*kills) == sorted(kills)


async def test_session_persistence_that_cannot_encode_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A log whose JSON-safety check is strict JSON text (no lone surrogates, no non-finite)."""

    def strict(value: Any, path: str = "data") -> None:
        try:
            json.dumps(value, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeEncodeError) as error:
            raise log_module.NotJsonSafeError(str(error)) from error

    monkeypatch.setattr(log_module, "_check_json_safe", strict)
    assert await _killed(*NON_FINITE) == sorted(NON_FINITE)


async def test_event_payload_sees_normalized_value_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tools/execution-start payload is handed a normalized copy while the pipeline keeps the
    real value: only the event-payload observation catches it. Built as a single-point source
    mutant of the real emission site in a cloned `execute` module."""
    source = inspect.getsource(execute_module)
    old = "ctx.events.emit(TOOLS_EXECUTION_START, call.id, call.name, call.arguments, scope=scope)"
    assert source.count(old) == 1
    clone = types.ModuleType(execute_module.__name__ + "_raw_mutant")
    clone.__package__ = execute_module.__package__
    sys.modules[clone.__name__] = clone
    try:
        new = old.replace("call.arguments,", "_project(call.arguments),")
        exec(compile(source.replace(old, new), clone.__name__, "exec"), clone.__dict__)
        clone._project = lambda value: _map(value, _replace, lambda n: n)  # type: ignore[attr-defined]
        monkeypatch.setattr(runner, "execute_call", clone.execute_call)
        assert await _killed(*LONE) == sorted(LONE)
    finally:
        del sys.modules[clone.__name__]


# ---- L0206-D002-R001: the update boundary (tools/update payload, on_execution_update delivery)


def _mutant_execute(old: str, new: str) -> types.ModuleType:
    source = inspect.getsource(execute_module)
    assert source.count(old) == 1, old
    clone = types.ModuleType(execute_module.__name__ + "_update_mutant")
    clone.__package__ = execute_module.__package__
    sys.modules[clone.__name__] = clone
    exec(compile(source.replace(old, new), clone.__name__, "exec"), clone.__dict__)
    clone._project = lambda value: _map(value, _replace, lambda n: n)  # type: ignore[attr-defined]
    return clone


@pytest.mark.parametrize(
    ("old", "boundary"),
    [
        (
            "ctx.events.emit(TOOLS_UPDATE, call.id, call.name, call.arguments, partial, "
            "scope=scope)",
            "update_event",
        ),
        ("on_execution_update(call.id, call.name, call.arguments, partial),", "update_delivery"),
    ],
    ids=["tools-update-payload", "on-execution-update-delivery"],
)
async def test_an_update_only_normalization_is_killed(
    monkeypatch: pytest.MonkeyPatch, old: str, boundary: str
) -> None:
    """A normalized raw argument at ONE update seam only: start, hook and execute stay correct,
    so only that update observation can catch it."""
    clone = _mutant_execute(old, old.replace("call.arguments,", "_project(call.arguments),", 1))
    try:
        monkeypatch.setattr(runner, "execute_call", clone.execute_call)
        for case in _cases(*LONE):
            seen = await runner.run_case(case)
            want = runner.observe(runner.decode(case["arguments"]))
            assert all(seen[b] == want for b in runner.BOUNDARIES), case["id"]
            assert seen[boundary] != [want], (case["id"], boundary)
        assert await _killed(*LONE) == sorted(LONE)
    finally:
        del sys.modules[clone.__name__]


# ---- L0206-D002-R002: the numeric fixture grammar


@pytest.mark.parametrize(
    "token",
    ["9007199254740993", "1e999", "-1e999", "NaN", "-0.0", "1.0", "1E3", "0.1000"],
)
def test_the_preflight_refuses_a_non_canonical_or_out_of_domain_number(token: str) -> None:
    """An unrounded integer, an overflowing literal, NaN, or any literal that is not the exact
    Number::toString of its binary64 value fails the document before dispatch."""
    with pytest.raises(AssertionError, match="number token"):
        runner.preflight({"n": {"number": token}})


@pytest.mark.parametrize(
    "token", ["0", "9007199254740992", "1.7976931348623157e+308", "5e-324", "0.1", "-1.5", "1e+21"]
)
def test_the_preflight_accepts_canonical_finite_literals(token: str) -> None:
    runner.preflight({"n": {"number": token}})


def test_named_tokens_are_exactly_the_non_finite_and_negative_zero_values() -> None:
    assert set(runner.NAMED) == {"+Infinity", "-Infinity", "-0"}


MIDPOINT = 2**1024 - 2**970  # the smallest int whose float conversion overflows
INVALID_INTS = [
    9007199254740993,
    1000000000000000100,
    -1000000000000000100,
    MIDPOINT - 1,  # rounds to the largest finite double, but is not one
    MIDPOINT,
    2**1024,
    2**1024 + 1,
    10**4299,
    -(10**4299),
    10**4300,  # str() raises at the interpreter's 4300-digit limit
    -(10**4300),
    10**5000,
]


@pytest.mark.parametrize(
    "value", INVALID_INTS, ids=lambda v: f"bits{v.bit_length()}{'neg' if v < 0 else ''}"
)
def test_observation_is_strict_and_total_for_every_out_of_domain_int(value: int) -> None:
    """CE-L0206-D002-01 N3': both exception boundaries (float overflow, decimal digit limit) observe
    as the controlled hex marker without raising, and no process-wide setting changes."""
    before = sys.get_int_max_str_digits()
    assert runner.observe(value) == {"non_binary64_int": hex(value)}
    assert sys.get_int_max_str_digits() == before


@pytest.mark.parametrize(
    ("value", "token"),
    [
        (1000000000000000128, "1000000000000000100"),
        (-1000000000000000128, "-1000000000000000100"),
        (2**53, "9007199254740992"),
        (10**18, "1000000000000000000"),
    ],
)
def test_exact_binary64_ints_observe_as_their_number_token(value: int, token: str) -> None:
    assert runner.observe(value) == {"number": token}


_STRICT_OBSERVE = runner.observe


def _lossy_observe(value: Any) -> Any:
    """The rejected observer: renders an int through float(), rounding a wrong value away."""
    if isinstance(value, int) and not isinstance(value, bool):
        return {"number": runner.render(float(value))}
    if isinstance(value, list):
        return [_lossy_observe(item) for item in value]
    if isinstance(value, dict):
        return {tuple(runner.units(k)): _lossy_observe(v) for k, v in value.items()}
    return _STRICT_OBSERVE(value)


@pytest.mark.parametrize("value", [1000000000000000100, -1000000000000000100, 9007199254740993])
def test_the_lossy_observer_is_killed_by_direct_units(value: int) -> None:
    """C-L0206-D002-01-02: the float-coercing observer is distinguishable only on a malformed
    runtime int -- the strict observer marks it, the lossy one renders a valid-looking token."""
    assert runner.observe(value) == {"non_binary64_int": hex(value)}
    assert _lossy_observe(value) != runner.observe(value)


NON_EXACT = ("number/non-exact-integer-spelling", "number/non-exact-integer-spelling-negative")


def test_the_number_decoder_goes_through_binary64() -> None:
    """N1/N2: a canonical integer spelling is decoded as its exact binary64 integer."""
    assert runner.number("1000000000000000100") == 1000000000000000128
    assert runner.number("-1000000000000000100") == -1000000000000000128
    assert runner.number("1e+21") == 1e21 and isinstance(runner.number("1e+21"), float)


async def test_the_spelled_digits_decoder_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old `int(token)` decoder delivers 1000000000000000100, which is not a binary64 value: the
    strict observer marks it and the independent expectation (the token) refuses it."""
    assert await _killed(*NON_EXACT) == []

    def spelled(token: str) -> Any:
        return (
            runner.NAMED[token]
            if token in runner.NAMED
            else (int(token) if not any(m in token for m in ".eE") else float(token))
        )

    monkeypatch.setattr(runner, "number", spelled)
    assert await _killed(*NON_EXACT) == sorted(NON_EXACT)


def _wrong(value: Any) -> Any:
    """Deliver the spelled (non-binary64) integer where its binary64 value is due."""
    return _map(
        value, lambda s: s, lambda n: 1000000000000000100 if n == 1000000000000000128 else n
    )


def _seam(old: str, target: str, seam: str) -> tuple[str, str, str]:
    """A single-point mutant: `target` inside the real source line `old` becomes `_project(...)`."""
    return old, old.replace(target, f"_project({target})", 1), seam


SEAMS = [
    _seam(
        "validated_arguments,\n            signal,\n            terminal=",
        "validated_arguments",
        "hook",
    ),
    _seam("arguments=decision.arguments)", "decision.arguments", "execute"),
    _seam(
        "ctx.events.emit(TOOLS_EXECUTION_START, call.id, call.name, call.arguments, scope=scope)",
        "call.arguments",
        "execution_start",
    ),
    _seam(
        "ctx.events.emit(TOOLS_UPDATE, call.id, call.name, call.arguments, partial, scope=scope)",
        "call.arguments",
        "update_event",
    ),
    _seam(
        "on_execution_update(call.id, call.name, call.arguments, partial),",
        "call.arguments",
        "update_delivery",
    ),
]


def _seam_mutant(old: str, new: str) -> types.ModuleType:
    source = inspect.getsource(execute_module)
    assert source.count(old) == 1, old
    clone = types.ModuleType(execute_module.__name__ + "_number_mutant")
    clone.__package__ = execute_module.__package__
    sys.modules[clone.__name__] = clone
    exec(compile(source.replace(old, new), clone.__name__, "exec"), clone.__dict__)
    clone._project = _wrong  # type: ignore[attr-defined]
    return clone


@pytest.mark.parametrize(("old", "new", "seam"), SEAMS, ids=[s[2] for s in SEAMS])
async def test_a_wrong_number_at_one_seam_is_refused(
    monkeypatch: pytest.MonkeyPatch, old: str, new: str, seam: str
) -> None:
    clone = _seam_mutant(old, new)
    try:
        monkeypatch.setattr(runner, "execute_call", clone.execute_call)
        case = _cases(NON_EXACT[0])[0]
        seen = await runner.run_case(case)
        want = runner.expect(case["arguments"])
        got = seen[seam][0] if seam.startswith("update") else seen[seam]
        assert got != want, seam
        assert await _killed(NON_EXACT[0]) == [NON_EXACT[0]]
    finally:
        del sys.modules[clone.__name__]


async def test_a_wrong_number_at_replay_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    wrap = _arguments_mutant(_wrong)
    monkeypatch.setattr(runner, "decode_message", wrap(derive_module.decode_message, False))
    assert await _killed(NON_EXACT[0]) == [NON_EXACT[0]]


async def test_only_the_strict_observer_refuses_a_wrong_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demonstration (C-L0206-D002-01-02): the hook-seam wrong value is refused with the strict
    observer and silently accepted with the lossy one -- strictness is what does the refusing."""
    old, new, _ = SEAMS[0]
    clone = _seam_mutant(old, new)
    try:
        monkeypatch.setattr(runner, "execute_call", clone.execute_call)
        assert await _killed(NON_EXACT[0]) == [NON_EXACT[0]]
        monkeypatch.setattr(runner, "observe", _lossy_observe)
        assert await _killed(NON_EXACT[0]) == []
    finally:
        del sys.modules[clone.__name__]


async def test_a_hook_receiving_an_unrounded_integer_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real hook-value witness: a pipeline that widens the binary64 value to the unrounded
    integer the provider text spelled (a non-JSON.parse decoder's result) fails the case."""
    case = next(c for c in CASES if c["id"] == "number/integer-2p53-plus-1")
    assert case["arguments"] == {"n": {"number": "9007199254740992"}}
    original = execute_module._prepare
    monkeypatch.setattr(
        execute_module,
        "_prepare",
        lambda d, a: {**original(d, a), "n": 9007199254740993},
    )
    assert await _killed(case["id"]) == [case["id"]]
