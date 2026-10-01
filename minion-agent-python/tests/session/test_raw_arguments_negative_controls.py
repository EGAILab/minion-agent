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
