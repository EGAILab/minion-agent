"""L0506-D001 (`TOOL-041`) binding-level negative controls -- the Owner decision's §8 list. Each
is a single-point mutant of the real Layer 06 code that a canonical `prepared_runtime` case must
kill."""

from __future__ import annotations

import inspect
import math
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from minion_agent.tools import execute as execute_module

from ..conformance import prepared_runtime_runner as runner
from ..conformance.test_prepared_runtime_conformance import CASES, check


def _cases(*ids: str) -> list[dict[str, Any]]:
    wanted = set(ids)
    selected = [case for case in CASES if case["id"] in wanted]
    assert len(selected) == len(wanted)
    return selected


async def _all_pass(cases: list[dict[str, Any]]) -> bool:
    try:
        for case in cases:
            check(case, await runner.run_case(case))
    except AssertionError:
        return False
    return True


def _map_numbers(value: Any, transform: Callable[[float], Any]) -> Any:
    if isinstance(value, dict):
        return {key: _map_numbers(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_numbers(item, transform) for item in value]
    if isinstance(value, float):
        return transform(value)
    return value


def _prepare_then(transform: Callable[[float], Any]) -> Callable[..., Any]:
    original = execute_module._prepare

    def mutant(definition: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        return _map_numbers(original(definition, arguments), transform)  # type: ignore[no-any-return]

    return mutant


OVERFLOW = ("undeclared-pos-inf", "undeclared-neg-inf")
NEGATIVE_ZERO = ("declared-integer-neg-zero", "undeclared-neg-zero", "declared-number-neg-zero")
NON_FINITE_UNDECLARED = ("undeclared-pos-inf", "undeclared-neg-inf", "undeclared-nan")


async def test_every_witness_passes_unmutated() -> None:
    assert await _all_pass(
        _cases(*OVERFLOW, *NEGATIVE_ZERO, *NON_FINITE_UNDECLARED, "declared-number-pos-inf")
    )


def _non_finite_rejected(value: float) -> float:
    if not math.isfinite(value):
        raise execute_module.ArgumentValidationError("number out of range")
    return value


@pytest.mark.parametrize(
    ("name", "transform"),
    [
        ("infinity rejected (a type that cannot hold it)", _non_finite_rejected),
        ("infinity mapped to null", lambda v: None if not math.isfinite(v) else v),
        (
            "infinity clamped to max finite",
            lambda v: math.copysign(sys.float_info.max, v) if math.isinf(v) else v,
        ),
        ("infinity stringified", lambda v: str(v) if not math.isfinite(v) else v),
    ],
)
async def test_non_finite_representation_mutants_are_killed(
    monkeypatch: pytest.MonkeyPatch, name: str, transform: Callable[[float], Any]
) -> None:
    del name
    monkeypatch.setattr(execute_module, "_prepare", _prepare_then(transform))
    assert not await _all_pass(_cases(*OVERFLOW))
    assert not await _all_pass(_cases(*NON_FINITE_UNDECLARED))


async def test_negative_zero_collapse_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execute_module, "_prepare", _prepare_then(lambda v: 0.0 if v == 0 else v))
    for case in _cases(*NEGATIVE_ZERO):
        assert not await _all_pass([case]), case["id"]


async def test_validator_accepting_non_finite_in_declared_number_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L0506-D001-C001 reverted: the plain Draft 2020-12 `number` admits non-finite floats."""
    monkeypatch.setattr(
        execute_module, "PreparedArgumentsValidator", execute_module.Draft202012Validator
    )
    for case in _cases("declared-number-pos-inf", "declared-number-neg-inf", "declared-number-nan"):
        assert not await _all_pass([case]), case["id"]


KEYWORD_APPLICABILITY = (  # the cells that discriminate the plain Draft 2020-12 mutant
    "bound-maximum-pos-inf",
    "bound-minimum-neg-inf",
    "bound-exclusive-maximum-pos-inf",
    "bound-exclusive-minimum-neg-inf",
    "multiple-of-pos-inf",
    "multiple-of-nan",
    "one-of-bounds-pos-inf",
    "one-of-bounds-neg-inf",
    "not-bound-pos-inf",
)


async def test_numeric_keywords_applied_to_non_finite_values_are_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L0506-D001-RC002: a validator treating a non-finite runtime number as a JSON-Schema number
    applies minimum/maximum/exclusive*/multipleOf to it (and flips oneOf/not). Pinned Pi does
    not: each case kills it."""
    monkeypatch.setattr(
        execute_module, "PreparedArgumentsValidator", execute_module.Draft202012Validator
    )
    for case in _cases(*KEYWORD_APPLICABILITY):
        assert not await _all_pass([case]), case["id"]


@pytest.fixture
def mutant_execute() -> Iterator[Callable[[str, str], types.ModuleType]]:
    created: list[str] = []

    def build(old: str, new: str) -> types.ModuleType:
        source = inspect.getsource(execute_module)
        assert source.count(old) == 1, old
        clone = types.ModuleType(execute_module.__name__ + "_mutant")
        clone.__package__ = execute_module.__package__
        sys.modules[clone.__name__] = clone
        created.append(clone.__name__)
        exec(compile(source.replace(old, new), clone.__name__, "exec"), clone.__dict__)
        return clone

    yield build
    for name in created:
        del sys.modules[name]


async def test_hook_projection_losing_non_finite_is_killed(
    monkeypatch: pytest.MonkeyPatch, mutant_execute: Callable[[str, str], types.ModuleType]
) -> None:
    """The hook is handed a JSON-style projection (non-finite -> None) while execute still gets
    the real value: only the hook-observation half of each case catches it."""
    mutant = mutant_execute(
        "            validated_arguments,\n            signal,\n            terminal=",
        "            _project(validated_arguments),\n            signal,\n            terminal=",
    )
    mutant._project = lambda value: _map_numbers(  # type: ignore[attr-defined]
        value, lambda v: None if not math.isfinite(v) else v
    )
    monkeypatch.setattr(runner, "execute_call", mutant.execute_call)
    for case in _cases(*NON_FINITE_UNDECLARED, *OVERFLOW):
        assert not await _all_pass([case]), case["id"]
