"""L0506-D002 (`TOOL-041` string domain) binding-level negative controls -- the Owner decision's §6
list (`minion-agent#49` comment 5924605017). Each is a single-point mutant of the real Layer 06
code that canonical `prepared_string` cases must kill (known-bad -> FAIL, candidate -> PASS)."""

from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from minion_agent.tools import execute as execute_module

from ..conformance import prepared_string_runner as runner
from ..conformance.test_prepared_string_conformance import CASES

LONE_HIGH = (
    "open/lone-high-start",
    "open/lone-high-middle",
    "open/lone-high-end",
    "open/lone-high-only",
)
LONE_LOW = (
    "open/lone-low-start",
    "open/lone-low-middle",
    "open/lone-low-end",
    "open/lone-low-only",
)
PAIRS = ("open/pair", "max-length-1/pair", "const-pair/pair", "pattern-one-char/pair")
KEYS = ("key/lone-high-only", "key/lone-low-only", "key/low-then-high")
# L0506-D002-R001: a scalar schema (enum [U+FFFD]) rejects every unpaired-surrogate instance
# (single-unit members only: a two-unit instance stays two characters after replacement)
FFFD_DISCRIMINATORS = ("enum-fffd/lone-high-only", "enum-fffd/lone-low-only")
MIXED = ("open/adjacent-highs", "open/low-then-high", "open/pair-then-lone-high")


def _cases(*ids: str) -> list[dict[str, Any]]:
    wanted = set(ids)
    selected = [case for case in CASES if case["id"] in wanted]
    assert len(selected) == len(wanted)
    return selected


async def _passes(case: dict[str, Any]) -> bool:
    try:
        runner.check(case, await runner.run_case(case))
    except AssertionError:
        return False
    return True


async def _killed(*ids: str) -> list[str]:
    """The ids among `ids` whose case FAILS (kills the active mutant)."""
    return sorted([case["id"] for case in _cases(*ids) if not await _passes(case)])


def _is_unpaired(ch: str) -> bool:
    return 0xD800 <= ord(ch) <= 0xDFFF  # a Python canonical str holds only unpaired surrogates


def _map_strings(value: Any, transform: Callable[[str], Any], keys: bool = True) -> Any:
    if isinstance(value, dict):
        return {
            (transform(k) if keys else k): _map_strings(item, transform, keys)
            for k, item in value.items()
        }
    if isinstance(value, list):
        return [_map_strings(item, transform, keys) for item in value]
    if isinstance(value, str):
        return transform(value)
    return value


def _replace_unpaired(s: str) -> str:
    return "".join("�" if _is_unpaired(ch) else ch for ch in s)


def _reject_unpaired(s: str) -> str:
    if any(_is_unpaired(ch) for ch in s):
        raise execute_module.ArgumentValidationError("lone surrogate")
    return s


def _pair_as_two_replacements(s: str) -> str:  # each code unit decoded on its own, lossily
    return "".join("��" if ord(ch) > 0xFFFF else ch for ch in s)


def _lone_low_only(s: str) -> str:
    return "".join("�" if 0xDC00 <= ord(ch) <= 0xDFFF else ch for ch in s)


def _prepare_then(transform: Callable[[str], Any], keys: bool = True) -> Callable[..., Any]:
    original = execute_module._prepare

    def mutant(definition: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        return _map_strings(original(definition, arguments), transform, keys)  # type: ignore[no-any-return]

    return mutant


async def test_every_witness_passes_unmutated() -> None:
    assert await _killed(*LONE_HIGH, *LONE_LOW, *PAIRS, *KEYS, *MIXED) == []


@pytest.mark.parametrize(
    ("name", "transform"),
    [
        ("lone surrogate replaced with U+FFFD during preparation", _replace_unpaired),
        ("lone surrogate rejected during preparation", _reject_unpaired),
        # a strict UTF-8 string type: the conversion itself fails on an unpaired surrogate
        ("strict UTF-8 string conversion failure", lambda s: s.encode("utf-8").decode("utf-8")),
        ("lossy UTF-16/UTF-8 conversion", lambda s: s.encode("utf-8", "replace").decode("utf-8")),
    ],
)
async def test_lone_surrogate_representation_mutants_are_killed(
    monkeypatch: pytest.MonkeyPatch, name: str, transform: Callable[[str], Any]
) -> None:
    del name
    monkeypatch.setattr(execute_module, "_prepare", _prepare_then(transform))
    assert await _killed(*LONE_HIGH) == sorted(LONE_HIGH)
    assert await _killed(*LONE_LOW) == sorted(LONE_LOW)
    assert await _killed(*KEYS) == sorted(KEYS)


async def test_premature_replacement_is_killed_by_a_scalar_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L0506-D002-R001: with the schema holding only the real U+FFFD, replacing a lone surrogate
    with U+FFFD before validation turns each rejection into an acceptance -- the verdict alone
    discriminates."""
    assert await _killed(*FFFD_DISCRIMINATORS) == []
    monkeypatch.setattr(execute_module, "_prepare", _prepare_then(_replace_unpaired))
    assert await _killed(*FFFD_DISCRIMINATORS) == sorted(FFFD_DISCRIMINATORS)


async def test_valid_pair_as_two_replacement_characters_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execute_module, "_prepare", _prepare_then(_pair_as_two_replacements))
    assert await _killed(*PAIRS) == sorted(PAIRS)


async def test_lone_low_mishandled_unlike_lone_high_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only lone LOW surrogates are replaced: the lone-low witnesses kill it, the lone-high ones
    alone would not -- both halves of the neighborhood are required."""
    monkeypatch.setattr(execute_module, "_prepare", _prepare_then(_lone_low_only))
    assert await _killed(*LONE_HIGH) == []
    assert await _killed(*LONE_LOW) == sorted(LONE_LOW)


async def test_keys_normalized_while_values_kept_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    def keys_only(value: Any) -> Any:
        if isinstance(value, dict):
            return {_replace_unpaired(k): keys_only(item) for k, item in value.items()}
        if isinstance(value, list):
            return [keys_only(item) for item in value]
        return value

    original = execute_module._prepare
    monkeypatch.setattr(execute_module, "_prepare", lambda d, a: keys_only(original(d, a)))
    assert await _killed(*KEYS) == sorted(KEYS)
    assert await _killed(*LONE_HIGH, *LONE_LOW) == []


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
        clone._project = lambda value: _map_strings(value, _replace_unpaired)  # type: ignore[attr-defined]
        return clone

    yield build
    for name in created:
        del sys.modules[name]


async def test_hook_sees_normalized_while_execute_sees_original_is_killed(
    monkeypatch: pytest.MonkeyPatch, mutant_execute: Callable[[str, str], types.ModuleType]
) -> None:
    mutant = mutant_execute(
        "            validated_arguments,\n            signal,\n            terminal=",
        "            _project(validated_arguments),\n            signal,\n            terminal=",
    )
    monkeypatch.setattr(runner, "execute_call", mutant.execute_call)
    assert await _killed(*LONE_HIGH, *LONE_LOW) == sorted([*LONE_HIGH, *LONE_LOW])


async def test_execute_sees_normalized_while_hook_sees_original_is_killed(
    monkeypatch: pytest.MonkeyPatch, mutant_execute: Callable[[str, str], types.ModuleType]
) -> None:
    mutant = mutant_execute(
        "arguments=decision.arguments)",
        "arguments=_project(decision.arguments))",
    )
    monkeypatch.setattr(runner, "execute_call", mutant.execute_call)
    assert await _killed(*LONE_HIGH, *LONE_LOW) == sorted([*LONE_HIGH, *LONE_LOW])
