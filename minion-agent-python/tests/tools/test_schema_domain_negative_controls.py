"""L05-D001 (`TOOL-016` / `TOOL-003` runtime-validation schema string domain) negative controls.
Each is a realistic wrong implementation, as a single-point mutant of the real seam, that the
canonical `schema_domain` cases must kill (known-bad -> FAIL, candidate -> PASS)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from minion_agent.tools import execute as execute_module

from ..conformance import schema_domain_runner as runner
from ..conformance.test_schema_domain_conformance import CASES


def _killed(*ids: str) -> Callable[[], Any]:
    async def run() -> list[str]:
        selected = [case for case in CASES if case["id"] in set(ids)]
        assert len(selected) == len(ids)
        killed = []
        for case in selected:
            try:
                runner.check(case, await runner.run_case(case))
            except Exception:
                killed.append(case["id"])
        return sorted(killed)

    return run


def _unpaired(ch: str) -> bool:
    return 0xD800 <= ord(ch) <= 0xDFFF


def _replace(s: str) -> str:
    return "".join("�" if _unpaired(ch) else ch for ch in s)


def _map(value: Any, strings: Callable[[str], str]) -> Any:
    if isinstance(value, dict):
        return {strings(k): _map(v, strings) for k, v in value.items()}
    if isinstance(value, list):
        return [_map(v, strings) for v in value]
    return strings(value) if isinstance(value, str) else value


LONE_SCHEMA = (  # Pi accepts: a lone surrogate in the schema matches the identical instance
    "properties-required/lone-high/lone-high",
    "const/lone-low/lone-low",
    "enum/lone-high/lone-high",
    "pattern/lone-low/lone-low",
    "property-names-const/lone-high/lone-high",
)
FFFD_LOOKALIKES = (  # Pi rejects: U+FFFD is not a lone surrogate
    "properties-required/lone-high/fffd",
    "const/lone-low/fffd",
    "enum/lone-high/fffd",
)
PAIR_HALVES = (  # Pi rejects: in Unicode mode a half is not a substring of the valid pair
    "pattern-unanchored/pair-high-half/pair",
    "pattern-unanchored/pair-low-half/pair",
    "pattern-unanchored/pair-high-half/pair-then-lone-high",
)


async def test_every_witness_passes_unmutated() -> None:
    assert await _killed(*LONE_SCHEMA, *FFFD_LOOKALIKES, *PAIR_HALVES)() == []


async def test_rejecting_a_lone_surrogate_schema_at_registration_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema seam that cannot hold an unpaired surrogate (a UTF-8-only schema type) refuses
    the tool at registration."""
    real = runner.ToolDefinition

    def strict(*args: Any, **kwargs: Any) -> Any:
        _map(kwargs["parameters"], lambda s: s.encode("utf-8").decode("utf-8"))
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "ToolDefinition", strict)
    assert await _killed(*LONE_SCHEMA)() == sorted(LONE_SCHEMA)


async def test_replacing_schema_literals_with_fffd_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lossy schema conversion (lone surrogate -> U+FFFD) matches the wrong instances."""
    real = execute_module.PreparedArgumentsValidator
    monkeypatch.setattr(
        execute_module, "PreparedArgumentsValidator", lambda schema: real(_map(schema, _replace))
    )
    assert await _killed(*LONE_SCHEMA)() == sorted(LONE_SCHEMA)
    assert await _killed(*FFFD_LOOKALIKES)() == sorted(FFFD_LOOKALIKES)


def _split(s: str) -> str:
    """A UTF-16 code-unit view: each astral character as two separate surrogate code points."""
    out = []
    for ch in s:
        if ord(ch) > 0xFFFF:
            v = ord(ch) - 0x10000
            out += [chr(0xD800 + (v >> 10)), chr(0xDC00 + (v & 0x3FF))]
        else:
            out.append(ch)
    return "".join(out)


async def test_a_code_unit_pattern_search_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Unicode-mode pattern (searching UTF-16 code units) finds a pair's half inside it."""
    validators = dict(execute_module.PreparedArgumentsValidator.VALIDATORS)
    real = validators["pattern"]

    def code_unit_pattern(validator: Any, patrn: str, instance: Any, schema: Any) -> Any:
        if isinstance(instance, str) and not re.search(_split(patrn), _split(instance)):
            return real(validator, "(?!)", instance, schema)  # the real keyword's own error
        return iter(())

    validators["pattern"] = code_unit_pattern
    monkeypatch.setattr(execute_module.PreparedArgumentsValidator, "VALIDATORS", validators)
    assert await _killed(*PAIR_HALVES)() == sorted(PAIR_HALVES)


async def test_normalizing_instance_property_names_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property-name normalization of the instance (lone surrogate keys -> U+FFFD) stops an exact
    schema property name from matching."""
    original = execute_module._prepare
    monkeypatch.setattr(
        execute_module,
        "_prepare",
        lambda d, a: {_replace(k): v for k, v in original(d, a).items()},
    )
    keyed = ("properties-required/lone-high/lone-high", "property-names-const/lone-high/lone-high")
    assert await _killed(*keyed)() == sorted(keyed)


# Pi accepts: no Unicode-mode match of a pair's half inside a key holding the pair
PATTERN_PROPERTY_HALVES = (
    "pattern-properties-unanchored/pair-high-half/pair",
    "pattern-properties-unanchored/pair-low-half/pair",
)
PATTERN_PROPERTY_GUARD = (  # Pi rejects: a genuinely unpaired surrogate matches where it occurs
    "pattern-properties-unanchored/lone-high/pair-then-lone-high",
)


async def test_a_code_unit_pattern_properties_matcher_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L05-D001-R001: `patternProperties` has its own keyword implementation. A code-unit key
    matcher (the `pattern` keyword left correct) finds a pair's half inside a key holding the pair
    and wrongly applies the subschema; the genuine-unpaired guard stays correct under it."""
    assert await _killed(*PATTERN_PROPERTY_HALVES, *PATTERN_PROPERTY_GUARD)() == []
    validators = dict(execute_module.PreparedArgumentsValidator.VALIDATORS)

    def code_unit_pattern_properties(
        validator: Any, pattern_properties: dict[str, Any], instance: Any, schema: Any
    ) -> Any:
        if not isinstance(instance, dict):
            return
        for pattern, subschema in pattern_properties.items():
            for key, value in instance.items():
                if re.search(_split(pattern), _split(key)):
                    yield from validator.descend(value, subschema, path=key, schema_path=pattern)

    validators["patternProperties"] = code_unit_pattern_properties
    monkeypatch.setattr(execute_module.PreparedArgumentsValidator, "VALIDATORS", validators)
    assert await _killed(*PATTERN_PROPERTY_HALVES)() == sorted(PATTERN_PROPERTY_HALVES)
    assert await _killed(*PATTERN_PROPERTY_GUARD)() == []
    assert await _killed(*PAIR_HALVES)() == []  # the `pattern` keyword itself is untouched
