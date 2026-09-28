"""`L13-WP131-RUST-I001` canonical-serialization witness: the canonical image scenarios pass only if
the binding's canonical Layer-02 serialization of the result is exactly the canonical base64 of the
final bytes -- decoded-byte equality alone is not accepted. Negative controls: a serializer emitting
line-wrapped (MIME-style) or unpadded base64 of the SAME bytes must fail the same scenario."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
import yaml

from . import builtin_tool_runner
from .builtin_tool_runner import run_builtin_tool_scenario

SCENARIO = (
    Path(__file__).resolve().parents[3]
    / "conformance"
    / "agent"
    / "builtin-read-image-bmp-converts-and-resizes.yaml"
)


async def _image_outcomes() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    document = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    outcomes = await run_builtin_tool_scenario(document)
    pairs = [(o["observed"]["image"], o["expected"]["image"]) for o in outcomes]
    return [(observed, expected) for observed, expected in pairs if expected is not None]


async def test_the_binding_serializes_the_canonical_base64_of_the_exact_bytes() -> None:
    pairs = await _image_outcomes()
    assert pairs
    for observed, expected in pairs:
        assert observed == expected
        assert observed["canonical_base64"] is True


@pytest.mark.parametrize(
    "non_canonical",
    [
        pytest.param(lambda data: base64.encodebytes(data).decode("ascii"), id="line-wrapped"),
        pytest.param(
            lambda data: base64.b64encode(data).decode("ascii").rstrip("="), id="unpadded"
        ),
    ],
)
async def test_a_non_canonical_serialization_of_the_same_bytes_fails(
    monkeypatch: pytest.MonkeyPatch, non_canonical: Any
) -> None:
    real = builtin_tool_runner._encode_block

    def serialize(block: Any) -> dict[str, Any]:
        encoded = real(block)
        encoded["data"] = non_canonical(base64.b64decode(encoded["data"]))
        return encoded

    monkeypatch.setattr(builtin_tool_runner, "_encode_block", serialize)
    pairs = await _image_outcomes()
    assert pairs
    assert all(observed["canonical_base64"] is False for observed, _ in pairs)
    assert all(observed != expected for observed, expected in pairs)
