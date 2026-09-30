"""L0506-D001 (`TOOL-041`) validation of prepared runtime numbers for a pydantic-model tool: the
same finite-only rule for declared numeric positions (`L0506-D001-I001`), pydantic staying the
authority for everything else."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict, Field

from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.execute import ArgumentValidationError, _validate


class Opaque:
    """A type pydantic validates (arbitrary_types_allowed) but cannot put in a JSON schema."""


class Inner(BaseModel):
    y: float


@dataclass
class Point:
    x: float


class Box(TypedDict):
    w: float


class Limits(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    limit: float = 0.0
    nullable: float | None = None
    annotated: Annotated[float, Field(description="a documented float")] = 0.0
    either: float | str = 0.0
    items: list[float] = []
    wrapped: list[Annotated[float, Field(description="an annotated item")]] = []
    pair: tuple[float, str] = (0.0, "")
    many: tuple[float, ...] = ()
    table: dict[str, float] = {}
    inner: Inner | None = None
    point: Point | None = None
    box: Box | None = None
    loose: Any = None
    loose_union: float | Any = 0.0
    thing: Opaque | None = None
    count: int = 0


async def _never(tool_call_id: str, arguments: dict[str, Any]) -> Any:  # pragma: no cover
    raise AssertionError("not executed")


TOOL = ToolDefinition(name="t", label="t", description="d", parameters=Limits, execute=_never)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize(
    "arguments",
    [
        lambda v: {"limit": v},
        lambda v: {"nullable": v},  # docs #200 I001 probe: an Optional numeric field
        lambda v: {"annotated": v},
        lambda v: {"either": v},
        lambda v: {"items": [1.0, v]},
        lambda v: {"wrapped": [v]},
        lambda v: {"pair": [v, "x"]},
        lambda v: {"many": [1.0, v]},
        lambda v: {"table": {"k": v}},
        lambda v: {"inner": {"y": v}},
        lambda v: {"point": {"x": v}},
        lambda v: {"box": {"w": v}},
        lambda v: {"limit": v, "thing": Opaque()},  # docs #200 I001 probe: an opaque field
    ],
    ids=[
        "float",
        "optional",
        "annotated",
        "union-with-str",
        "list",
        "annotated-item",
        "tuple-position",
        "tuple-var",
        "dict",
        "nested-model",
        "dataclass",
        "typeddict",
        "with-opaque-field",
    ],
)
def test_a_declared_numeric_position_rejects_non_finite(arguments: Any, value: float) -> None:
    with pytest.raises(ArgumentValidationError, match="Input should be a finite number"):
        _validate(TOOL, arguments(value))


@pytest.mark.parametrize(
    "arguments",
    [
        {"loose": math.inf},
        {"loose_union": math.nan},
        {"extra": -math.inf},
        {"loose": [math.inf]},
    ],
    ids=["any", "union-with-any", "undeclared-extra", "inside-any"],
)
def test_an_unconstrained_position_keeps_its_non_finite_value(arguments: dict[str, Any]) -> None:
    validated = _validate(TOOL, arguments)
    key = next(iter(arguments))
    assert repr(validated[key]) == repr(arguments[key])


def test_negative_zero_and_finite_values_pass() -> None:
    validated = _validate(TOOL, {"limit": -0.0, "nullable": 1e308, "items": [-0.0], "count": 3})
    assert math.copysign(1.0, validated["limit"]) < 0
    assert math.copysign(1.0, validated["items"][0]) < 0
    assert validated["nullable"] == 1e308


def test_finite_nested_structures_pass() -> None:
    validated = _validate(
        TOOL,
        {
            "inner": {"y": 1.0},
            "point": {"x": 2.0},
            "box": {"w": 3.0},
            "wrapped": [4.0],
            "loose_union": [5],  # no union member describes a list: nothing declared below it
        },
    )
    assert validated["inner"] == {"y": 1.0} and validated["wrapped"] == [4.0]
