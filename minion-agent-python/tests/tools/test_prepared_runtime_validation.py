"""L0506-D001 (`TOOL-041`) validation of prepared runtime numbers for a pydantic-model tool: the
same finite-only rule for declared numeric positions (`L0506-D001-I001`), pydantic staying the
authority for everything else."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, Literal, TypedDict, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic.errors import PydanticInvalidForJsonSchema

from minion_agent.tools import execute as execute_module
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


class NumericFirst(BaseModel):
    items: list[float] | list[Any]


class OpenFirst(BaseModel):
    items: list[Any] | list[float]


class ModelFirst(BaseModel):
    thing: Inner | dict[str, Any]


class DictFirst(BaseModel):
    thing: dict[str, Any] | Inner


class WithLiteral(BaseModel):
    mode: Literal["a", "b"] = "a"
    limit: float = 0.0


class Node(BaseModel):
    value: float
    child: Node | None = None


class Left(BaseModel):
    value: float
    right: Right | None = None


class Right(BaseModel):
    value: float
    left: Left | None = None


Node.model_rebuild()
Left.model_rebuild()


def _tool(model: type[BaseModel]) -> ToolDefinition:
    return ToolDefinition(name="t", label="t", description="d", parameters=model, execute=_never)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize(
    ("model", "arguments"),
    [
        (NumericFirst, lambda v: {"items": [v]}),  # docs #90 review: numeric branch first
        (OpenFirst, lambda v: {"items": [v]}),
        (ModelFirst, lambda v: {"thing": {"y": v}}),  # the dict[str, Any] alternative accepts it
        (DictFirst, lambda v: {"thing": {"y": v}}),
    ],
    ids=[
        "list-numeric-or-list-any",
        "list-any-or-list-numeric",
        "model-or-dict-any",
        "dict-any-or-model",
    ],
)
def test_an_unconstrained_alternative_accepts_in_either_order(
    model: type[BaseModel], arguments: Any, value: float
) -> None:
    """CE-L0506-D001-I001-01: a union accepts a non-finite value when SOME alternative accepts the
    complete value (pinned Pi's anyOf), whatever the branch order."""
    _validate(_tool(model), arguments(value))


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_every_depth_of_a_recursive_model_is_finite_only(value: float) -> None:
    with pytest.raises(ArgumentValidationError, match="finite number"):
        _validate(_tool(Node), {"value": 1.0, "child": {"value": 2.0, "child": {"value": value}}})
    with pytest.raises(ArgumentValidationError, match="finite number"):
        _validate(_tool(Left), {"value": 1.0, "right": {"value": 2.0, "left": {"value": value}}})
    _validate(_tool(Node), {"value": 1.0, "child": {"value": -0.0}})


def test_a_literal_field_is_left_as_declared() -> None:
    assert _validate(_tool(WithLiteral), {"mode": "b", "limit": 2.0})["mode"] == "b"
    with pytest.raises(ArgumentValidationError, match="finite number"):
        _validate(_tool(WithLiteral), {"mode": "b", "limit": math.inf})


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_numeric_or_string_rejects_non_finite(value: float) -> None:
    """Pinned Pi ACCEPTS these by coercing the number into the string branch ("Infinity");
    Minion's certified Layer 06 does not reproduce TypeBox coercion (disclosed divergence), and
    rejects -- exactly as its JSON-Schema path does (CE-L0506-D001-I001-01)."""

    class Either(BaseModel):
        field: float | str

    with pytest.raises(ArgumentValidationError, match="finite number"):
        _validate(_tool(Either), {"field": value})


# CE-L0506-D001-I001-01 negative controls: single-point mutants of the finite rule, each killed by a
# witness above. A witness is (model, arguments, accepted?) -- pinned Pi's verdict.
WITNESSES: list[tuple[type[BaseModel], dict[str, Any], bool]] = [
    (Limits, {"limit": math.inf}, False),
    (Limits, {"nullable": math.inf}, False),
    (Limits, {"items": [math.inf]}, False),
    (Limits, {"inner": {"y": math.inf}}, False),
    (Limits, {"box": {"w": math.inf}}, False),
    (Limits, {"point": {"x": math.inf}}, False),
    (Limits, {"loose": math.inf}, True),
    (Limits, {"loose_union": math.inf}, True),
    (Limits, {"extra": math.inf}, True),
    (NumericFirst, {"items": [math.inf]}, True),
    (OpenFirst, {"items": [math.inf]}, True),
    (ModelFirst, {"thing": {"y": math.inf}}, True),
    (Node, {"value": 1.0, "child": {"value": math.inf}}, False),
]


def _accepted(model: type[BaseModel], arguments: dict[str, Any]) -> bool:
    try:
        _validate(_tool(model), arguments)
    except ArgumentValidationError:
        return False
    return True


def _killed() -> bool:
    return any(_accepted(model, args) is not expected for model, args, expected in WITNESSES)


def test_every_negative_control_witness_holds_unmutated() -> None:
    assert not _killed()


def _sweep(model: type[BaseModel], arguments: Any) -> None:
    del model
    if not math.isfinite(sum(v for v in _floats(arguments))):
        raise ArgumentValidationError("non-finite anywhere")


def _floats(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [f for item in value.values() for f in _floats(item)]
    if isinstance(value, list):
        return [f for item in value for f in _floats(item)]
    return [value] if isinstance(value, float) else []


def _top_level_number_schema_only(model: type[BaseModel], arguments: dict[str, Any]) -> None:
    try:
        properties = model.model_json_schema().get("properties", {})
    except PydanticInvalidForJsonSchema:  # an opaque field: no schema, so nothing is filtered
        properties = {}
    for name, value in arguments.items():
        if properties.get(name, {}).get("type") == "number" and not math.isfinite(value):
            raise ArgumentValidationError("non-finite number")


@pytest.mark.parametrize(
    "mutant",
    [
        pytest.param(lambda model, arguments: None, id="no-finite-check"),
        pytest.param(_sweep, id="whole-model-non-finite-sweep"),
        pytest.param(_top_level_number_schema_only, id="json-schema-number-fields-only"),
    ],
)
def test_finite_rule_mutants_are_killed(monkeypatch: pytest.MonkeyPatch, mutant: Any) -> None:
    monkeypatch.setattr(execute_module, "_reject_declared_non_finite", mutant)
    assert _killed()


def test_first_union_member_only_mutant_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first-match rule: a union is judged by its first member alone (Codex's
    `list[float] | list[Any]` pair kills it)."""
    original = execute_module._shape

    def first_member(annotation: Any, finite: bool) -> Any:
        if get_origin(annotation) in (Union, UnionType):
            return original(get_args(annotation)[0], finite)
        return original(annotation, finite)

    monkeypatch.setattr(execute_module, "_shape", first_member)
    monkeypatch.setattr(execute_module, "_SHAPES", {})
    assert _killed()
