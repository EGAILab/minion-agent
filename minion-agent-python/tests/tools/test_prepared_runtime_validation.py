"""L0506-D001 (`TOOL-041`) validation of prepared runtime numbers for a pydantic-model tool: the
same finite-only rule for declared numeric fields, pydantic staying the authority for everything
else."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.execute import ArgumentValidationError, _validate


class Limits(BaseModel):
    model_config = ConfigDict(extra="allow")
    limit: float
    count: int = 0


class Opaque:
    """A type pydantic can validate (arbitrary_types_allowed) but cannot put in a JSON schema."""


class WithOpaque(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    limit: float
    thing: Opaque | None = None


async def _never(tool_call_id: str, arguments: dict[str, Any]) -> Any:  # pragma: no cover
    raise AssertionError("not executed")


def _tool(model: type[BaseModel]) -> ToolDefinition:
    return ToolDefinition(name="t", label="t", description="d", parameters=model, execute=_never)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_a_declared_float_rejects_non_finite(value: float) -> None:
    with pytest.raises(ArgumentValidationError, match="is not of type 'number'"):
        _validate(_tool(Limits), {"limit": value})


def test_negative_zero_and_finite_values_pass() -> None:
    validated = _validate(_tool(Limits), {"limit": -0.0, "count": 3})
    assert validated["limit"] == 0 and math.copysign(1.0, validated["limit"]) < 0
    assert _validate(_tool(Limits), {"limit": 1e308})["limit"] == 1e308


def test_an_unconstrained_extra_keeps_its_non_finite_value() -> None:
    validated = _validate(_tool(Limits), {"limit": 1.0, "extra": math.inf})
    assert validated["extra"] == math.inf


def test_a_model_without_a_json_schema_is_left_to_pydantic() -> None:
    """Disclosed limit: the finite-only check needs the model's JSON schema."""
    validated = _validate(_tool(WithOpaque), {"limit": math.inf})
    assert validated["limit"] == math.inf
