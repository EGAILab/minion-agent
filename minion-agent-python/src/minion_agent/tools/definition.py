"""Tool definitions: name, schema, execution, and batching mode.

Parameters are a pydantic model exported as JSON Schema, which keeps the
model-facing contract language-neutral (design spec section 7) while Python
callers still get validation for free.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from ..llm import ToolSchema
from .result import ToolResult

type ToolUpdate = Callable[[str], None]
"""Report a partial result. Live only -- partial output never reaches a model."""

type ToolFn = Callable[..., Awaitable[ToolResult | str] | ToolResult | str]
"""Called with the validated arguments, and with an `update` callback when the
tool declares a second parameter."""


class ExecutionMode(StrEnum):
    """Whether a tool may overlap with others in the same batch.

    `PARALLEL` is the default because most tools are safe to overlap and
    `SEQUENTIAL` is the claim that carries consequences: under pi's contagion
    rule one sequential tool serializes the whole batch (design spec section 6).
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One registered tool."""

    name: str
    description: str
    parameters: type[BaseModel] | None
    execute: ToolFn
    mode: ExecutionMode = ExecutionMode.PARALLEL

    def schema(self) -> ToolSchema:
        """The model-facing schema for this tool.

        A tool with no parameter model still publishes an empty object schema
        rather than nothing: a model told a tool has no schema has no defined
        way to call it.
        """
        parameters = (
            dict(_EMPTY_SCHEMA) if self.parameters is None else self.parameters.model_json_schema()
        )
        return ToolSchema(name=self.name, description=self.description, parameters=parameters)
