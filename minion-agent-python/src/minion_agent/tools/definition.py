"""Tool definitions: name, schema, execution, and batching mode.

Parameters are normally a pydantic model exported as JSON Schema, which keeps
the model-facing contract language-neutral (design spec section 7) while
Python callers still get validation for free. A raw JSON Schema `dict` is also
first-class (`TOOL-F010`): the model-facing contract is the JSON Schema
itself, not any one host language's schema-authoring library, and canonical
conformance evidence must be able to construct a real `ToolDefinition` from a
language-neutral schema value without inventing a Pydantic model dynamically.
This module only *stores* the schema (Layer 05); Layer 06's `execute.py`
*validates* execution arguments against it -- for a raw `dict` via the general
`jsonschema` library, for a pydantic model via pydantic itself (`L06-R001`; an
earlier revision of this docstring said a raw-dict tool "bypasses Python-side
argument validation," which stopped being true once Layer 06 closed that gap).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from ..llm import ConstrainedSampling, ToolSchema
from .result import ToolResult

type ToolUpdate = Callable[[ToolResult], None]
"""Report a partial result. Live only -- partial output never reaches a model. Matches pinned Pi's
own `AgentToolUpdateCallback<T> = (partialResult: AgentToolResult<T>) => void` exactly (Layer 08,
`L08-R011`): `partialResult` is the SAME structured shape a tool's own final result is -- Pi's
`AgentToolResult<T>` has `content`/`details`/`usage`/`addedToolNames`/`terminate`, exactly Minion's
own `ToolResult` -- not a bare string. An earlier revision narrowed this to `Callable[[str], None]`,
a real payload reduction pinned Pi does not have: a tool that wants to report partial STRUCTURED
progress (e.g. partial `details` for UI rendering, matching Pi's own `AgentToolResult.details: T`)
had no way to do so. `execute.py::_execute_and_finalize`'s own `update` closure normalizes
`tool_call_id`/`tool_name` on the tool-supplied partial the same way it already does for the final
result -- a tool need not (and should not) stamp its own call's real id/name onto a partial."""

type ToolFn = Callable[..., Awaitable[ToolResult | str] | ToolResult | str]
"""Called with `(tool_call_id, validated_arguments)`, and with an `update` callback appended when
the tool declares a third parameter (arity-detected, see `execute.py::_wants_update`).

Target capability shape, matching pinned Pi's `AgentTool.execute` (`packages/agent/src/types.ts`):
`(tool_call_id, params, signal?, on_update?) -> AgentToolResult`. Layer 05 owns only this shape's
existence and its association with a registered tool. Layer 06 (`TOOL-017`) closes the
`tool_call_id` half of the gap `TOOL-F003` disclosed: every call now receives its own real
`tool_call_id` as the first positional argument, and `on_update` is realized too (arity-detected,
above). The `signal` (cancellation) parameter remains behaviorally unrealized in Python -- but the
cross-language state is asymmetric, not uniformly absent (`IR-L05/06-006`, corrected here; an
earlier revision of this docstring said "no `AbortSignal`-equivalent type exists anywhere in this
codebase yet, in either language," which was already false for Rust when it was written): certified
Rust Layer 05 already reserves a structural signal seam (`ToolExecutionSignal`,
`ToolExecutionRequest.signal` in `minion-agent-rust/crates/minion-agent/src/tools/definition.rs`)
without exercising cancellation behavior. Python has no `AbortSignal`-equivalent abstraction at
all yet; Rust has one, unused. Layer 06 certifies **non-cancelled** execution semantics only in
both languages; assurance Layer 09 owns cancellation/abort propagation, timing, and result
semantics, and can add that behavior without requiring Rust to discard or redesign its existing
signal-bearing capability seam."""

type PrepareArguments = Callable[[dict[str, Any]], dict[str, Any]]
"""Pi's optional `AgentTool.prepareArguments?: (args: unknown) => Static<TParameters>` --
a compatibility shim for raw tool-call arguments before schema validation. Layer 05 owns only its
existence, public signature, and association with a registered tool; Layer 06 owns when/whether it
actually runs in the pipeline (`TOOL-F002`)."""


class ExecutionMode(StrEnum):
    """Whether a tool may overlap with others in the same batch.

    `SEQUENTIAL` is the claim that carries consequences: under pi's contagion
    rule one sequential tool serializes the whole batch (design spec section 6).
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One registered tool."""

    name: str
    description: str
    parameters: type[BaseModel] | dict[str, Any]
    """The required object-valued JSON Schema representation (`TOOL-F010`): a pydantic model
    class, or a raw, language-neutral JSON Schema `dict`. Layer 05 only stores this value; Layer
    06's `execute.py` validates execution arguments against it before `execute` runs -- via
    pydantic for a model class, via the general `jsonschema` library for a raw `dict` (`L06-R001`;
    an earlier revision of this docstring said a raw `dict` was "not Python-validated," which
    stopped being true once Layer 06 closed that gap -- construction here never validates
    anything itself, regardless of representation). Required: missing/`None` is not a shorthand for
    "no parameters" (`L05-R005`) -- a tool that takes nothing still supplies the explicit empty
    schema `{"type": "object", "properties": {}}`. "Object-valued" describes the JSON
    *representation* of the schema itself (the value is a mapping) -- it does not require the
    schema to describe an object *instance* via a top-level `"type": "object"` keyword. Pinned
    Pi's `Tool<TParameters extends TSchema>` (`packages/ai/src/types.ts`) is generic over
    TypeBox's whole `TSchema` domain, not narrowed to `TObject`, so `{"type": "string"}` and a
    top-level `{"oneOf": [...]}` are equally valid tool parameter schemas (`L05-R005`, corrected
    after an earlier repair mistakenly required `parameters["type"] == "object"`)."""
    execute: ToolFn
    label: str
    """Human-readable label for UI display (pinned Pi `AgentTool.label`, required -- TOOL-F001)."""
    mode: ExecutionMode | None = None
    """Per-tool execution-mode override (pinned Pi `AgentTool.executionMode?`). `None` means "no
    per-tool preference stated" -- distinct from an explicit `PARALLEL` override -- and defers to
    whatever run-level default execution mode applies (Layer-06 territory). A tool's own
    `PARALLEL`/`SEQUENTIAL` value, when present, is what the batch contagion rule (`TOOL-001`)
    reads; `None` never contributes exclusivity."""
    constrained_sampling: ConstrainedSampling | Literal[False] | None = None
    """Pinned Pi `Tool.constrainedSampling?: false | ConstrainedSamplingConfig`. Preserved end to
    end into the model-facing schema (`schema()` below); provider-specific enforcement/fallback is
    Real Providers (assurance Layer 11) territory."""
    prepare_arguments: PrepareArguments | None = None
    """Pinned Pi `AgentTool.prepareArguments?`. Field/signature only -- Layer 05 does not certify
    when or whether the pipeline invokes it (`TOOL-F002`)."""

    def __post_init__(self) -> None:
        """Reject `None`/non-mapping `parameters` at construction, not only via typing
        (`L05-R005`): a dynamically-typed caller can still pass `None` or a JSON-Schema-spec
        boolean shorthand past `mypy`. This checks only that the value is *some* mapping -- the
        JSON representation "object-valued" actually requires -- never a particular JSON Schema
        keyword (e.g. a top-level `"type": "object"`). Pinned Pi's `Tool<TParameters extends
        TSchema>` accepts any TypeBox schema, not only object-instance schemas (`{"type":
        "string"}` and a top-level `{"oneOf": [...]}` are both valid); Layer 05 is not a JSON
        Schema validator and does not otherwise inspect nested keywords."""
        if isinstance(self.parameters, type) and issubclass(self.parameters, BaseModel):
            return
        if isinstance(self.parameters, dict):
            return
        raise TypeError(
            "ToolDefinition.parameters is required and must be a pydantic BaseModel subclass or "
            "an object-valued JSON Schema mapping -- missing/None and the JSON-Schema-spec "
            "boolean-shorthand forms are not accepted; pass {'type': 'object', 'properties': {}} "
            "explicitly for a no-argument tool (L05-R005)"
        )

    def schema(self) -> ToolSchema:
        """The model-facing schema for this tool. A raw JSON Schema `dict` publishes unchanged."""
        if isinstance(self.parameters, dict):
            parameters = self.parameters
        else:
            parameters = self.parameters.model_json_schema()
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=parameters,
            constrained_sampling=self.constrained_sampling,
        )
