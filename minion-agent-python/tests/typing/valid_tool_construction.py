"""Permanent static-type evidence for the Layer-05 (Tool model + registry) public vocabulary.

Not a pytest test: mypy checking this module IS the test, matching `valid_message_construction.py`'s
own established convention. This module's only job is to fail `mypy --strict` if any of these
frozen-vocabulary constructions ever stop type-checking.

Run explicitly (not part of the default `mypy` gate, which is scoped to `src/minion_agent` only):

    mypy src/minion_agent tests/typing/valid_tool_construction.py

Never imported or executed by pytest.
"""

from __future__ import annotations

from minion_agent.llm import GrammarConstrainedSampling, JsonSchemaConstrainedSampling
from minion_agent.tools.definition import ExecutionMode, ToolDefinition

_EMPTY_SCHEMA = {"type": "object", "properties": {}}

# `label` is required (pinned Pi `AgentTool.label`, no `?`) -- TOOL-F001.
_labeled: ToolDefinition = ToolDefinition(
    name="echo",
    description="repeat",
    parameters=_EMPTY_SCHEMA,
    execute=lambda args: "ok",
    label="Echo",
)

# A parameterless tool still supplies the explicit empty-object schema (`L05-R005`); `parameters`
# is required and missing/`None` are not shorthand for it.
_parameterless: ToolDefinition = ToolDefinition(
    name="noop",
    description="do nothing",
    parameters=_EMPTY_SCHEMA,
    execute=lambda args: "ok",
    label="Noop",
)

# `mode` defaults to `None` (no per-tool override), not a concrete `ExecutionMode` -- TOOL-F004.
_default_mode: ToolDefinition = ToolDefinition(
    name="a", description="a", parameters=_EMPTY_SCHEMA, execute=lambda args: "ok", label="A"
)
_explicit_sequential: ToolDefinition = ToolDefinition(
    name="b",
    description="b",
    parameters=_EMPTY_SCHEMA,
    execute=lambda args: "ok",
    label="B",
    mode=ExecutionMode.SEQUENTIAL,
)

# `constrained_sampling`: absent (`None`) | `False` | a config variant -- TOOL-F005.
_constrained_false: ToolDefinition = ToolDefinition(
    name="c",
    description="c",
    parameters=_EMPTY_SCHEMA,
    execute=lambda args: "ok",
    label="C",
    constrained_sampling=False,
)
_constrained_json_schema: ToolDefinition = ToolDefinition(
    name="d",
    description="d",
    parameters=_EMPTY_SCHEMA,
    execute=lambda args: "ok",
    label="D",
    constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
)
_constrained_grammar: ToolDefinition = ToolDefinition(
    name="e",
    description="e",
    parameters=_EMPTY_SCHEMA,
    execute=lambda args: "ok",
    label="E",
    constrained_sampling=GrammarConstrainedSampling(openai_lark="start: WORD+"),
)

# `prepare_arguments` -- field/signature only (TOOL-F002); Layer 05 does not certify invocation.
_with_prepare_arguments: ToolDefinition = ToolDefinition(
    name="f",
    description="f",
    parameters=_EMPTY_SCHEMA,
    execute=lambda args: "ok",
    label="F",
    prepare_arguments=lambda args: dict(args),
)

# "Object-valued" is the schema's own JSON representation (a mapping), not a requirement that the
# schema describe an object instance (`L05-R005`) -- pinned Pi's `Tool<TParameters extends
# TSchema>` is generic over TypeBox's whole `TSchema` domain, not narrowed to `TObject`.
_non_object_instance: ToolDefinition = ToolDefinition(
    name="g", description="g", parameters={"type": "string"}, execute=lambda args: "ok", label="G"
)
_top_level_combinator: ToolDefinition = ToolDefinition(
    name="h",
    description="h",
    parameters={"oneOf": [{"type": "string"}, {"type": "number"}]},
    execute=lambda args: "ok",
    label="H",
)
