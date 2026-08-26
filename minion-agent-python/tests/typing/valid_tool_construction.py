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

# `label` is required (pinned Pi `AgentTool.label`, no `?`) -- TOOL-F001.
_labeled: ToolDefinition = ToolDefinition(
    name="echo", description="repeat", parameters=None, execute=lambda args: "ok", label="Echo"
)

# A parameterless tool still constructs; `schema()` synthesizes the empty-object shape at runtime.
_parameterless: ToolDefinition = ToolDefinition(
    name="noop", description="do nothing", parameters=None, execute=lambda args: "ok", label="Noop"
)

# `mode` defaults to `None` (no per-tool override), not a concrete `ExecutionMode` -- TOOL-F004.
_default_mode: ToolDefinition = ToolDefinition(
    name="a", description="a", parameters=None, execute=lambda args: "ok", label="A"
)
_explicit_sequential: ToolDefinition = ToolDefinition(
    name="b",
    description="b",
    parameters=None,
    execute=lambda args: "ok",
    label="B",
    mode=ExecutionMode.SEQUENTIAL,
)

# `constrained_sampling`: absent (`None`) | `False` | a config variant -- TOOL-F005.
_constrained_false: ToolDefinition = ToolDefinition(
    name="c",
    description="c",
    parameters=None,
    execute=lambda args: "ok",
    label="C",
    constrained_sampling=False,
)
_constrained_json_schema: ToolDefinition = ToolDefinition(
    name="d",
    description="d",
    parameters=None,
    execute=lambda args: "ok",
    label="D",
    constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
)
_constrained_grammar: ToolDefinition = ToolDefinition(
    name="e",
    description="e",
    parameters=None,
    execute=lambda args: "ok",
    label="E",
    constrained_sampling=GrammarConstrainedSampling(openai_lark="start: WORD+"),
)

# `prepare_arguments` -- field/signature only (TOOL-F002); Layer 05 does not certify invocation.
_with_prepare_arguments: ToolDefinition = ToolDefinition(
    name="f",
    description="f",
    parameters=None,
    execute=lambda args: "ok",
    label="F",
    prepare_arguments=lambda args: dict(args),
)
