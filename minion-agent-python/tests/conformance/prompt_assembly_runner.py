"""Thin adapter for the WP-14.2 canonical prompt-assembly scenarios (`conformance/agent/
prompt-assembly/`, `prompt-assembly-scenario.schema.json`).

It only builds the binding's real input records from a document and calls the binding's real
function for the document's `kind`; it formats nothing itself.
"""

from __future__ import annotations

from typing import Any

from minion_agent.skills import Skill
from minion_agent.system_prompt import (
    compose_prompt,
    format_skill_invocation,
    format_skills_block,
    format_tools_section,
)
from minion_agent.tools.definition import ToolDefinition


def skill(record: dict[str, Any]) -> Skill:
    return Skill(
        name=record["name"],
        description=record["description"],
        content=record["content"],
        file_path=record["file_path"],
        disable_model_invocation=record["disable_model_invocation"],
    )


def tool(record: dict[str, Any]) -> ToolDefinition:
    guidelines = record.get("guidelines")
    return ToolDefinition(
        name=record["name"],
        description=f"{record['name']} tool",
        parameters={"type": "object", "properties": {}},
        execute=lambda tool_call_id, args: "",
        label=record["name"],
        prompt_snippet=record.get("snippet"),
        prompt_guidelines=None if guidelines is None else tuple(guidelines),
    )


def run(case: dict[str, Any]) -> str:
    """The binding's output for one `prompt_assembly` case."""
    kind = case["kind"]
    data = case["input"]
    if kind == "skills_block":
        return format_skills_block([skill(s) for s in data["skills"]])
    if kind == "invocation":
        return format_skill_invocation(skill(data["skill"]), data.get("additional_instructions"))
    if kind == "tools_section":
        return format_tools_section([tool(t) for t in data["tools"]])
    if kind == "compose":
        return compose_prompt(
            data["base"],
            tuple(tool(t) for t in data["tools"]),
            tools_section=data["tools_section"],
            sections=data["sections"],
            skills=[skill(s) for s in data["skills"]],
        )
    raise ValueError(f"unknown prompt_assembly kind {kind!r}")
