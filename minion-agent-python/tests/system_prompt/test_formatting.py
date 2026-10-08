"""WP-14.2 formatting units (HAR-002, HAR-014, HAR-018). The canonical scenarios carry the bulk of
the byte evidence; these pin the edges the scenarios do not isolate."""

from __future__ import annotations

from typing import Any

import pytest

from minion_agent.skills import Skill
from minion_agent.system_prompt import (
    format_skill_invocation,
    format_skills_block,
    format_tools_section,
    normalize_guidelines,
    normalize_snippet,
)
from minion_agent.tools.definition import ToolDefinition

ASTRAL = chr(0x1F600)
NBSP, BOM, LS, IDS = chr(0xA0), chr(0xFEFF), chr(0x2028), chr(0x3000)


def _skill(file_path: str = "/s/a/SKILL.md", **fields: Any) -> Skill:
    values: dict[str, Any] = {
        "name": "a",
        "description": "d",
        "content": "body",
        "file_path": file_path,
        "disable_model_invocation": False,
    }
    values.update(fields)
    return Skill(**values)


def _tool(name: str, snippet: str | None = None, guidelines: tuple[str, ...] | None = None) -> Any:
    return ToolDefinition(
        name=name,
        description="",
        parameters={"type": "object", "properties": {}},
        execute=lambda tool_call_id, args: "",
        label=name,
        prompt_snippet=snippet,
        prompt_guidelines=guidelines,
    )


def _dirname_of(path: str) -> str:
    text = format_skill_invocation(_skill(path))
    return text.split("References are relative to ", 1)[1].split(".\n", 1)[0]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # pinned Pi formatSkillInvocation (Node v22.15.1, data/14-wp141/pinned/skills.ts): the
        # drive check and the separator index are on UTF-16 code units -- an astral character
        # before them shifts every index by one
        (f"{ASTRAL}:\\b.md", f"{ASTRAL}:"),
        (f"{ASTRAL}/b.md", ASTRAL),
        (f"a{ASTRAL}\\c.md", f"a{ASTRAL}"),
        (ASTRAL, "/"),
    ],
)
def test_dirname_indexes_utf16_code_units_like_pi(path: str, expected: str) -> None:
    assert _dirname_of(path) == expected


def test_disable_model_invocation_filters_only_exact_true() -> None:
    hidden = _skill(name="hidden", disable_model_invocation=True)
    shown = _skill(name="shown")
    block = format_skills_block([hidden, shown])
    assert "<name>shown</name>" in block and "hidden" not in block
    assert format_skills_block([hidden]) == ""


def test_snippet_normalization_edges() -> None:
    assert normalize_snippet(None) is None
    assert normalize_snippet("") is None
    assert normalize_snippet(f" {NBSP}{BOM} ") is None
    assert normalize_snippet(f"a\r\n\r\nb{LS}c{IDS}{IDS}d ") == "a b c d"


def test_guideline_normalization_keeps_interior_text_and_first_occurrences() -> None:
    assert normalize_guidelines(None) == []
    assert normalize_guidelines(()) == []
    assert normalize_guidelines([" x ", "", IDS, "x", "a\n\nb", "a\n\nb"]) == ["x", "a\n\nb"]


def test_tools_section_is_empty_without_metadata() -> None:
    assert format_tools_section([_tool("read"), _tool("x", snippet="  ")]) == ""


def test_tools_section_dedupes_guidelines_across_tools_in_tool_order() -> None:
    section = format_tools_section(
        [_tool("b", snippet="B", guidelines=("g2", "g1")), _tool("a", guidelines=("g1", "g3"))]
    )
    assert section == "Available tools:\n- b: B\n\nGuidelines:\n- g2\n- g1\n- g3"


def test_tool_prompt_metadata_never_reaches_the_schema() -> None:
    plain = _tool("t")
    rich = _tool("t", snippet="S", guidelines=("G",))
    assert rich.schema() == plain.schema()
