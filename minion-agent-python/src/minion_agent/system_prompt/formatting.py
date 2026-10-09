"""Model-visible text for skills and tool metadata (Layer 14, WP-14.2; spec/harness.md WP-14.2).

- `HAR-002` `format_skills_block`: pinned Pi's harness `formatSkillsForSystemPrompt`.
- `HAR-014` `format_skill_invocation`: pinned Pi's harness `formatSkillInvocation`, with
  `dirnameEnvPath`.
- `HAR-018` `normalize_snippet`, `normalize_guidelines`, `format_tools_section`: pinned Pi's coding
  agent normalizers and its "Available tools" / "Guidelines" rendering, minus its product text.

Strings are Unicode scalar-value strings (Owner decision `WP142-R001`). Code-unit rules (the
`dirname` indices) operate on UTF-16 code units, as JavaScript does.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..auth.js_json import js_trim
from ..skills import Skill
from ..tools.builtin._utf16 import from_units, to_units
from ..tools.definition import ToolDefinition

_SKILLS_PREAMBLE = (
    "The following skills provide specialized instructions for specific tasks.",
    "Read the full skill file when the task matches its description.",
    "When a skill file references a relative path, resolve it against the skill directory "
    "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
    "",
    "<available_skills>",
)


def _escape_xml(value: str) -> str:
    """Pi `escapeXml`: exactly `& < > " '`, in this order; nothing else."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_block(skills: Iterable[Skill]) -> str:
    """`HAR-002`: the available-skills block, or `""` when no skill is model-visible. Input order is
    kept; nothing is sorted or deduplicated; `location` is the addressed `file_path` as given."""
    visible = [skill for skill in skills if skill.disable_model_invocation is not True]
    if not visible:
        return ""
    lines = list(_SKILLS_PREAMBLE)
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _dirname(path: str) -> str:
    """Pi `dirnameEnvPath`, on UTF-16 code units: strip trailing `/` and `\\`; the last separator
    of either kind at `i`; `i == 2` after a drive colon -> the first 3 units; `i <= 0` -> `/`."""
    units = to_units(path).rstrip("/\\")
    i = max(units.rfind("/"), units.rfind("\\"))
    if i == 2 and units[1] == ":":
        return from_units(units[:3])
    if i <= 0:
        return "/"
    return from_units(units[:i])


def format_skill_invocation(skill: Skill, additional_instructions: str | None = None) -> str:
    """`HAR-014`: the explicit invocation text, with no escaping. Non-empty additional
    instructions follow after a blank line; absent or empty ones add nothing."""
    block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {_dirname(skill.file_path)}.\n\n"
        f"{skill.content}\n</skill>"
    )
    return f"{block}\n\n{additional_instructions}" if additional_instructions else block


def _collapse_js_whitespace(text: str) -> str:
    """Each run of JS `\\s` characters (ECMA-262 WhiteSpace and LineTerminator, the set `trim`
    removes) becomes one space."""
    out: list[str] = []
    in_run = False
    for ch in text:
        if js_trim(ch) == "":
            if not in_run:
                out.append(" ")
            in_run = True
        else:
            out.append(ch)
            in_run = False
    return "".join(out)


def normalize_snippet(text: str | None) -> str | None:
    """Pi `_normalizePromptSnippet`: absent or empty -> none; runs of CR/LF -> a space; runs of JS
    whitespace -> a space; JS trim; empty -> none."""
    if not text:
        return None
    one_line = _collapse_js_whitespace(_collapse_line_breaks(text))
    trimmed = js_trim(one_line)
    return trimmed or None


def _collapse_line_breaks(text: str) -> str:
    """`.replace(/[\\r\\n]+/g, " ")`."""
    out: list[str] = []
    in_run = False
    for ch in text:
        if ch in "\r\n":
            if not in_run:
                out.append(" ")
            in_run = True
        else:
            out.append(ch)
            in_run = False
    return "".join(out)


def normalize_guidelines(guidelines: Sequence[str] | None) -> list[str]:
    """Pi `_normalizePromptGuidelines`: JS-trim each, drop empty ones, keep the first occurrence of
    each exact string, in order. Interior text (including blank lines) is kept."""
    if not guidelines:
        return []
    unique: dict[str, None] = {}
    for guideline in guidelines:
        normalized = js_trim(guideline)
        if normalized:
            unique.setdefault(normalized, None)
    return list(unique)


def format_tools_section(tools: Iterable[ToolDefinition]) -> str:
    """`HAR-018`: the opt-in tools section over `tools`, in order. "Available tools:" lists each
    tool with a snippet; "Guidelines:" lists each guideline once, first occurrence wins across
    tools. A block with no lines is omitted, and with both omitted the section is `""`."""
    snippet_lines: list[str] = []
    guidelines: dict[str, None] = {}
    for tool in tools:
        snippet = normalize_snippet(tool.prompt_snippet)
        if snippet:
            snippet_lines.append(f"- {tool.name}: {snippet}")
        for guideline in normalize_guidelines(tool.prompt_guidelines):
            guidelines.setdefault(guideline, None)
    blocks: list[str] = []
    if snippet_lines:
        blocks.append("Available tools:\n" + "\n".join(snippet_lines))
    if guidelines:
        blocks.append("Guidelines:\n" + "\n".join(f"- {g}" for g in guidelines))
    return "\n\n".join(blocks)
