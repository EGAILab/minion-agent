"""System prompt assembly (Layer 14, WP-14.2; spec/harness.md WP-14.2)."""

from .composer import PromptComposer, PromptConfiguration, compose_prompt
from .formatting import (
    format_skill_invocation,
    format_skills_block,
    format_tools_section,
    normalize_guidelines,
    normalize_snippet,
)

__all__ = [
    "PromptComposer",
    "PromptConfiguration",
    "compose_prompt",
    "format_skill_invocation",
    "format_skills_block",
    "format_tools_section",
    "normalize_guidelines",
    "normalize_snippet",
]
