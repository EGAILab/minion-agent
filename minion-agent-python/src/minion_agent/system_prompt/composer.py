"""System prompt composition (Layer 14, WP-14.2: `HAR-015`, `HAR-016`; spec/harness.md WP-14.2).

`HAR-015` (`MINION_ARCHITECTURAL_MAPPING`, Owner `PP-14-4`): a fixed, ordered list of sections --
base, the opt-in tools section, the contributed sections, then the skills block only when the tool
snapshot has a tool named exactly `read` -- with the non-empty ones joined by a blank line.

`HAR-016` (Owner `PP-14-9` Option A): `PromptComposer` is installed as the Layer 08 driver's prompt
assembler (`L08-D001`). The driver calls it for each request without a per-step override, with that
request's own tool snapshot, so the tools section, the `read` gate and the request's schemas come
from one snapshot. The composer never reads the tool registry.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..skills import Skill
from ..tools.definition import ToolDefinition
from .formatting import format_skills_block, format_tools_section


def compose_prompt(
    base: str,
    tools: Sequence[ToolDefinition],
    *,
    tools_section: bool,
    sections: Iterable[str],
    skills: Iterable[Skill],
) -> str:
    """`HAR-015`: the assembled system prompt for one tool snapshot."""
    parts = [base, format_tools_section(tools) if tools_section else "", *sections]
    if any(tool.name == "read" for tool in tools):
        parts.append(format_skills_block(skills))
    return "\n\n".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class PromptConfiguration:
    """The composer's whole configuration, replaced only as a whole (`HAR-016`). Constructing one
    copies the membership of `skills` and `sections`; the `Skill` records themselves are shared by
    identity and read when each request is assembled, as in Pi."""

    skills: tuple[Skill, ...] = ()
    tools_section: bool = False
    sections: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        *,
        skills: Iterable[Skill] = (),
        tools_section: bool = False,
        sections: Iterable[str] = (),
    ) -> PromptConfiguration:
        """A configuration from any iterables, their membership copied now."""
        return cls(skills=tuple(skills), tools_section=tools_section, sections=tuple(sections))


@dataclass(slots=True)
class PromptComposer:
    """The `L08-D001` prompt assembler for WP-14.2: `composer(base, tools) -> str`.

    `replace` swaps the configuration as a whole; the next request build that starts afterwards
    uses the new value, and one assembly reads exactly one value. Mutation and replacement belong
    to the execution context that drives the agent, between request builds (spec HAR-016,
    concurrency model)."""

    configuration: PromptConfiguration = field(default_factory=PromptConfiguration)

    def replace(self, configuration: PromptConfiguration) -> None:
        self.configuration = configuration

    def __call__(self, base: str, tools: tuple[ToolDefinition, ...]) -> str:
        configuration = self.configuration
        return compose_prompt(
            base,
            tools,
            tools_section=configuration.tools_section,
            sections=configuration.sections,
            skills=configuration.skills,
        )
