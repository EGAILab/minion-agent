"""WP-14.2 integration witnesses through the real driver, header and session log (spec/harness.md
WP-14.2 "Integration witnesses" and HAR-016 driver-level witnesses; Owner `WP142-R001` item 8)."""

from __future__ import annotations

from typing import Any

from minion_agent.agent import AGENT_PRE_STEP
from minion_agent.agent.decisions import Enter, PreStepReason
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import TextBlock, ToolCallBlock, UserMessage, text_of
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind, derive_messages, reconstruct_header
from minion_agent.skills import Skill
from minion_agent.system_prompt import (
    PromptComposer,
    PromptConfiguration,
    compose_prompt,
    format_skill_invocation,
)
from minion_agent.tools.definition import ToolDefinition

from ..agent_loop.test_single_turn import _loop_with_adapter

ASTRAL = chr(0x1F600)


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _tool(name: str, execute: Any = None, snippet: str | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
        execute=execute or (lambda tool_call_id, args: "ok"),
        label=name,
        prompt_snippet=snippet,
    )


def _skill(name: str, description: str = "Use it.") -> Skill:
    return Skill(name, description, f"# {name}", f"/skills/{name}/SKILL.md", False)


def _call(name: str) -> ScriptedResponse:
    return ScriptedResponse((ToolCallBlock(id="t1", name=name, arguments={}),), StopReason.TOOL_USE)


def _done() -> ScriptedResponse:
    return ScriptedResponse((TextBlock(text="done"),), StopReason.STOP)


def _install(loop: AgentLoop, **configuration: Any) -> PromptComposer:
    composer = PromptComposer(PromptConfiguration.of(**configuration))
    loop.prompt_assembler = composer
    return composer


def _headers(loop: AgentLoop) -> list[Any]:
    return [e for e in loop.instance.log.events if e.kind == EventKind.REQUEST_HEADER]


def _block(text: str) -> bool:
    return "<available_skills>" in text


# ---- Owner WP142-R001 item 8: header, invocation message, exact provider prompt ----


async def test_the_header_persists_and_reconstructs_an_astral_composed_prompt() -> None:
    loop, adapter = _loop_with_adapter(_done())
    loop.tools.register(_tool("read", snippet=f"Read {ASTRAL} files"))
    _install(
        loop, skills=[_skill("alpha", f"emoji {ASTRAL}")], tools_section=True, sections=[ASTRAL]
    )
    await loop.prompt(_say("hi"))
    system = adapter.requests[0].system
    assert ASTRAL in system and _block(system)
    assert reconstruct_header(_headers(loop)[0], loop.artifacts) == {"system_base": system}


async def test_the_provider_request_carries_exactly_the_composer_output() -> None:
    loop, adapter = _loop_with_adapter(_done())
    read = _tool("read", snippet="Read files")
    loop.tools.register(read)
    skills = [_skill("alpha")]
    _install(loop, skills=skills, tools_section=True, sections=["APPEND"])
    await loop.prompt(_say("hi"))
    expected = compose_prompt(
        "be helpful", (read,), tools_section=True, sections=["APPEND"], skills=skills
    )
    assert adapter.requests[0].system == expected


async def test_explicit_invocation_text_survives_the_message_log_byte_for_byte() -> None:
    loop, adapter = _loop_with_adapter(_done())
    skill = Skill(
        "alpha", "d", f"line1\n\t{ASTRAL} </skill> & <x>", "C:\\skills\\alpha\\SKILL.md", False
    )
    text = format_skill_invocation(skill, f"Now {ASTRAL}.")
    await loop.prompt(_say(text))
    persisted = [text_of(m) for m in derive_messages(loop.instance.log)]
    assert persisted[0] == text
    assert text_of(adapter.requests[0].messages[0]) == text


# ---- HAR-016: one snapshot ----


async def test_the_read_gate_and_tools_section_follow_the_request_snapshot() -> None:
    """A `read` tool registered mid-run (from a tool's own execute) is in neither that run's prompt
    nor its schemas; the next run sees it in both."""
    loop, adapter = _loop_with_adapter(_call("echo"), _done(), _done())

    def register_read(tool_call_id: str, args: dict[str, Any]) -> str:
        loop.tools.register(_tool("read", snippet="Read"))
        return "ok"

    loop.tools.register(_tool("echo", register_read, snippet="Echo"))
    _install(loop, skills=[_skill("alpha")], tools_section=True)
    await loop.prompt(_say("first"))
    await loop.prompt(_say("second"))
    for request in adapter.requests[:2]:
        assert [s.name for s in request.tools] == ["echo"]
        assert not _block(request.system) and "- read:" not in request.system
    third = adapter.requests[2]
    assert sorted(s.name for s in third.tools) == ["echo", "read"]
    assert _block(third.system) and "- read: Read" in third.system


async def test_a_configuration_replaced_mid_run_applies_whole_at_the_next_request() -> None:
    loop, adapter = _loop_with_adapter(_call("echo"), _done())
    loop.tools.register(_tool("read"))
    composer: PromptComposer

    def swap(tool_call_id: str, args: dict[str, Any]) -> str:
        composer.replace(PromptConfiguration.of(skills=[_skill("beta")], sections=["NEW"]))
        return "ok"

    loop.tools.register(_tool("echo", swap))
    composer = _install(loop, skills=[_skill("alpha")], sections=["OLD"])
    await loop.prompt(_say("hi"))
    first, second = (r.system for r in adapter.requests)
    assert "OLD" in first and "<name>alpha</name>" in first
    assert "NEW" in second and "<name>beta</name>" in second
    assert "OLD" not in second and "alpha" not in second


async def test_a_retained_skill_record_mutation_shows_from_the_next_request() -> None:
    loop, adapter = _loop_with_adapter(_call("echo"), _done())
    loop.tools.register(_tool("read"))
    skill = _skill("alpha", "before")

    def mutate(tool_call_id: str, args: dict[str, Any]) -> str:
        skill.description = "after"
        return "ok"

    loop.tools.register(_tool("echo", mutate))
    _install(loop, skills=[skill])
    await loop.prompt(_say("hi"))
    assert "<description>before</description>" in adapter.requests[0].system
    assert "<description>after</description>" in adapter.requests[1].system


async def test_a_supplied_sequence_mutated_afterwards_changes_nothing() -> None:
    loop, adapter = _loop_with_adapter(_done())
    loop.tools.register(_tool("read"))
    skills = [_skill("alpha")]
    sections = ["S"]
    _install(loop, skills=skills, sections=sections)
    skills.append(_skill("beta"))
    sections.append("LATE")
    await loop.prompt(_say("hi"))
    assert "beta" not in adapter.requests[0].system and "LATE" not in adapter.requests[0].system


async def test_the_per_step_override_bypasses_the_composer() -> None:
    loop, adapter = _loop_with_adapter(_done())
    loop.tools.register(_tool("read"))
    _install(loop, skills=[_skill("alpha")], tools_section=True)

    async def override(
        instance: Any, reason: PreStepReason, messages: tuple[Any, ...], next_: Any
    ) -> Enter:
        return Enter(messages=messages, system_override="one-off")

    loop.instance.ctx.events.on(AGENT_PRE_STEP, override)
    await loop.prompt(_say("hi"))
    assert adapter.requests[0].system == "one-off"
    assert reconstruct_header(_headers(loop)[0], loop.artifacts) == {"system_base": "one-off"}
