"""Executes agent conformance scenarios against the composed stack.

Asserts the two observable surfaces section 8 names: the derived Pi event
stream and the log's message projection. Nothing here inspects loop internals.

One adapter is built directly from the scenario rather than through
`mock_adapter_plugin`. The plugin's config model carries text only, and a
scenario needs tool calls, truncation, and post-terminal chunks; registering
two adapters for one model -- as an earlier draft did -- would leave which of
them answers a request dependent on registration order.
"""

from __future__ import annotations

from typing import Any

from minion_agent.agent.envelope import ClaimPolicy
from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.plugin import agents_plugin, tools_plugin
from minion_agent.agent.projection import TurnEnd, event_names, project
from minion_agent.agent_loop import agent_loop_plugin
from minion_agent.llm import (
    AssistantMessage,
    ContentBlock,
    ModelId,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    text_of,
)
from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.llm.plugin import llm_plugin
from minion_agent.runtime import Context
from minion_agent.session import derive_messages
from minion_agent.session.service import session_plugin

_ROLE = {
    UserMessage: "user",
    AssistantMessage: "assistant",
    ToolResultMessage: "tool_result",
}


def _block(raw: dict[str, Any]) -> ContentBlock:
    if raw["type"] == "tool_call":
        return ToolCallBlock(id=raw["id"], name=raw["name"], arguments=raw.get("arguments", {}))
    return TextBlock(text=raw.get("text", ""))


def _script(document: dict[str, Any]) -> list[ScriptedResponse]:
    """Translate the scripted provider responses into adapter form."""
    return [
        ScriptedResponse(
            content=tuple(_block(raw) for raw in response.get("content", [])),
            stop_reason=StopReason(response["stop_reason"]),
            error_message=response.get("error_message"),
            truncated=response.get("truncated", False),
            chunks_after_terminal=response.get("chunks_after_terminal", 0),
        )
        for response in document["provider_script"]
    ]


def _stub(text: str) -> Any:
    def run(args: dict[str, Any]) -> str:
        return text

    return run


async def run_agent_scenario(document: dict[str, Any]) -> dict[str, Any]:
    """Run the scenario and return its observable surfaces."""
    ctx = Context()
    await ctx.plugin(session_plugin)
    await ctx.plugin(llm_plugin)
    await ctx.plugin(tools_plugin)
    await ctx.plugin(agents_plugin)
    await ctx.plugin(agent_loop_plugin)
    ctx.llm.register(MockAdapter(_script(document)))

    for name, stub in document.get("tools", {}).items():
        ctx.tools.register(name, _stub(stub.get("result", {}).get("text", "")))

    config = document.get("config", {})
    handle = ctx.agents.create(
        "scenario",
        AgentDefinition(
            name="scenario",
            model=ModelId("mock", config.get("model", "mock-1")),
            system=config.get("system", ""),
            max_steps=config.get("max_steps", 16),
        ),
    )
    loop = ctx.agent_loop.for_instance(handle.instance)
    if "next_turn_policy" in config:
        loop.next_turn_policy = ClaimPolicy(config["next_turn_policy"])

    error: Exception | None = None
    for step in document["steps"]:
        for alias in ("followup", "steer", "inject"):
            if alias in step:
                spec = step[alias]
                text = spec if isinstance(spec, str) else spec["text"]
                origin = None if isinstance(spec, str) else spec.get("origin")
                message = UserMessage(content=(TextBlock(text=text),), timestamp=1)
                getattr(handle.instance.inbox, alias)(message, origin=origin)
        if step.get("await_idle"):
            try:
                await loop.run_until_idle()
            except Exception as raised:
                error = raised
                break

    events = project(handle.instance.log)
    messages = derive_messages(handle.instance.log)
    return {
        "events": event_names(events),
        "messages": [{"role": _ROLE[type(m)], "text": text_of(m)} for m in messages],
        "causes": [list(event.causes) for event in events if isinstance(event, TurnEnd)],
        "assistant_stop_reasons": [
            m.stop_reason.value for m in messages if isinstance(m, AssistantMessage)
        ],
        "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
    }
