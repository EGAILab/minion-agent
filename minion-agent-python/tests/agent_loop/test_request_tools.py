"""The request carries the tools visible from the instance's scope."""

from minion_agent.llm import TextBlock, UserMessage
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import ArtifactStore, EventKind, reconstruct_tools
from minion_agent.tools.definition import ToolDefinition

from .test_single_turn import _loop_with_adapter


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _echo() -> ToolDefinition:
    return ToolDefinition(
        name="echo",
        description="repeat",
        parameters={"type": "object", "properties": {}},
        execute=lambda tool_call_id, args: "ok",
        label="Echo",
    )


async def test_a_request_offers_the_visible_tools() -> None:
    loop, adapter = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))
    loop.tools.register(_echo())
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert [tool.name for tool in adapter.requests[0].tools] == ["echo"]


async def test_a_request_with_no_tools_offers_none() -> None:
    """Empty is meaningful: the step genuinely offers nothing."""
    loop, adapter = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert adapter.requests[0].tools == ()


async def test_a_withdrawn_tool_leaves_the_next_request() -> None:
    """Registration is an effect; withdrawing it changes what the model is
    offered from the next step onward."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((), StopReason.STOP), ScriptedResponse((), StopReason.STOP)
    )
    withdraw = loop.tools.register(_echo())
    loop.instance.inbox.followup(_say("first"))
    await loop.run_until_idle()

    withdraw()
    loop.instance.inbox.followup(_say("second"))
    await loop.run_until_idle()

    assert [tool.name for tool in adapter.requests[0].tools] == ["echo"]
    assert adapter.requests[1].tools == ()


async def test_the_logged_header_reconstructs_the_dispatched_tools() -> None:
    """Model-visible means logged: the invariant has to cover tools, or it
    covers the prompt and quietly exempts half of what the model was told."""
    loop, adapter = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))
    store: ArtifactStore = loop.artifacts
    loop.tools.register(_echo())
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    header = next(e for e in loop.instance.log.events if e.kind == EventKind.REQUEST_HEADER)
    assert reconstruct_tools(header, store) == adapter.requests[0].tools
