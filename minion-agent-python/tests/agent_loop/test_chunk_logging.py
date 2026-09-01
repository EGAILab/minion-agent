"""Streaming deltas are logged for fidelity, never for derivation."""

from dataclasses import replace

from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.registry import AgentRegistry
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import (
    AssistantMessage,
    LlmService,
    ModelId,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    UserMessage,
    text_of,
)
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason, Usage
from minion_agent.llm.service import Request
from minion_agent.llm.stream import (
    AssistantStream,
    StreamDone,
    StreamStart,
    ThinkingDelta,
    ToolCallDelta,
)
from minion_agent.runtime import Context
from minion_agent.session import EventKind, SessionService, decode_message, derive_messages
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.registry import ToolRegistry

from .test_single_turn import _loop


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


async def test_text_deltas_are_logged() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="streamed"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    chunks = [e for e in loop.instance.log.events if e.kind == EventKind.ASSISTANT_CHUNK]
    partials = [decode_message(chunk.data["partial"]) for chunk in chunks]
    assert [text_of(partial) for partial in partials] == ["streamed"]


async def test_a_chunk_records_which_block_it_belongs_to() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="a"), TextBlock(text="b")), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    chunks = [e for e in loop.instance.log.events if e.kind == EventKind.ASSISTANT_CHUNK]
    assert [chunk.data["content_index"] for chunk in chunks] == [0, 1]


async def test_chunks_do_not_derive_into_model_history() -> None:
    """Log-only. A delta that projected would duplicate the message it is
    part of, and the model would read the answer twice."""
    loop = _loop(ScriptedResponse((TextBlock(text="once"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    chunks = [e for e in loop.instance.log.events if e.kind == EventKind.ASSISTANT_CHUNK]
    assert chunks, "setup sanity check: chunk logging must actually have fired"
    assert [text_of(m) for m in derive_messages(loop.instance.log)] == ["hello", "once"]


async def test_chunks_precede_the_message_they_assemble() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="x"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    kinds = [e.kind for e in loop.instance.log.events]
    assert kinds.index(EventKind.ASSISTANT_CHUNK) < kinds.index(EventKind.ASSISTANT_MESSAGE)


async def test_a_response_with_no_text_logs_no_chunks() -> None:
    control = _loop(ScriptedResponse((TextBlock(text="present"),), StopReason.STOP))
    control.instance.inbox.followup(_say("hello"))
    await control.run_until_idle()
    control_chunks = [e for e in control.instance.log.events if e.kind == EventKind.ASSISTANT_CHUNK]
    assert control_chunks, "setup sanity check: chunk logging must actually fire when text exists"

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert not [e for e in loop.instance.log.events if e.kind == EventKind.ASSISTANT_CHUNK]


class _DeltaAdapter:
    """Streams a thinking delta and a tool-call delta.

    The mock adapter deliberately emits text deltas only -- the conformance
    table for this task pins `tool-round-trip`'s tool-call step to add no
    `message_update` at all, so extending `MockAdapter` to stream tool-call
    deltas would contradict that pinned behaviour. `log_chunk`'s other two
    match arms (`ThinkingDelta`, `ToolCallDelta`) therefore need a stream that
    actually carries them, built directly against the `Adapter` protocol.
    """

    provider = "delta-mock"
    api = "mock"
    models = frozenset({"delta-1"})

    async def stream(self, request: Request) -> AssistantStream:
        pending = AssistantMessage(
            content=(),
            stop_reason=StopReason.PENDING,
            usage=Usage(),
            model=request.model.model,
            provider=request.model.provider,
            timestamp=0,
        )
        yield StreamStart(partial=pending)
        thinking_partial = replace(pending, content=(ThinkingBlock(thinking="hmm"),))
        yield ThinkingDelta(content_index=0, delta="hmm", partial=thinking_partial)
        toolcall_partial = replace(
            thinking_partial,
            content=(
                *thinking_partial.content,
                ToolCallBlock(id="t1", name="echo", arguments={}),
            ),
        )
        yield ToolCallDelta(content_index=1, delta="{}", partial=toolcall_partial)
        settled = AssistantMessage(
            content=(),
            stop_reason=StopReason.STOP,
            usage=Usage(),
            model=request.model.model,
            provider=request.model.provider,
            timestamp=1,
        )
        yield StreamDone(message=settled, partial=settled)


async def test_thinking_and_tool_call_deltas_are_logged_too() -> None:
    """Coverage for `log_chunk`'s other two delta arms, which no scenario in
    this plan's conformance suite ever exercises through the mock adapter.
    Each logged chunk's `partial` is the full assistant message assembled so
    far (`L08-R003`), not a raw delta string."""
    ctx = Context()
    declare_tools_events(ctx.events)
    sessions = SessionService()
    llm = LlmService()
    llm.register(_DeltaAdapter())
    registry = AgentRegistry(ctx=ctx, sessions=sessions)
    handle = registry.create(
        "room-a",
        AgentDefinition(name="ada", model=ModelId("delta-mock", "delta-1"), system="be helpful"),
    )
    loop = AgentLoop(
        instance=handle.instance,
        llm=llm,
        tools=ToolRegistry(),
        artifacts=sessions.artifacts,
    )
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    chunks = [e for e in loop.instance.log.events if e.kind == EventKind.ASSISTANT_CHUNK]
    assert [chunk.data["kind"] for chunk in chunks] == ["thinking_delta", "toolcall_delta"]

    partials = [decode_message(chunk.data["partial"]) for chunk in chunks]
    assert isinstance(partials[0].content[0], ThinkingBlock)
    assert partials[0].content[0].thinking == "hmm"
    assert any(isinstance(block, ToolCallBlock) for block in partials[1].content)
