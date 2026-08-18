"""Stream chunks carry partials so a consumer can render any prefix."""

from collections.abc import AsyncIterator

import pytest

from minion_agent.llm.content import TextBlock
from minion_agent.llm.errors import AdapterProtocolError
from minion_agent.llm.messages import AssistantMessage, StopReason, Usage
from minion_agent.llm.stream import (
    StreamChunk,
    StreamDone,
    StreamError,
    StreamStart,
    TextDelta,
    collect,
)


def _message(text: str, reason: StopReason = StopReason.STOP) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text=text),),
        stop_reason=reason,
        usage=Usage(),
        model="mock-1",
        provider="mock",
        timestamp=1,
    )


async def _stream(*chunks: StreamChunk) -> AsyncIterator[StreamChunk]:
    for chunk in chunks:
        yield chunk


async def test_collect_returns_the_settled_message() -> None:
    partial = _message("", StopReason.PENDING)
    final = _message("hello")

    result = await collect(
        _stream(
            StreamStart(partial=partial),
            TextDelta(content_index=0, delta="hello", partial=partial),
            StreamDone(message=final, partial=final),
        )
    )

    assert result is final


async def test_collect_returns_an_error_message_without_raising() -> None:
    """The stream never raises once returned; failures ride it (spec section 4)."""
    failed = _message("", StopReason.ERROR)

    result = await collect(
        _stream(StreamError(reason=StopReason.ERROR, message=failed, partial=failed))
    )

    assert result.stop_reason is StopReason.ERROR


async def test_collect_on_an_empty_stream_raises_a_protocol_error() -> None:
    """An adapter that yields nothing violated its contract; that is a bug in
    the adapter, not an in-band model failure, so it raises."""
    with pytest.raises(AdapterProtocolError, match="terminal chunk"):
        await collect(_stream())


async def test_deltas_carry_the_partial_message() -> None:
    partial = _message("he", StopReason.PENDING)
    chunk = TextDelta(content_index=0, delta="llo", partial=partial)

    assert chunk.partial.stop_reason is StopReason.PENDING
    assert chunk.delta == "llo"
