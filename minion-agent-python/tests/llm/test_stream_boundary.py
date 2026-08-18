"""The post-return stream contract: nothing escapes iteration.

§4 is absolute after the boundary — "runtime streaming failures terminate the
stream with a final message". A truncated response is the ordinary case of
that, not an adapter bug, so it settles rather than raising.
"""

from collections.abc import AsyncIterator

import pytest

from minion_agent.llm import (
    LlmService,
    ModelId,
    Request,
    StopReason,
    TextBlock,
    UnknownModelError,
    collect,
    text_of,
)
from minion_agent.llm.messages import AssistantMessage, Usage
from minion_agent.llm.stream import StreamChunk, StreamDone, StreamError, StreamStart, TextDelta


def _partial(text: str, reason: StopReason = StopReason.PENDING) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text=text),),
        stop_reason=reason,
        usage=Usage(input=3, output=4),
        model="mock-1",
        provider="mock",
        timestamp=1,
    )


class _Adapter:
    """Yields exactly the chunks it was given, however malformed."""

    provider = "mock"
    models = frozenset({"mock-1"})

    def __init__(self, *chunks: StreamChunk) -> None:
        self._chunks = chunks

    def stream(self, request: Request) -> AsyncIterator[StreamChunk]:
        async def run() -> AsyncIterator[StreamChunk]:
            for chunk in self._chunks:
                yield chunk

        return run()


def _service(*chunks: StreamChunk) -> LlmService:
    service = LlmService()
    service.register(_Adapter(*chunks))
    return service


def _request() -> Request:
    return Request(model=ModelId("mock", "mock-1"), system="", messages=())


async def test_premature_eof_settles_instead_of_raising() -> None:
    partial = _partial("half a sen")
    service = _service(
        StreamStart(partial=partial),
        TextDelta(content_index=0, delta="half a sen", partial=partial),
    )

    message = await collect(service.stream(_request()))

    assert message.stop_reason is StopReason.ERROR
    assert "without a terminal chunk" in (message.error_message or "")


async def test_premature_eof_preserves_the_accumulated_partial() -> None:
    """Replacing a real partial response with an empty message would discard
    what the model actually produced."""
    partial = _partial("half a sen")
    service = _service(
        StreamStart(partial=partial),
        TextDelta(content_index=0, delta="half a sen", partial=partial),
    )

    message = await collect(service.stream(_request()))

    assert text_of(message) == "half a sen"
    assert message.usage.total == 7
    assert (message.provider, message.model) == ("mock", "mock-1")


async def test_a_stream_that_yields_nothing_still_settles() -> None:
    service = _service()

    message = await collect(service.stream(_request()))

    assert message.stop_reason is StopReason.ERROR
    assert (message.provider, message.model) == ("mock", "mock-1")


async def test_nothing_escapes_iteration_on_premature_eof() -> None:
    """Stated separately from collect(): raw iteration must not raise either."""
    partial = _partial("x")
    service = _service(StreamStart(partial=partial))

    chunks = [chunk async for chunk in service.stream(_request())]

    assert isinstance(chunks[-1], StreamError)


async def test_the_public_stream_fuses_after_the_first_terminal() -> None:
    """A provider emitting a second terminal cannot produce a second public one."""
    settled = _partial("done", StopReason.STOP)
    extra = _partial("ignored", StopReason.STOP)
    service = _service(
        StreamDone(message=settled, partial=settled),
        StreamDone(message=extra, partial=extra),
        TextDelta(content_index=0, delta="after the end", partial=extra),
    )

    chunks = [chunk async for chunk in service.stream(_request())]

    assert len(chunks) == 1
    assert text_of(chunks[0].message) == "done"


async def test_a_well_formed_stream_is_unchanged() -> None:
    settled = _partial("hello", StopReason.STOP)
    service = _service(
        StreamStart(partial=settled),
        TextDelta(content_index=0, delta="hello", partial=settled),
        StreamDone(message=settled, partial=settled),
    )

    chunks = [chunk async for chunk in service.stream(_request())]

    assert len(chunks) == 3
    assert isinstance(chunks[-1], StreamDone)


async def test_a_represented_provider_error_still_rides_the_stream() -> None:
    failed = _partial("", StopReason.ERROR)
    service = _service(StreamError(reason=StopReason.ERROR, message=failed, partial=failed))

    message = await collect(service.stream(_request()))

    assert message.stop_reason is StopReason.ERROR


async def test_an_unknown_model_still_fails_eagerly() -> None:
    """The other side of the boundary is unchanged: caller bugs raise."""
    service = _service()

    with pytest.raises(UnknownModelError):
        service.stream(Request(model=ModelId("mock", "nope"), system="", messages=()))


async def test_the_source_is_not_drained_past_its_terminal() -> None:
    """Section 4: the first terminal wins and the stream fuses. The source is
    not read further merely to discover whether the provider would violate the
    protocol again -- so the chunks it queued after its terminal stay unpulled.
    """
    from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse

    adapter = MockAdapter(
        [
            ScriptedResponse(
                content=(TextBlock(text="complete"),),
                stop_reason=StopReason.STOP,
                chunks_after_terminal=2,
            )
        ]
    )
    service = LlmService()
    service.register(adapter)
    request = Request(
        model=ModelId("mock", "mock-1"), system="", messages=(), max_output_tokens=None
    )

    chunks = [chunk async for chunk in service.stream(request)]

    # start, delta, done -- and nothing after.
    assert adapter.pulled == 3
    assert sum(isinstance(chunk, StreamDone) for chunk in chunks) == 1
    assert text_of(chunks[-1].message) == "complete"
