"""A scripted adapter: deterministic responses, no network, no clock.

This is a real adapter, not a test double. Conformance depends on it honouring
the never-raises contract exactly — including when a scenario under-scripts it,
which fails in-band so the scenario reports a diagnosable model error rather
than crashing the run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from ..content import ContentBlock, TextBlock
from ..messages import AssistantMessage, StopReason, Usage
from ..service import Request
from ..stream import StreamChunk, StreamDone, StreamError, StreamStart, TextDelta

_ERROR_REASONS = frozenset({StopReason.ERROR, StopReason.ABORTED})

_NO_USAGE = Usage()
"""Shared zero usage. Safe as a default because `Usage` is frozen, so no
scripted response can mutate what another one sees."""


@dataclass(frozen=True, slots=True)
class ScriptedResponse:
    """One response the adapter will return, in order."""

    content: tuple[ContentBlock, ...]
    stop_reason: StopReason
    usage: Usage = _NO_USAGE
    error_message: str | None = None
    truncated: bool = False
    """End the raw stream without a terminal chunk.

    A truncated provider response is the ordinary case section 4 exists for.
    The adapter is deliberately allowed to produce it: the service, not the
    adapter, is what must settle it."""
    chunks_after_terminal: int = 0
    """Emit this many extra deltas after the terminal, then a second terminal.

    A misbehaving provider. The service must fuse after the first terminal, so
    none of this reaches a consumer."""


class MockAdapter:
    """Returns scripted responses in order, recording each request."""

    provider = "mock"
    models = frozenset({"mock-1"})

    def __init__(self, script: Sequence[ScriptedResponse]) -> None:
        self._script = list(script)
        self._next = 0
        self.requests: list[Request] = []
        self.pulled = 0
        """Chunks a consumer actually pulled, across every stream."""

    def _take(self) -> ScriptedResponse:
        if self._next >= len(self._script):
            return ScriptedResponse(
                content=(),
                stop_reason=StopReason.ERROR,
                error_message=(
                    f"mock script exhausted after {len(self._script)} response(s); "
                    "the scenario asked for one more"
                ),
            )
        response = self._script[self._next]
        self._next += 1
        return response

    def stream(self, request: Request) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        response = self._take()

        def build(reason: StopReason) -> AssistantMessage:
            return AssistantMessage(
                content=response.content,
                stop_reason=reason,
                usage=response.usage,
                model=request.model.model,
                provider=request.model.provider,
                timestamp=len(self.requests),
                error_message=response.error_message,
            )

        # Built eagerly, then replayed. A generator would suspend at each
        # yield, so anything scripted after the terminal could never run: the
        # consumer stops pulling and the generator is closed where it stands.
        # A provider misbehaving *after* its terminal is exactly what
        # `chunks_after_terminal` is for, and it has to be expressible.
        pending = build(StopReason.PENDING)
        chunks: list[StreamChunk] = [StreamStart(partial=pending)]

        for index, block in enumerate(response.content):
            if isinstance(block, TextBlock):
                chunks.append(TextDelta(content_index=index, delta=block.text, partial=pending))

        if not response.truncated:
            settled = build(response.stop_reason)
            if response.stop_reason in _ERROR_REASONS:
                chunks.append(
                    StreamError(reason=response.stop_reason, message=settled, partial=settled)
                )
            else:
                chunks.append(StreamDone(message=settled, partial=settled))

            for extra in range(response.chunks_after_terminal):
                chunks.append(TextDelta(content_index=extra, delta="after", partial=settled))
            if response.chunks_after_terminal:
                chunks.append(StreamDone(message=settled, partial=settled))

        return _Replay(chunks, self)


class _Replay:
    """Replays a pre-built chunk list, counting how many were pulled.

    `pulled` is what makes "the source is not drained past its terminal"
    observable: a consumer that keeps reading to check for further protocol
    violations would show up here.
    """

    def __init__(self, chunks: list[StreamChunk], adapter: MockAdapter) -> None:
        self._chunks = chunks
        self._index = 0
        self._adapter = adapter

    def __aiter__(self) -> _Replay:
        return self

    async def __anext__(self) -> StreamChunk:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        self._adapter.pulled += 1
        return chunk
