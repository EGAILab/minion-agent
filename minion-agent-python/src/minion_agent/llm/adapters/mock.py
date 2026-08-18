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


@dataclass(frozen=True, slots=True)
class ScriptedResponse:
    """One response the adapter will return, in order."""

    content: tuple[ContentBlock, ...]
    stop_reason: StopReason
    usage: Usage = Usage()
    error_message: str | None = None


class MockAdapter:
    """Returns scripted responses in order, recording each request."""

    provider = "mock"
    models = frozenset({"mock-1"})

    def __init__(self, script: Sequence[ScriptedResponse]) -> None:
        self._script = list(script)
        self._next = 0
        self.requests: list[Request] = []

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

        async def run() -> AsyncIterator[StreamChunk]:
            pending = build(StopReason.PENDING)
            yield StreamStart(partial=pending)

            for index, block in enumerate(response.content):
                if isinstance(block, TextBlock):
                    yield TextDelta(content_index=index, delta=block.text, partial=pending)

            settled = build(response.stop_reason)
            if response.stop_reason in _ERROR_REASONS:
                yield StreamError(
                    reason=response.stop_reason, message=settled, partial=settled
                )
            else:
                yield StreamDone(message=settled, partial=settled)

        return run()
