"""The `ctx.llm` seam: model identity, adapters, and the never-raises boundary.

The boundary cuts at the moment a stream is returned (design spec section 4):

* Before  — ordinary exceptions. Unknown models, bad configuration, and
  programming errors raise, because they are caller bugs discoverable
  immediately. Reporting a mistyped model name as a streamed error message
  would bury a caller bug in the transcript.
* After   — nothing escapes iteration. Provider, network, model, and
  cancellation failures terminate the stream with an error chunk.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from .errors import UnknownModelError
from .messages import AssistantMessage, Message, StopReason, Usage
from .stream import AssistantStream, StreamDone, StreamError
from .tools import ToolSchema


@dataclass(frozen=True, slots=True)
class ModelId:
    """Model identity: the `provider + api + model` triple (design spec
    section 4). `api` is the wire protocol (e.g. `openai-completions`); a
    single API may serve several providers, and compatibility checks key on
    all three, not `model` alone.

    `api` defaults to `"mock"` only because the mock adapter is the sole
    registered adapter today (LLM-F006's disposition, see
    assurance/layers/02-llm.md) -- every current caller wants that value.
    The default becomes actively wrong once a second API exists (Phase 5);
    remove it then, so every caller must say which API it means."""

    provider: str
    model: str
    api: str = "mock"


@dataclass(frozen=True, slots=True)
class Request:
    """One logical request to a model."""

    model: ModelId
    system: str
    messages: tuple[Message, ...]
    max_output_tokens: int | None = None
    tools: tuple[ToolSchema, ...] = ()
    """Tools visible to the model for this request.

    Empty is meaningful, not a placeholder: a step whose scope exposes no
    tools genuinely offers none."""


class Adapter(Protocol):
    """A provider adapter.

    `stream` must not raise for provider, network, model, or cancellation
    failures; it encodes them in the returned stream instead.
    """

    provider: str
    api: str
    models: frozenset[str]

    def stream(self, request: Request) -> AssistantStream: ...


class LlmService:
    """Resolves a request's model to an adapter."""

    __service_name__ = "llm"

    def __init__(self) -> None:
        self._adapters: dict[ModelId, Adapter] = {}

    def register(self, adapter: Adapter) -> Callable[[], None]:
        """Register every model `adapter` supplies; returns a withdrawal handle.

        The handle removes only registrations this adapter still holds, so
        withdrawing a superseded adapter cannot remove its replacement.
        """
        ids = [ModelId(adapter.provider, model, adapter.api) for model in adapter.models]
        for model_id in ids:
            self._adapters[model_id] = adapter

        def withdraw() -> None:
            for model_id in ids:
                if self._adapters.get(model_id) is adapter:
                    del self._adapters[model_id]

        return withdraw

    def models(self) -> frozenset[ModelId]:
        """Every currently resolvable model."""
        return frozenset(self._adapters)

    def stream(self, request: Request) -> AssistantStream:
        """Dispatch `request` to its adapter.

        Raises `UnknownModelError` eagerly when no adapter supplies the model —
        a caller bug, discoverable immediately. Everything after this point
        rides the returned stream.
        """
        adapter = self._adapters.get(request.model)
        if adapter is None:
            raise UnknownModelError(
                f"no adapter supplies {request.model.provider}/{request.model.model}"
            )
        return _settled(adapter.stream(request), request)


def _empty_partial(request: Request) -> AssistantMessage:
    """A pending message for a stream that produced nothing at all."""
    return AssistantMessage(
        content=(),
        stop_reason=StopReason.PENDING,
        usage=Usage(),
        model=request.model.model,
        provider=request.model.provider,
        timestamp=0,
        api=request.model.api,
    )


async def _settled(source: AssistantStream, request: Request) -> AssistantStream:
    """Guarantee the contract §4 states for a returned stream.

    Three guarantees, all observable:

    * **Nothing escapes iteration.** A raw stream that ends before emitting a
      terminal is a runtime streaming failure — a truncated response is the
      ordinary case — so it settles as a terminal error chunk rather than
      raising. The accumulated partial is preserved: discarding it would
      replace a real partial response with an unrelated empty one.
    * **An adapter that raises instead of encoding a failure still settles
      in-band.** Pi's contract (`StreamFunction`) requires request/provider/
      network/runtime failures to be encoded in the returned stream, not
      thrown — an adapter that gets this wrong and raises mid-iteration
      (LLM-F007) must not be able to break the guarantee for every
      well-behaved adapter's caller.
    * **The first terminal wins, and the stream then fuses.** Nothing is
      yielded afterward, and the source is not drained further merely to
      discover whether a provider would have violated the protocol again.
    """
    partial: AssistantMessage | None = None

    def _error_terminal(message: str) -> StreamError:
        settled = replace(
            partial if partial is not None else _empty_partial(request),
            stop_reason=StopReason.ERROR,
            error_message=message,
        )
        return StreamError(reason=StopReason.ERROR, message=settled, partial=settled)

    try:
        async for chunk in source:
            partial = chunk.partial
            yield chunk
            if isinstance(chunk, StreamDone | StreamError):
                return
    except Exception as error:
        yield _error_terminal(f"provider stream raised {type(error).__name__}: {error}")
        return

    yield _error_terminal(
        "provider stream ended without a terminal chunk; the response is incomplete"
    )
