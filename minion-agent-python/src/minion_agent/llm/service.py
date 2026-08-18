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
from dataclasses import dataclass
from typing import Protocol

from .errors import UnknownModelError
from .messages import Message
from .stream import AssistantStream


@dataclass(frozen=True, slots=True)
class ModelId:
    """A provider and one of its models."""

    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class Request:
    """One logical request to a model."""

    model: ModelId
    system: str
    messages: tuple[Message, ...]
    max_output_tokens: int | None = None


class Adapter(Protocol):
    """A provider adapter.

    `stream` must not raise for provider, network, model, or cancellation
    failures; it encodes them in the returned stream instead.
    """

    provider: str
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
        ids = [ModelId(adapter.provider, model) for model in adapter.models]
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

        Raises `UnknownModelError` eagerly when no adapter supplies the model.
        """
        adapter = self._adapters.get(request.model)
        if adapter is None:
            raise UnknownModelError(
                f"no adapter supplies {request.model.provider}/{request.model.model}"
            )
        return adapter.stream(request)
