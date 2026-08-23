"""The seam resolves adapters and enforces the never-raises boundary."""

from collections.abc import AsyncIterator

import pytest

from minion_agent.llm.content import TextBlock
from minion_agent.llm.errors import UnknownModelError
from minion_agent.llm.messages import AssistantMessage, StopReason, Usage
from minion_agent.llm.service import LlmService, ModelId, Request
from minion_agent.llm.stream import StreamChunk, StreamDone, StreamError, collect


def _settled(reason: StopReason = StopReason.STOP) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text="ok"),),
        stop_reason=reason,
        usage=Usage(input=1, output=1),
        model="mock-1",
        provider="mock",
        timestamp=1,
    )


class GoodAdapter:
    provider = "mock"
    api = "mock"
    models = frozenset({"mock-1"})

    def stream(self, request: Request) -> AsyncIterator[StreamChunk]:
        async def run() -> AsyncIterator[StreamChunk]:
            message = _settled()
            yield StreamDone(message=message, partial=message)

        return run()


class FailingAdapter:
    """Fails the way an adapter must: in-band, never by raising."""

    provider = "mock"
    api = "mock"
    models = frozenset({"mock-1"})

    def stream(self, request: Request) -> AsyncIterator[StreamChunk]:
        async def run() -> AsyncIterator[StreamChunk]:
            message = _settled(StopReason.ERROR)
            yield StreamError(reason=StopReason.ERROR, message=message, partial=message)

        return run()


def _request(model: str = "mock-1") -> Request:
    return Request(model=ModelId("mock", model), system="", messages=())


async def test_a_registered_model_streams() -> None:
    service = LlmService()
    service.register(GoodAdapter())

    result = await collect(service.stream(_request()))

    assert result.stop_reason is StopReason.STOP


async def test_an_unknown_model_raises_eagerly() -> None:
    """A caller bug, discoverable immediately — not buried in the transcript."""
    service = LlmService()
    service.register(GoodAdapter())

    with pytest.raises(UnknownModelError, match="mock-9"):
        service.stream(_request("mock-9"))


async def test_provider_failures_ride_the_stream() -> None:
    service = LlmService()
    service.register(FailingAdapter())

    result = await collect(service.stream(_request()))

    assert result.stop_reason is StopReason.ERROR


def test_models_lists_every_registered_pair() -> None:
    service = LlmService()
    service.register(GoodAdapter())

    assert service.models() == frozenset({ModelId("mock", "mock-1")})


def test_model_id_defaults_api_to_mock() -> None:
    """Identity is the provider+api+model triple (design spec section 4).
    `api` defaults to "mock" only because no second API exists yet
    (LLM-F006) -- every existing caller means this value."""
    assert ModelId("mock", "mock-1").api == "mock"


def test_registering_an_adapter_carries_its_declared_api() -> None:
    class OtherApiAdapter:
        provider = "mock"
        api = "not-mock"
        models = frozenset({"other-1"})

        def stream(self, request: Request) -> AsyncIterator[StreamChunk]:
            async def run() -> AsyncIterator[StreamChunk]:
                yield StreamDone(message=_settled(), partial=_settled())

            return run()

    service = LlmService()
    service.register(OtherApiAdapter())

    assert service.models() == frozenset({ModelId("mock", "other-1", "not-mock")})


def test_unregistering_withdraws_the_models() -> None:
    service = LlmService()
    withdraw = service.register(GoodAdapter())

    withdraw()

    assert service.models() == frozenset()
    with pytest.raises(UnknownModelError):
        service.stream(_request())


def test_withdrawing_twice_is_harmless() -> None:
    service = LlmService()
    withdraw = service.register(GoodAdapter())

    withdraw()
    withdraw()

    assert service.models() == frozenset()


def test_a_later_adapter_replaces_an_earlier_one_for_the_same_model() -> None:
    """Adapters are not services: last registration wins, and withdrawing the
    superseded one must not remove the live registration."""
    service = LlmService()
    first, second = GoodAdapter(), FailingAdapter()
    withdraw_first = service.register(first)
    service.register(second)

    withdraw_first()

    assert service.models() == frozenset({ModelId("mock", "mock-1")})
