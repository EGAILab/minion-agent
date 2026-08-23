"""The scripted adapter behaves exactly like a provider that never raises."""

from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.content import TextBlock, ToolCallBlock
from minion_agent.llm.messages import (
    AssistantMessageDiagnostic,
    Cost,
    DeferredHandle,
    DiagnosticError,
    StopReason,
    Usage,
    UserMessage,
)
from minion_agent.llm.service import ModelId, Request
from minion_agent.llm.stream import StreamDone, StreamError, TextDelta, collect


def _request(text: str = "hello") -> Request:
    return Request(
        model=ModelId("mock", "mock-1"),
        system="be helpful",
        messages=(UserMessage(content=(TextBlock(text=text),), timestamp=1),),
    )


async def test_a_scripted_text_response_streams_and_settles() -> None:
    adapter = MockAdapter([ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP)])

    chunks = [chunk async for chunk in adapter.stream(_request())]

    assert isinstance(chunks[-1], StreamDone)
    assert any(isinstance(chunk, TextDelta) for chunk in chunks)
    assert chunks[-1].message.stop_reason is StopReason.STOP


async def test_responses_are_returned_in_script_order() -> None:
    adapter = MockAdapter(
        [
            ScriptedResponse((TextBlock(text="first"),), StopReason.STOP),
            ScriptedResponse((TextBlock(text="second"),), StopReason.STOP),
        ]
    )

    first = await collect(adapter.stream(_request()))
    second = await collect(adapter.stream(_request()))

    assert (first.content[0].text, second.content[0].text) == ("first", "second")


async def test_a_tool_call_response_settles_with_tool_use() -> None:
    adapter = MockAdapter(
        [
            ScriptedResponse(
                (ToolCallBlock(id="t1", name="bash", arguments={"command": "ls"}),),
                StopReason.TOOL_USE,
            )
        ]
    )

    message = await collect(adapter.stream(_request()))

    assert message.stop_reason is StopReason.TOOL_USE
    assert message.content[0].name == "bash"


async def test_a_scripted_error_rides_the_stream() -> None:
    adapter = MockAdapter([ScriptedResponse((), StopReason.ERROR, error_message="upstream 500")])

    chunks = [chunk async for chunk in adapter.stream(_request())]

    assert isinstance(chunks[-1], StreamError)
    assert chunks[-1].message.error_message == "upstream 500"


async def test_exhausting_the_script_fails_in_band() -> None:
    """An under-scripted scenario gets a diagnosable failure, not a crash."""
    adapter = MockAdapter([])

    message = await collect(adapter.stream(_request()))

    assert message.stop_reason is StopReason.ERROR
    assert "exhausted" in (message.error_message or "")


async def test_the_adapter_records_what_it_was_asked() -> None:
    adapter = MockAdapter([ScriptedResponse((), StopReason.STOP)])

    await collect(adapter.stream(_request("remember me")))

    assert adapter.requests[0].system == "be helpful"
    assert len(adapter.requests) == 1


async def test_usage_is_carried_through() -> None:
    adapter = MockAdapter([ScriptedResponse((), StopReason.STOP, usage=Usage(input=7, output=3))])

    message = await collect(adapter.stream(_request()))

    assert message.usage.total == 10


async def test_an_aborted_response_also_rides_the_stream() -> None:
    adapter = MockAdapter([ScriptedResponse((), StopReason.ABORTED)])

    chunks = [chunk async for chunk in adapter.stream(_request())]

    assert isinstance(chunks[-1], StreamError)
    assert chunks[-1].reason is StopReason.ABORTED


async def test_response_identity_and_diagnostic_fields_are_carried_through() -> None:
    """LLM-F010: a real (reference) adapter must be able to script every
    Pi-required AssistantMessage field, not just the original 7 -- otherwise
    no canonical scenario could ever observe them through the real stream."""
    diagnostic = AssistantMessageDiagnostic(
        type="retry", timestamp=1, error=DiagnosticError(message="timeout")
    )
    handle = DeferredHandle(provider="mock", model_id="mock-1", api="mock", id="req-1")
    adapter = MockAdapter(
        [
            ScriptedResponse(
                (),
                StopReason.DEFERRED,
                response_model="mock-1-router",
                response_id="resp_1",
                diagnostics=(diagnostic,),
                deferred=handle,
                raw_stop_reason="provider_deferred",
                end_turn=True,
            )
        ]
    )

    message = await collect(adapter.stream(_request()))

    assert message.stop_reason is StopReason.DEFERRED
    assert message.response_model == "mock-1-router"
    assert message.response_id == "resp_1"
    assert message.diagnostics == (diagnostic,)
    assert message.deferred == handle
    assert message.raw_stop_reason == "provider_deferred"
    assert message.end_turn is True


async def test_unset_response_identity_fields_stay_absent_not_synthesized() -> None:
    """The adapter must not invent values for fields a scenario didn't
    script -- absence has to stay observable as None, not a fabricated
    default that would let a runner's projection lie about what the real
    object carries."""
    adapter = MockAdapter([ScriptedResponse((), StopReason.STOP)])

    message = await collect(adapter.stream(_request()))

    assert (
        message.response_model,
        message.response_id,
        message.diagnostics,
        message.deferred,
        message.raw_stop_reason,
        message.end_turn,
    ) == (None, None, None, None, None, None)


async def test_cost_is_carried_through_usage() -> None:
    cost = Cost(input=0.01, output=0.02, total=0.03)
    adapter = MockAdapter(
        [ScriptedResponse((), StopReason.STOP, usage=Usage(input=5, total_tokens=5, cost=cost))]
    )

    message = await collect(adapter.stream(_request()))

    assert message.usage.total_tokens == 5
    assert message.usage.cost == cost
