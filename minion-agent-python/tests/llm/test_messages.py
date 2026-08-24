"""Messages carry content plus the accounting a caller needs."""

import pytest

from minion_agent.llm.content import TextBlock, ToolCallBlock
from minion_agent.llm.messages import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    Cost,
    DeferredHandle,
    DiagnosticError,
    StopReason,
    ToolResultMessage,
    Usage,
    UserMessage,
    text_of,
)


def _assistant(**overrides: object) -> AssistantMessage:
    defaults: dict[str, object] = {
        "content": (TextBlock(text="hi"),),
        "stop_reason": StopReason.STOP,
        "usage": Usage(),
        "model": "mock-1",
        "provider": "mock",
        "timestamp": 1,
    }
    return AssistantMessage(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_usage_total_sums_every_token_class() -> None:
    usage = Usage(input=10, output=5, cache_read=2, cache_write=3)

    assert usage.total == 20


def test_usage_defaults_to_zero() -> None:
    assert Usage().total == 0


def test_reasoning_is_optional_and_not_double_counted() -> None:
    """Reasoning tokens are a subset of output, never an extra class."""
    usage = Usage(input=1, output=10, reasoning=4)

    assert usage.total == 11


def test_user_message_carries_content_and_timestamp() -> None:
    message = UserMessage(content=(TextBlock(text="hello"),), timestamp=7)

    assert text_of(message) == "hello"
    assert message.timestamp == 7


def test_assistant_message_records_provider_identity() -> None:
    message = _assistant()

    assert (message.provider, message.model) == ("mock", "mock-1")
    assert message.stop_reason is StopReason.STOP


def test_assistant_message_may_carry_an_error() -> None:
    message = _assistant(stop_reason=StopReason.ERROR, error_message="upstream 500")

    assert message.error_message == "upstream 500"


def test_assistant_message_api_defaults_to_mock() -> None:
    """Defaults to "mock" only because the mock adapter is the sole
    registered adapter today (LLM-F003's disposition)."""
    assert _assistant().api == "mock"


def test_assistant_message_response_identity_fields_default_and_are_settable() -> None:
    plain = _assistant()
    assert (plain.response_model, plain.response_id, plain.raw_stop_reason, plain.end_turn) == (
        None,
        None,
        None,
        None,
    )

    full = _assistant(
        response_model="anthropic/claude-3",
        response_id="resp_123",
        raw_stop_reason="max_tokens",
        end_turn=True,
    )
    assert full.response_model == "anthropic/claude-3"
    assert full.response_id == "resp_123"
    assert full.raw_stop_reason == "max_tokens"
    assert full.end_turn is True


def test_assistant_message_diagnostics_default_and_are_settable() -> None:
    assert _assistant().diagnostics is None

    diagnostic = AssistantMessageDiagnostic(
        type="retry", timestamp=1, error=DiagnosticError(message="timeout")
    )
    message = _assistant(diagnostics=(diagnostic,))
    assert message.diagnostics == (diagnostic,)


def test_assistant_message_deferred_defaults_and_is_settable() -> None:
    assert _assistant().deferred is None

    handle = DeferredHandle(provider="mock", model_id="mock-1", api="mock", id="req-1")
    message = _assistant(stop_reason=StopReason.DEFERRED, deferred=handle)
    assert message.deferred == handle


def test_tool_result_message_links_to_its_call() -> None:
    message = ToolResultMessage(
        tool_call_id="t1", content=(TextBlock(text="ok"),), timestamp=2, tool_name="echo"
    )

    assert message.tool_call_id == "t1"
    assert not message.is_error


def test_tool_result_message_optional_fields_default_and_are_settable() -> None:
    """`tool_name` is required (delta finding A), not one of the optional
    fields this test covers -- it is supplied directly, not asserted to default."""
    plain = ToolResultMessage(tool_call_id="t1", content=(), timestamp=2, tool_name="bash")
    assert (plain.details, plain.usage, plain.added_tool_names) == (
        None,
        None,
        None,
    )

    full = ToolResultMessage(
        tool_call_id="t1",
        content=(),
        timestamp=2,
        tool_name="bash",
        details={"exit_code": 0},
        usage=Usage(input=1),
        added_tool_names=("new_tool",),
    )
    assert full.tool_name == "bash"
    assert full.details == {"exit_code": 0}
    assert full.usage == Usage(input=1)
    assert full.added_tool_names == ("new_tool",)


def test_stop_reason_includes_deferred() -> None:
    assert StopReason.DEFERRED == "deferred"


def test_usage_cost_and_total_tokens_default_and_are_settable() -> None:
    plain = Usage()
    assert plain.total_tokens == 0
    assert plain.cost == Cost()

    priced = Usage(total_tokens=42, cost=Cost(input=0.01, output=0.02, total=0.03))
    assert priced.total_tokens == 42
    assert priced.cost.total == 0.03


def test_usage_cache_write_1h_defaults_to_none() -> None:
    assert Usage().cache_write_1h is None
    assert Usage(cache_write_1h=5).cache_write_1h == 5


def test_deferred_handle_carries_provider_identity() -> None:
    handle = DeferredHandle(provider="mock", model_id="mock-1", api="mock", id="req-1")

    assert (handle.provider, handle.model_id, handle.api, handle.id) == (
        "mock",
        "mock-1",
        "mock",
        "req-1",
    )
    assert handle.expires_at is None


def test_assistant_message_diagnostic_carries_a_structured_error() -> None:
    diagnostic = AssistantMessageDiagnostic(
        type="retry",
        timestamp=1,
        error=DiagnosticError(message="upstream 500", code=500),
    )

    assert diagnostic.error is not None
    assert diagnostic.error.message == "upstream 500"
    assert diagnostic.error.code == 500


def test_text_of_concatenates_only_text_blocks() -> None:
    message = _assistant(
        content=(
            TextBlock(text="a"),
            ToolCallBlock(id="t1", name="bash", arguments={}),
            TextBlock(text="b"),
        )
    )

    assert text_of(message) == "ab"


def test_messages_are_frozen() -> None:
    message = UserMessage(content=(), timestamp=1)

    with pytest.raises(Exception):  # noqa: B017
        message.timestamp = 2  # type: ignore[misc]
