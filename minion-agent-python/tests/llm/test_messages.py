"""Messages carry content plus the accounting a caller needs."""

import pytest

from minion_agent.llm.content import TextBlock, ToolCallBlock
from minion_agent.llm.messages import (
    AssistantMessage,
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


def test_tool_result_message_links_to_its_call() -> None:
    message = ToolResultMessage(tool_call_id="t1", content=(TextBlock(text="ok"),), timestamp=2)

    assert message.tool_call_id == "t1"
    assert not message.is_error


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
