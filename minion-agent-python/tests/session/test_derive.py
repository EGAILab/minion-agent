"""Derivation projects the surface into model history."""

import pytest

from minion_agent.llm.content import ImageBlock, TextBlock, ThinkingBlock, ToolCallBlock
from minion_agent.llm.messages import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    Cost,
    DeferredHandle,
    DiagnosticError,
    Message,
    StopReason,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from minion_agent.session.derive import decode_message, derive_messages, encode_message
from minion_agent.session.events import EventKind
from minion_agent.session.log import SessionLog


def _user(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text=text),),
        stop_reason=StopReason.STOP,
        usage=Usage(input=1, output=1),
        model="mock-1",
        provider="mock",
        timestamp=2,
    )


def _append(log: SessionLog, kind: EventKind, message: Message) -> None:
    log.append(kind, {"message": encode_message(message)})


def test_an_empty_log_derives_nothing() -> None:
    assert derive_messages(SessionLog("s1")) == ()


def test_surface_events_derive_in_order() -> None:
    log = SessionLog("s1")
    _append(log, EventKind.USER_MESSAGE, _user("hello"))
    _append(log, EventKind.ASSISTANT_MESSAGE, _assistant("hi"))

    derived = derive_messages(log)

    assert [type(message).__name__ for message in derived] == [
        "UserMessage",
        "AssistantMessage",
    ]


def test_log_only_events_do_not_derive() -> None:
    log = SessionLog("s1")
    log.append(EventKind.TURN_START, {"turn": 1})
    _append(log, EventKind.USER_MESSAGE, _user("hello"))
    log.append(EventKind.ASSISTANT_CHUNK, {"delta": "h"})
    log.append(EventKind.REQUEST_HEADER, {"system": "x"})

    assert len(derive_messages(log)) == 1


def test_user_message_round_trips() -> None:
    message = _user("hello")

    assert decode_message(encode_message(message)) == message


def test_string_valued_user_message_round_trips_as_a_string() -> None:
    """UserMessage.content is str | tuple[...] (LLM-F012/SES-F009), both first-class. A string
    must survive Session persistence exactly as a string -- not normalized into a one-element
    TextBlock array."""
    message = UserMessage(content="hello", timestamp=1)

    restored = decode_message(encode_message(message))

    assert restored == message
    assert isinstance(restored.content, str)
    assert restored.content == "hello"


def test_string_and_single_text_block_user_messages_remain_distinct_after_round_trip() -> None:
    """Both representations project to the same visible text, but Session must not conflate
    them: a real string stays a string, and a real single-TextBlock array stays an array."""
    as_string = UserMessage(content="hello", timestamp=1)
    as_block = UserMessage(content=(TextBlock(text="hello"),), timestamp=1)

    restored_string = decode_message(encode_message(as_string))
    restored_block = decode_message(encode_message(as_block))

    assert isinstance(restored_string.content, str)
    assert isinstance(restored_block.content, tuple)
    assert restored_string != restored_block


def test_assistant_message_round_trips_with_accounting() -> None:
    message = _assistant("hi")

    restored = decode_message(encode_message(message))

    assert restored == message
    assert restored.usage.input == 1


def test_tool_result_message_round_trips() -> None:
    message = ToolResultMessage(
        tool_call_id="t1",
        content=(TextBlock(text="ok"),),
        timestamp=3,
        tool_name="echo",
        is_error=True,
    )

    assert decode_message(encode_message(message)) == message


def test_tool_result_message_optional_fields_round_trip() -> None:
    message = ToolResultMessage(
        tool_call_id="t1",
        content=(TextBlock(text="ok"),),
        timestamp=3,
        tool_name="bash",
        details={"exit_code": 0},
        usage=Usage(input=2, total_tokens=2),
        added_tool_names=("new_tool",),
    )

    assert decode_message(encode_message(message)) == message


def test_usage_cost_and_total_tokens_round_trip() -> None:
    message = _assistant("hi")
    message = AssistantMessage(
        content=message.content,
        stop_reason=message.stop_reason,
        usage=Usage(input=1, output=2, total_tokens=3, cost=Cost(input=0.1, total=0.1)),
        model=message.model,
        provider=message.provider,
        timestamp=message.timestamp,
    )

    restored = decode_message(encode_message(message))

    assert restored == message
    assert restored.usage.cost.total == 0.1


def test_assistant_message_response_identity_and_diagnostics_round_trip() -> None:
    message = AssistantMessage(
        content=(TextBlock(text="hi"),),
        stop_reason=StopReason.DEFERRED,
        usage=Usage(input=1),
        model="mock-1",
        provider="mock",
        timestamp=2,
        api="not-mock",
        response_model="mock-1-router",
        response_id="resp_1",
        diagnostics=(
            AssistantMessageDiagnostic(
                type="retry",
                timestamp=1,
                error=DiagnosticError(message="timeout", name="TimeoutError", code=504),
                details={"attempt": 2},
            ),
        ),
        deferred=DeferredHandle(
            provider="mock",
            model_id="mock-1",
            api="mock",
            id="req-1",
            expires_at=1000,
            poll_after_ms=500,
            data={"batch_id": "b1"},
        ),
        raw_stop_reason="max_tokens",
        end_turn=True,
    )

    assert decode_message(encode_message(message)) == message


def test_every_content_block_round_trips() -> None:
    message = UserMessage(
        content=(
            TextBlock(text="t"),
            ThinkingBlock(thinking="r"),
            ImageBlock(mime_type="image/png", reference="sha256:abc"),
            ToolCallBlock(id="t1", name="bash", arguments={"command": "ls"}),
        ),
        timestamp=1,
    )

    assert decode_message(encode_message(message)) == message


def test_replay_signature_fields_round_trip() -> None:
    """Opaque provider replay metadata (design spec section 4) must survive
    session serialization -- the log is the only place this data is stored
    between requests."""
    message = UserMessage(
        content=(
            TextBlock(text="t", text_signature="sig-text"),
            ThinkingBlock(thinking="r", thinking_signature="sig-think", redacted=True),
            ToolCallBlock(
                id="t1",
                name="bash",
                arguments={"command": "ls"},
                thought_signature="sig-tool",
                namespace="mcp:fs",
            ),
        ),
        timestamp=1,
    )

    assert decode_message(encode_message(message)) == message


def test_an_inline_image_round_trips_through_base64() -> None:
    message = UserMessage(
        content=(ImageBlock(mime_type="image/png", data=b"\x89PNG\r\n"),), timestamp=1
    )

    assert decode_message(encode_message(message)) == message


def test_encoded_messages_are_json_safe() -> None:
    """Encoding must produce something the log will accept."""
    log = SessionLog("s1")

    log.append(EventKind.USER_MESSAGE, {"message": encode_message(_user("hello"))})

    assert len(log) == 1


def test_an_inline_image_encodes_json_safely() -> None:
    """Raw bytes would be rejected by the log; base64 is why they are not."""
    log = SessionLog("s1")
    message = UserMessage(
        content=(ImageBlock(mime_type="image/png", data=b"\x89PNG"),), timestamp=1
    )

    log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})

    assert len(log) == 1


def test_an_unknown_block_type_is_rejected_on_decode() -> None:
    with pytest.raises(ValueError, match="unknown content block type"):
        decode_message({"role": "user", "content": [{"type": "video"}], "timestamp": 1})


def test_an_unknown_role_is_rejected_on_decode() -> None:
    with pytest.raises(ValueError, match="unknown message role"):
        decode_message({"role": "system", "content": [], "timestamp": 1})
