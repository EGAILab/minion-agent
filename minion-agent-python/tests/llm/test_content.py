"""Content blocks are the provider-neutral vocabulary a model sees."""

import pytest

from minion_agent.llm.content import ImageBlock, TextBlock, ThinkingBlock, ToolCallBlock


def test_text_block_carries_its_text() -> None:
    assert TextBlock(text="hello").text == "hello"


def test_text_block_signature_defaults_to_none_and_is_settable() -> None:
    assert TextBlock(text="hello").text_signature is None
    assert TextBlock(text="hello", text_signature="sig-1").text_signature == "sig-1"


def test_thinking_block_is_distinct_from_text() -> None:
    assert ThinkingBlock(thinking="reasoning") != TextBlock(text="reasoning")


def test_thinking_block_signature_and_redacted_default_and_are_settable() -> None:
    plain = ThinkingBlock(thinking="reasoning")
    assert plain.thinking_signature is None
    assert plain.redacted is False

    signed = ThinkingBlock(thinking="reasoning", thinking_signature="sig-1", redacted=True)
    assert signed.thinking_signature == "sig-1"
    assert signed.redacted is True


def test_tool_call_block_carries_id_name_and_arguments() -> None:
    call = ToolCallBlock(id="t1", name="bash", arguments={"command": "ls"})

    assert (call.id, call.name, call.arguments) == ("t1", "bash", {"command": "ls"})


def test_tool_call_block_signature_and_namespace_default_and_are_settable() -> None:
    plain = ToolCallBlock(id="t1", name="bash", arguments={})
    assert plain.thought_signature is None
    assert plain.namespace is None

    tagged = ToolCallBlock(
        id="t1", name="bash", arguments={}, thought_signature="sig-1", namespace="mcp:fs"
    )
    assert tagged.thought_signature == "sig-1"
    assert tagged.namespace == "mcp:fs"


def test_image_block_accepts_inline_data() -> None:
    block = ImageBlock(mime_type="image/png", data=b"\x89PNG")

    assert block.mime_type == "image/png"
    assert block.reference is None


def test_image_block_accepts_a_reference() -> None:
    block = ImageBlock(mime_type="image/png", reference="sha256:abc")

    assert block.data is None


def test_image_block_requires_exactly_one_source() -> None:
    """Neither is meaningless; both is ambiguous about what the model saw."""
    with pytest.raises(ValueError, match="exactly one"):
        ImageBlock(mime_type="image/png")

    with pytest.raises(ValueError, match="exactly one"):
        ImageBlock(mime_type="image/png", data=b"x", reference="sha256:abc")


def test_blocks_are_frozen() -> None:
    block = TextBlock(text="hello")

    with pytest.raises(Exception):  # noqa: B017
        block.text = "changed"  # type: ignore[misc]
