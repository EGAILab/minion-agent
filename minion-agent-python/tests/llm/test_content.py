"""Content blocks are the provider-neutral vocabulary a model sees."""

import pytest

from minion_agent.llm.content import ImageBlock, TextBlock, ThinkingBlock, ToolCallBlock


def test_text_block_carries_its_text() -> None:
    assert TextBlock(text="hello").text == "hello"


def test_thinking_block_is_distinct_from_text() -> None:
    assert ThinkingBlock(thinking="reasoning") != TextBlock(text="reasoning")


def test_tool_call_block_carries_id_name_and_arguments() -> None:
    call = ToolCallBlock(id="t1", name="bash", arguments={"command": "ls"})

    assert (call.id, call.name, call.arguments) == ("t1", "bash", {"command": "ls"})


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
