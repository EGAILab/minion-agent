"""Permanent static-type evidence for `LLM-F012`.

Not a pytest test: mypy checking this module IS the test. The `XFORM-R001` re-review found that
`UserMessage.content` remained statically `tuple[ContentBlock, ...]` even after the runtime
transform bug was fixed -- a runtime unit test could not have caught that, since Python does not
enforce dataclass field types at runtime. This module's only job is to fail `mypy --strict` if any
of these frozen-vocabulary constructions ever stop type-checking.

Run explicitly (not part of the default `mypy` gate, which is scoped to `src/minion_agent` only):

    mypy tests/typing/valid_message_construction.py --strict

Never imported or executed by pytest.
"""

from __future__ import annotations

from minion_agent.llm.content import ImageBlock, TextBlock, ThinkingBlock, ToolCallBlock
from minion_agent.llm.messages import (
    AssistantMessage,
    StopReason,
    ToolResultMessage,
    Usage,
    UserMessage,
)

# UserMessage.content: str | tuple[TextBlock | ImageBlock, ...] -- both first-class.
_user_string: UserMessage = UserMessage(content="hello", timestamp=1)
_user_text_block: UserMessage = UserMessage(content=(TextBlock(text="hello"),), timestamp=1)
_user_image_block: UserMessage = UserMessage(
    content=(ImageBlock(mime_type="image/png", data=b"x"),), timestamp=1
)
_user_mixed: UserMessage = UserMessage(
    content=(TextBlock(text="hi"), ImageBlock(mime_type="image/png", data=b"x")), timestamp=1
)

# AssistantMessage.content: tuple[TextBlock | ThinkingBlock | ToolCallBlock, ...] -- no ImageBlock.
_assistant: AssistantMessage = AssistantMessage(
    content=(
        TextBlock(text="hi"),
        ThinkingBlock(thinking="reasoning"),
        ToolCallBlock(id="c1", name="t", arguments={}),
    ),
    stop_reason=StopReason.STOP,
    usage=Usage(),
    model="m1",
    provider="p",
    timestamp=1,
)

# ToolResultMessage.content: tuple[TextBlock | ImageBlock, ...] -- no ThinkingBlock/ToolCallBlock.
_tool_result: ToolResultMessage = ToolResultMessage(
    tool_call_id="c1",
    content=(TextBlock(text="ok"), ImageBlock(mime_type="image/png", data=b"x")),
    timestamp=1,
    tool_name="t",
)
