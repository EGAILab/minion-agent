"""Target-model message transformation (XFORM).

Reproduces Pi's `transformMessages()` (`packages/ai/src/api/transform-messages.ts`, pinned
`b7bb00b936dbe21b8e160b3e89efdec361846699`): the stage between session-derived, provider-neutral
`Message` history and one specific target model's request, deciding what survives, what gets
downgraded or stripped, and what gets synthesized for that target's identity and capability.

Runs after session projection (design spec section 5's own explicit sequencing) and before a
target's provider encoder. Owns only generic compatibility transformation: the concrete
tool-call-ID normalization algorithm a specific target API requires is Phase-5/provider territory
(`AI-023`) -- Pi itself never hardcodes one, it takes `normalizeToolCallId` as an injected
per-adapter callback, and every one of its real call sites supplies its own. This module reproduces
exactly that shape: it owns the generic map-building/consistent-rewrite orchestration, nothing more.

Not yet wired into a production request path: no real (non-mock) provider adapter exists yet
(Phase 5 deferred, matching Layer 02's own `LLM-012`/`LLM-020`/`LLM-021` disposition for the same
reason) -- the mock adapter has no real target-model wire constraints to reconcile against. This is
a complete, tested, real library function, exercised directly by conformance evidence and unit
tests, waiting for a real caller.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence

from .content import ContentBlock, ImageBlock, TextBlock, ThinkingBlock, ToolCallBlock
from .messages import AssistantMessage, Message, StopReason, ToolResultMessage, UserMessage
from .service import ModelId

_NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
_NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"
_ORPHAN_TOOL_RESULT_TEXT = "No result provided"


@dataclasses.dataclass(frozen=True, slots=True)
class TargetModel:
    """What `transform_messages()` needs to know about the model a request is being prepared for.

    Not a full provider/model-registry entry -- Pi's own `Model<TApi>` carries `baseUrl`/`cost`/
    `contextWindow`/etc. that `transformMessages()` itself never reads. This carries only the
    frozen `provider + api + model_id` identity triple (reused from `ModelId`, not reinvented) plus
    the one capability flag the function's image-downgrade rule consults.
    """

    identity: ModelId
    supports_images: bool
    """Pi's `model.input.includes("image")`."""


NormalizeToolCallId = Callable[[str, TargetModel, AssistantMessage], str]
"""A target-API-supplied tool-call-ID normalizer, matching Pi's injected `normalizeToolCallId`
parameter exactly: `transform_messages()` never hardcodes an algorithm, only invokes this one,
cross-model only, and consistently rewrites the matching `ToolResultMessage.tool_call_id`."""


def _now_ms() -> int:
    """Wall-clock milliseconds, matching Pi's `Date.now()` for a synthesized result's timestamp.

    Not asserted by canonical evidence for that reason -- see `spec/target-model-transformation.md`.
    """
    return int(time.time() * 1000)


def _normalize_legacy_content(message: Message) -> Message:
    """Untyped/legacy callers (custom tools, hand-built histories, old session files -- Pi's own
    comment) may hand this function a `Message` whose `content` is `None` despite the static type
    forbidding it: Python does not enforce dataclass field types at runtime. Pi normalizes this
    first, unconditionally, for every role; so does this function."""
    if message.content is None:
        return dataclasses.replace(message, content=())
    return message


def _replace_images_with_placeholder(
    content: tuple[ContentBlock, ...], placeholder: str
) -> tuple[ContentBlock, ...]:
    result: list[ContentBlock] = []
    previous_was_placeholder = False
    for block in content:
        if isinstance(block, ImageBlock):
            if not previous_was_placeholder:
                result.append(TextBlock(text=placeholder))
            previous_was_placeholder = True
            continue
        result.append(block)
        previous_was_placeholder = isinstance(block, TextBlock) and block.text == placeholder
    return tuple(result)


def _downgrade_unsupported_images(
    messages: tuple[Message, ...], target: TargetModel
) -> tuple[Message, ...]:
    if target.supports_images:
        return messages
    downgraded: list[Message] = []
    for message in messages:
        if isinstance(message, UserMessage):
            # UserMessage.content is string | tuple[ContentBlock, ...] (spec/llm.md; Pi's own
            # Array.isArray(msg.content) guard). A string carries no image blocks by
            # construction, so it passes through untouched -- treating it as an iterable of
            # blocks would corrupt it into a tuple of individual characters.
            if isinstance(message.content, str):
                downgraded.append(message)
            else:
                downgraded.append(
                    dataclasses.replace(
                        message,
                        content=_replace_images_with_placeholder(
                            message.content, _NON_VISION_USER_IMAGE_PLACEHOLDER
                        ),
                    )
                )
        elif isinstance(message, ToolResultMessage):
            downgraded.append(
                dataclasses.replace(
                    message,
                    content=_replace_images_with_placeholder(
                        message.content, _NON_VISION_TOOL_IMAGE_PLACEHOLDER
                    ),
                )
            )
        else:
            downgraded.append(message)
    return tuple(downgraded)


def _is_same_model(assistant: AssistantMessage, target: TargetModel) -> bool:
    return (
        assistant.provider == target.identity.provider
        and assistant.api == target.identity.api
        and assistant.model == target.identity.model
    )


def _transform_thinking(block: ThinkingBlock, same_model: bool) -> tuple[ContentBlock, ...]:
    if block.redacted:
        # Redacted thinking is opaque encrypted content, only valid for the same model.
        return (block,) if same_model else ()
    if same_model and block.thinking_signature:
        # Signed same-model thinking is kept for replay, even if the text itself is empty
        # (OpenAI encrypted reasoning carries no visible text).
        return (block,)
    if not block.thinking or not block.thinking.strip():
        return ()
    if same_model:
        return (block,)
    return (TextBlock(text=block.thinking),)


def _transform_tool_call(
    call: ToolCallBlock,
    same_model: bool,
    target: TargetModel,
    assistant: AssistantMessage,
    normalize_tool_call_id: NormalizeToolCallId | None,
    tool_call_id_map: dict[str, str],
) -> ToolCallBlock:
    if not same_model and call.thought_signature:
        # Truthy check, matching Pi's `if (!isSameModel && toolCall.thoughtSignature)` exactly:
        # an empty-string signature is left untouched, not stripped.
        call = dataclasses.replace(call, thought_signature=None)
    if not same_model and normalize_tool_call_id is not None:
        normalized_id = normalize_tool_call_id(call.id, target, assistant)
        if normalized_id != call.id:
            tool_call_id_map[call.id] = normalized_id
            call = dataclasses.replace(call, id=normalized_id)
    return call


def _transform_assistant_content(
    assistant: AssistantMessage,
    target: TargetModel,
    normalize_tool_call_id: NormalizeToolCallId | None,
    tool_call_id_map: dict[str, str],
) -> tuple[ContentBlock, ...]:
    same_model = _is_same_model(assistant, target)
    transformed: list[ContentBlock] = []
    for block in assistant.content:
        if isinstance(block, ThinkingBlock):
            transformed.extend(_transform_thinking(block, same_model))
        elif isinstance(block, TextBlock):
            transformed.append(block if same_model else TextBlock(text=block.text))
        elif isinstance(block, ToolCallBlock):
            transformed.append(
                _transform_tool_call(
                    block, same_model, target, assistant, normalize_tool_call_id, tool_call_id_map
                )
            )
        else:
            transformed.append(block)
    return tuple(transformed)


def _content_transform_pass(
    messages: tuple[Message, ...],
    target: TargetModel,
    normalize_tool_call_id: NormalizeToolCallId | None,
) -> tuple[Message, ...]:
    """First pass: content transform (thinking/text/tool-call) plus tool-call-ID normalization
    for assistant messages, and the matching `tool_call_id` rewrite for tool-result messages --
    all in one forward pass, since a result can only be rewritten once its call's mapping already
    exists (results always follow their calls in transcript order)."""
    tool_call_id_map: dict[str, str] = {}
    transformed: list[Message] = []
    for message in messages:
        if isinstance(message, UserMessage):
            transformed.append(message)
        elif isinstance(message, ToolResultMessage):
            normalized_id = tool_call_id_map.get(message.tool_call_id)
            if normalized_id is not None and normalized_id != message.tool_call_id:
                transformed.append(dataclasses.replace(message, tool_call_id=normalized_id))
            else:
                transformed.append(message)
        else:
            transformed.append(
                dataclasses.replace(
                    message,
                    content=_transform_assistant_content(
                        message, target, normalize_tool_call_id, tool_call_id_map
                    ),
                )
            )
    return tuple(transformed)


def _orphan_synthesis_pass(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    """Second pass: drop historical errored/aborted assistants entirely (their own tool calls
    included, uncounted and unsynthesized), and synthesize an error result for any tool call left
    unresolved when a later user/assistant message interrupts, or when history ends."""
    result: list[Message] = []
    pending_calls: list[ToolCallBlock] = []
    resolved_ids: set[str] = set()

    def flush_orphans() -> None:
        for call in pending_calls:
            if call.id not in resolved_ids:
                result.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        content=(TextBlock(text=_ORPHAN_TOOL_RESULT_TEXT),),
                        timestamp=_now_ms(),
                        tool_name=call.name,
                        is_error=True,
                    )
                )
        pending_calls.clear()
        resolved_ids.clear()

    for message in messages:
        if isinstance(message, AssistantMessage):
            flush_orphans()
            if message.stop_reason in (StopReason.ERROR, StopReason.ABORTED):
                continue
            calls = [block for block in message.content if isinstance(block, ToolCallBlock)]
            if calls:
                pending_calls[:] = calls
                resolved_ids.clear()
            result.append(message)
        elif isinstance(message, ToolResultMessage):
            resolved_ids.add(message.tool_call_id)
            result.append(message)
        else:
            flush_orphans()
            result.append(message)

    flush_orphans()
    return tuple(result)


def transform_messages(
    messages: Sequence[Message],
    target: TargetModel,
    normalize_tool_call_id: NormalizeToolCallId | None = None,
) -> tuple[Message, ...]:
    """Transform `messages` for dispatch to `target`.

    Pipeline order is normative (observable): legacy-null-content normalization, then unsupported-
    image downgrade, then per-message content transform (thinking/text/tool-call handling plus
    tool-call-ID normalization), then orphan-tool-result synthesis and errored/aborted-assistant
    exclusion. Changing this order changes output for real inputs (design spec's own ordering rule).
    """
    normalized = tuple(_normalize_legacy_content(message) for message in messages)
    image_aware = _downgrade_unsupported_images(normalized, target)
    content_transformed = _content_transform_pass(image_aware, target, normalize_tool_call_id)
    return _orphan_synthesis_pass(content_transformed)
