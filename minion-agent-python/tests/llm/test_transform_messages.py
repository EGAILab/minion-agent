"""Target-model transformation (XFORM), reproducing Pi's transformMessages()
(packages/ai/src/api/transform-messages.ts, pinned b7bb00b936dbe21b8e160b3e89efdec361846699).

Covers the matrix canonical YAML cannot efficiently express: every thinking/text/tool-call
compatibility cell, dedup mechanics, orphan-synthesis edge cases, and invariants (immutability,
determinism, vocabulary-validity of output) -- not duplicating what conformance evidence already
proves for the representative cases.
"""

from __future__ import annotations

import dataclasses

from minion_agent.llm.content import ImageBlock, TextBlock, ThinkingBlock, ToolCallBlock
from minion_agent.llm.messages import (
    AssistantMessage,
    StopReason,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from minion_agent.llm.service import ModelId
from minion_agent.llm.transform_messages import TargetModel, transform_messages

SAME = ModelId(provider="p", model="m1", api="a")
OTHER = ModelId(provider="p", model="m2", api="a")

TARGET_TEXT_ONLY = TargetModel(identity=SAME, supports_images=False)
TARGET_VISION = TargetModel(identity=SAME, supports_images=True)


def _assistant(
    content: tuple,
    *,
    provider: str = "p",
    model: str = "m1",
    api: str = "a",
    stop_reason: StopReason = StopReason.STOP,
) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        stop_reason=stop_reason,
        usage=Usage(),
        model=model,
        provider=provider,
        api=api,
        timestamp=1,
    )


def _user(content: tuple, timestamp: int = 1) -> UserMessage:
    return UserMessage(content=content, timestamp=timestamp)


def _result(call_id: str, tool_name: str = "t", is_error: bool = False) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        content=(TextBlock(text="ok"),),
        timestamp=1,
        tool_name=tool_name,
        is_error=is_error,
    )


# --- legacy null content (AI-026) ---------------------------------------------------------


def test_legacy_null_content_normalizes_to_empty_on_every_role() -> None:
    user = dataclasses.replace(_user((TextBlock(text="x"),)), content=None)  # type: ignore[arg-type]
    assistant = dataclasses.replace(_assistant((TextBlock(text="x"),)), content=None)  # type: ignore[arg-type]
    result = dataclasses.replace(_result("t1"), content=None)  # type: ignore[arg-type]

    out = transform_messages([user, assistant, result], TARGET_TEXT_ONLY)

    assert [m.content for m in out] == [(), (), ()]


# --- image downgrade + dedup (AI-020) ------------------------------------------------------


def test_supported_target_leaves_images_untouched() -> None:
    user = _user((ImageBlock(mime_type="image/png", data=b"x"),))
    out = transform_messages([user], TARGET_VISION)
    assert out[0].content == user.content


def test_unsupported_user_image_becomes_the_exact_placeholder() -> None:
    user = _user((ImageBlock(mime_type="image/png", data=b"x"),))
    out = transform_messages([user], TARGET_TEXT_ONLY)
    assert out[0].content == (TextBlock(text="(image omitted: model does not support images)"),)


def test_unsupported_tool_result_image_becomes_the_distinct_exact_placeholder() -> None:
    result = dataclasses.replace(
        _result("t1"), content=(ImageBlock(mime_type="image/png", data=b"x"),)
    )
    out = transform_messages([result], TARGET_TEXT_ONLY)
    assert out[0].content == (
        TextBlock(text="(tool image omitted: model does not support images)"),
    )


def test_adjacent_images_collapse_into_one_placeholder() -> None:
    user = _user(
        (
            ImageBlock(mime_type="image/png", data=b"a"),
            ImageBlock(mime_type="image/png", data=b"b"),
            ImageBlock(mime_type="image/png", data=b"c"),
        )
    )
    out = transform_messages([user], TARGET_TEXT_ONLY)
    assert out[0].content == (TextBlock(text="(image omitted: model does not support images)"),)


def test_images_separated_by_text_each_get_their_own_placeholder() -> None:
    user = _user(
        (
            ImageBlock(mime_type="image/png", data=b"a"),
            TextBlock(text="between"),
            ImageBlock(mime_type="image/png", data=b"b"),
        )
    )
    out = transform_messages([user], TARGET_TEXT_ONLY)
    assert out[0].content == (
        TextBlock(text="(image omitted: model does not support images)"),
        TextBlock(text="between"),
        TextBlock(text="(image omitted: model does not support images)"),
    )


def test_a_preexisting_text_block_matching_the_placeholder_suppresses_the_next_image() -> None:
    """Pi's exact mechanism: `previousWasPlaceholder` is set whenever the emitted block's text
    equals the placeholder, not only when the placeholder was just synthesized -- so a real text
    block that happens to already read the placeholder string suppresses the following image's
    placeholder too."""
    user = _user(
        (
            TextBlock(text="(image omitted: model does not support images)"),
            ImageBlock(mime_type="image/png", data=b"a"),
        )
    )
    out = transform_messages([user], TARGET_TEXT_ONLY)
    assert out[0].content == (TextBlock(text="(image omitted: model does not support images)"),)


# --- thinking compatibility matrix (AI-021) ------------------------------------------------


def test_same_model_signed_thinking_retained_even_when_empty() -> None:
    block = ThinkingBlock(thinking="", thinking_signature="sig")
    out = transform_messages([_assistant((block,))], TARGET_TEXT_ONLY)
    assert out[0].content == (block,)


def test_same_model_unsigned_nonempty_thinking_retained() -> None:
    block = ThinkingBlock(thinking="reasoning")
    out = transform_messages([_assistant((block,))], TARGET_TEXT_ONLY)
    assert out[0].content == (block,)


def test_same_model_unsigned_empty_thinking_removed() -> None:
    block = ThinkingBlock(thinking="   ")
    out = transform_messages([_assistant((block,))], TARGET_TEXT_ONLY)
    assert out[0].content == ()


def test_same_model_redacted_thinking_retained_unchanged() -> None:
    """The gap the frozen spec was missing before this pass: Pi's redacted check runs first and
    returns the block unchanged for a same-model target regardless of signature/text."""
    block = ThinkingBlock(thinking="", thinking_signature="sig", redacted=True)
    out = transform_messages([_assistant((block,))], TARGET_TEXT_ONLY)
    assert out[0].content == (block,)


def test_cross_model_redacted_thinking_omitted_regardless_of_signature() -> None:
    block = ThinkingBlock(thinking="secret", thinking_signature="sig", redacted=True)
    out = transform_messages([_assistant((block,), model="m2")], TARGET_TEXT_ONLY)
    assert out[0].content == ()


def test_cross_model_signed_nonempty_thinking_converts_to_text_and_drops_signature() -> None:
    block = ThinkingBlock(thinking="reasoning", thinking_signature="sig")
    out = transform_messages([_assistant((block,), model="m2")], TARGET_TEXT_ONLY)
    assert out[0].content == (TextBlock(text="reasoning"),)


def test_cross_model_signed_empty_thinking_removed_despite_signature() -> None:
    """Cross-model, the same-model+signature retention check never applies, so an empty signed
    thinking block falls straight through to the empty-removal rule."""
    block = ThinkingBlock(thinking="", thinking_signature="sig")
    out = transform_messages([_assistant((block,), model="m2")], TARGET_TEXT_ONLY)
    assert out[0].content == ()


def test_cross_model_unsigned_nonempty_thinking_converts_to_text() -> None:
    block = ThinkingBlock(thinking="reasoning")
    out = transform_messages([_assistant((block,), model="m2")], TARGET_TEXT_ONLY)
    assert out[0].content == (TextBlock(text="reasoning"),)


def test_cross_model_unsigned_empty_thinking_removed() -> None:
    block = ThinkingBlock(thinking="")
    out = transform_messages([_assistant((block,), model="m2")], TARGET_TEXT_ONLY)
    assert out[0].content == ()


# --- text_signature stripping (AI-022), independent of thinking ----------------------------


def test_same_model_text_signature_retained() -> None:
    block = TextBlock(text="hi", text_signature="sig")
    out = transform_messages([_assistant((block,))], TARGET_TEXT_ONLY)
    assert out[0].content == (block,)


def test_cross_model_text_signature_stripped() -> None:
    block = TextBlock(text="hi", text_signature="sig")
    out = transform_messages([_assistant((block,), model="m2")], TARGET_TEXT_ONLY)
    assert out[0].content == (TextBlock(text="hi"),)


# --- thought_signature stripping (AI-022), independent of text/thinking --------------------


def test_same_model_thought_signature_retained() -> None:
    call = ToolCallBlock(id="c1", name="t", arguments={}, thought_signature="sig", namespace="ns")
    out = transform_messages([_assistant((call,))], TARGET_TEXT_ONLY)
    assert out[0].content == (call,)


def test_cross_model_thought_signature_stripped_but_namespace_and_id_survive() -> None:
    call = ToolCallBlock(
        id="c1", name="t", arguments={"x": 1}, thought_signature="sig", namespace="ns"
    )
    out = transform_messages([_assistant((call,), model="m2")], TARGET_TEXT_ONLY)
    assert out[0].content == (
        ToolCallBlock(
            id="c1", name="t", arguments={"x": 1}, thought_signature=None, namespace="ns"
        ),
    )


def test_cross_model_empty_string_thought_signature_is_not_stripped() -> None:
    """Matches Pi's literal truthy check (`if (!isSameModel && toolCall.thoughtSignature)`): an
    empty string is falsy in JS, so it is left untouched rather than "stripped" to None."""
    call = ToolCallBlock(id="c1", name="t", arguments={}, thought_signature="")
    out = transform_messages([_assistant((call,), model="m2")], TARGET_TEXT_ONLY)
    assert out[0].content == (call,)


# --- tool-call ID normalization (AI-023), cross-model only, generic orchestration only -----


def test_id_normalization_never_applies_same_model() -> None:
    def normalize(call_id: str, target: TargetModel, source: AssistantMessage) -> str:
        return f"norm-{call_id}"

    call = ToolCallBlock(id="orig", name="t", arguments={})
    out = transform_messages([_assistant((call,))], TARGET_TEXT_ONLY, normalize)
    assert out[0].content == (call,)


def test_id_normalization_applies_cross_model_and_rewrites_the_matching_result() -> None:
    def normalize(call_id: str, target: TargetModel, source: AssistantMessage) -> str:
        return f"norm-{call_id}"

    call = ToolCallBlock(id="orig", name="t", arguments={})
    assistant = _assistant((call,), model="m2")
    result = _result("orig")

    out = transform_messages([assistant, result], TARGET_TEXT_ONLY, normalize)

    assert out[0].content == (ToolCallBlock(id="norm-orig", name="t", arguments={}),)
    assert isinstance(out[1], ToolResultMessage)
    assert out[1].tool_call_id == "norm-orig"


def test_an_unrelated_tool_result_id_is_never_accidentally_rewritten() -> None:
    def normalize(call_id: str, target: TargetModel, source: AssistantMessage) -> str:
        return f"norm-{call_id}"

    call = ToolCallBlock(id="orig", name="t", arguments={})
    assistant = _assistant((call,), model="m2")
    unrelated = _result("unrelated")

    out = transform_messages([assistant, unrelated], TARGET_TEXT_ONLY, normalize)

    assert out[1].tool_call_id == "unrelated"


def test_multiple_calls_each_get_independently_normalized_and_matched() -> None:
    def normalize(call_id: str, target: TargetModel, source: AssistantMessage) -> str:
        return f"norm-{call_id}"

    calls = (
        ToolCallBlock(id="a", name="t", arguments={}),
        ToolCallBlock(id="b", name="t", arguments={}),
    )
    assistant = _assistant(calls, model="m2")
    results = (_result("a"), _result("b"))

    out = transform_messages([assistant, *results], TARGET_TEXT_ONLY, normalize)

    assert [c.id for c in out[0].content] == ["norm-a", "norm-b"]
    assert [r.tool_call_id for r in out[1:]] == ["norm-a", "norm-b"]


# --- orphan tool-result synthesis (AI-024) -------------------------------------------------


def test_orphan_synthesized_before_a_later_user_message() -> None:
    call = ToolCallBlock(id="c1", name="lookup", arguments={})
    out = transform_messages(
        [_assistant((call,)), _user((TextBlock(text="hi"),))], TARGET_TEXT_ONLY
    )

    assert isinstance(out[1], ToolResultMessage)
    assert out[1].tool_call_id == "c1"
    assert out[1].tool_name == "lookup"
    assert out[1].is_error is True
    assert out[1].content == (TextBlock(text="No result provided"),)
    assert isinstance(out[2], UserMessage)


def test_orphan_synthesized_before_a_later_assistant_message() -> None:
    call = ToolCallBlock(id="c1", name="lookup", arguments={})
    second = _assistant((TextBlock(text="next"),))
    out = transform_messages([_assistant((call,)), second], TARGET_TEXT_ONLY)

    assert isinstance(out[1], ToolResultMessage)
    assert out[1].tool_call_id == "c1"
    assert out[2] == second  # not `is`: the content pass always rebuilds assistant messages,
    # matching Pi's own `{...assistantMsg, content: transformedContent}` (always a new object)


def test_orphan_synthesized_at_end_of_history() -> None:
    call = ToolCallBlock(id="c1", name="lookup", arguments={})
    out = transform_messages([_assistant((call,))], TARGET_TEXT_ONLY)

    assert len(out) == 2
    assert isinstance(out[1], ToolResultMessage)
    assert out[1].tool_call_id == "c1"


def test_no_synthesis_when_a_real_result_already_exists() -> None:
    call = ToolCallBlock(id="c1", name="lookup", arguments={})
    out = transform_messages([_assistant((call,)), _result("c1")], TARGET_TEXT_ONLY)

    assert len(out) == 2
    assert out[1].content == (TextBlock(text="ok"),)  # the real result, not a synthetic one


def test_multiple_unresolved_calls_each_get_their_own_synthetic_result_in_source_order() -> None:
    calls = (
        ToolCallBlock(id="a", name="first", arguments={}),
        ToolCallBlock(id="b", name="second", arguments={}),
    )
    out = transform_messages([_assistant(calls)], TARGET_TEXT_ONLY)

    assert [m.tool_call_id for m in out[1:]] == ["a", "b"]
    assert [m.tool_name for m in out[1:]] == ["first", "second"]


def test_errored_assistants_own_tool_calls_are_never_synthesized() -> None:
    """The subtle case: a discarded errored/aborted assistant's tool calls are simply dropped,
    never tracked as pending, so they never receive a synthetic result either."""
    call = ToolCallBlock(id="c1", name="lookup", arguments={})
    errored = _assistant((call,), stop_reason=StopReason.ERROR)
    out = transform_messages([errored, _user((TextBlock(text="hi"),))], TARGET_TEXT_ONLY)

    assert len(out) == 1
    assert isinstance(out[0], UserMessage)


# --- historical error/aborted exclusion (AI-025) -------------------------------------------


def test_errored_assistant_excluded_from_replay() -> None:
    assistant = _assistant((TextBlock(text="x"),), stop_reason=StopReason.ERROR)
    out = transform_messages([assistant], TARGET_TEXT_ONLY)
    assert out == ()


def test_aborted_assistant_excluded_from_replay() -> None:
    assistant = _assistant((TextBlock(text="x"),), stop_reason=StopReason.ABORTED)
    out = transform_messages([assistant], TARGET_TEXT_ONLY)
    assert out == ()


def test_only_error_and_aborted_are_excluded_not_other_stop_reasons() -> None:
    other_reasons = (
        StopReason.STOP,
        StopReason.LENGTH,
        StopReason.TOOL_USE,
        StopReason.PENDING,
        StopReason.DEFERRED,
    )
    for reason in other_reasons:
        assistant = _assistant((TextBlock(text="x"),), stop_reason=reason)
        out = transform_messages([assistant], TARGET_TEXT_ONLY)
        assert len(out) == 1, f"{reason} must not be excluded"


# --- invariants -----------------------------------------------------------------------------


def test_source_messages_are_never_mutated() -> None:
    call = ToolCallBlock(id="c1", name="t", arguments={}, thought_signature="sig")
    assistant = _assistant((call,), model="m2")
    before = dataclasses.replace(assistant)

    transform_messages([assistant], TARGET_TEXT_ONLY)

    assert assistant == before


def _strip_synthetic_timestamps(messages: tuple) -> list:
    return [
        dataclasses.replace(m, timestamp=0) if isinstance(m, ToolResultMessage) else m
        for m in messages
    ]


def test_output_is_deterministic_for_identical_input_apart_from_synthetic_timestamps() -> None:
    call = ToolCallBlock(id="c1", name="t", arguments={})
    messages = [_assistant((call,))]

    first = transform_messages(messages, TARGET_TEXT_ONLY)
    second = transform_messages(messages, TARGET_TEXT_ONLY)

    assert _strip_synthetic_timestamps(first) == _strip_synthetic_timestamps(second)


def test_non_image_capable_target_output_contains_no_image_block() -> None:
    user = _user((ImageBlock(mime_type="image/png", data=b"x"), TextBlock(text="hi")))
    out = transform_messages([user], TARGET_TEXT_ONLY)
    assert all(not isinstance(block, ImageBlock) for message in out for block in message.content)


def test_a_role_invalid_block_the_schema_forbids_still_passes_through_defensively() -> None:
    """SES-F005's per-role content union forbids ImageBlock in assistant content, but nothing at
    the Python dataclass level enforces that -- only the conformance schema does. Matches Pi's own
    defensive `return block` fallback for a content type its thinking/text/toolCall branches don't
    handle, rather than crashing on a state the type system alone cannot prevent."""
    image = ImageBlock(mime_type="image/png", data=b"x")
    out = transform_messages([_assistant((image,))], TARGET_TEXT_ONLY)
    assert out[0].content == (image,)


def test_cross_model_output_carries_no_forbidden_signatures() -> None:
    text = TextBlock(text="hi", text_signature="sig")
    call = ToolCallBlock(id="c1", name="t", arguments={}, thought_signature="sig")
    out = transform_messages([_assistant((text, call), model="m2")], TARGET_TEXT_ONLY)

    for block in out[0].content:
        assert getattr(block, "text_signature", None) is None
        assert getattr(block, "thought_signature", None) is None
