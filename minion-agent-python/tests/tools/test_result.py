"""What a tool returns, and what survives into model history."""

import dataclasses
from dataclasses import replace

import pytest

from minion_agent.llm import TextBlock, text_of
from minion_agent.tools.result import text_result


def test_a_result_defaults_to_success() -> None:
    result = text_result("t1", "done")

    assert not result.is_error
    assert not result.terminate
    assert result.added_tool_names == ()
    assert result.details == {}


def test_a_result_converts_to_the_message_the_model_sees() -> None:
    message = text_result("t1", "done").to_message()

    assert message.tool_call_id == "t1"
    assert text_of(message) == "done"
    assert not message.is_error


def test_details_do_not_reach_the_model() -> None:
    """Details are for listeners and telemetry. Putting them in front of the
    model would make every audit annotation part of the transcript."""
    result = replace(text_result("t1", "done"), details={"audited": True})

    assert "audited" not in text_of(result.to_message())


def test_terminate_does_not_reach_the_model_either() -> None:
    """It is a loop instruction, not something the model said."""
    message = text_result("t1", "stopping", terminate=True).to_message()

    assert text_of(message) == "stopping"


def test_an_error_result_is_still_a_result() -> None:
    result = text_result("t1", "boom", is_error=True)

    assert result.is_error
    assert result.to_message().is_error


def test_results_are_frozen_so_transformation_is_explicit() -> None:
    """post-execute replaces fields; it never mutates in place, or a listener
    could change a result another listener already returned. Frozen-ness
    enforces this: replace() constructs a new instance, and direct mutation
    is refused."""
    original = text_result("t1", "one")

    transformed = replace(original, content=(TextBlock(text="two"),))

    assert text_of(original.to_message()) == "one"
    assert text_of(transformed.to_message()) == "two"

    # Verify frozen-ness: direct mutation must fail.
    with pytest.raises(dataclasses.FrozenInstanceError):
        original.tool_call_id = "changed"  # type: ignore[misc]


def test_added_tool_names_are_carried() -> None:
    result = replace(text_result("t1", "loaded"), added_tool_names=("deploy",))

    assert result.added_tool_names == ("deploy",)
