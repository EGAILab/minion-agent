"""Layer 08, PASS 2: prompt()/continue_() entry points, run-local
prepareNextTurn overrides, and pinned pi's handleRunFailure fallback."""

from typing import Any

import pytest

from minion_agent.agent.decisions import Enter, Reject, RunConfigUpdate, TurnStopping
from minion_agent.agent.events import AGENT_PRE_STEP, AGENT_PREPARE_NEXT_TURN, AGENT_TURN_STOPPING
from minion_agent.agent.identity import AgentStatus, ThinkingLevel
from minion_agent.agent.instance import AgentActiveError
from minion_agent.agent_loop.driver import _RunSnapshot
from minion_agent.llm import (
    AssistantMessage,
    ModelId,
    TextBlock,
    ToolCallBlock,
    UserMessage,
    text_of,
)
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind, derive_messages

from .test_single_turn import _loop, _loop_with_adapter, _register, _say


def test_run_snapshot_apply_overrides_only_the_given_fields() -> None:
    snapshot = _RunSnapshot(
        system_prompt="original", model=ModelId("mock", "mock-1"), thinking_level=ThinkingLevel.OFF
    )

    snapshot.apply(RunConfigUpdate())
    assert (snapshot.system_prompt, snapshot.model, snapshot.thinking_level) == (
        "original",
        ModelId("mock", "mock-1"),
        ThinkingLevel.OFF,
    )

    override_model = ModelId("mock", "mock-2")
    snapshot.apply(RunConfigUpdate(model=override_model, thinking_level=ThinkingLevel.HIGH))
    assert snapshot.system_prompt == "original"
    assert snapshot.model == override_model
    assert snapshot.thinking_level is ThinkingLevel.HIGH


async def test_prompt_starts_a_run_with_the_given_message() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))

    await loop.prompt(_say("hello"))

    assert [text_of(m) for m in derive_messages(loop.instance.log)] == ["hello", "hi"]
    assert loop.instance.status is AgentStatus.IDLE


async def test_prompt_accepts_a_tuple_of_messages() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))

    await loop.prompt((_say("one"), _say("two")))

    assert [text_of(m) for m in derive_messages(loop.instance.log)][:2] == ["one", "two"]


async def test_prompt_rejects_while_active() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.set_status(AgentStatus.RUNNING)

    with pytest.raises(AgentActiveError, match="Use steer\\(\\) or"):
        await loop.prompt(_say("hello"))


async def test_continue_rejects_while_active() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.set_status(AgentStatus.RUNNING)

    with pytest.raises(AgentActiveError, match="Wait for completion before continuing"):
        await loop.continue_()


async def test_continue_rejects_with_no_transcript() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))

    with pytest.raises(AgentActiveError, match="No messages to continue from"):
        await loop.continue_()


async def test_continue_rejects_when_last_message_is_assistant_and_nothing_queued() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    await loop.prompt(_say("hello"))

    with pytest.raises(AgentActiveError, match="Cannot continue from message role: assistant"):
        await loop.continue_()


async def test_continue_drains_steering_when_last_message_is_assistant() -> None:
    """The steering-pre-drain path (pinned pi's `skipInitialSteeringPoll`):
    the drained batch enters directly, without a duplicate initial poll."""
    loop = _loop(
        ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP),
        ScriptedResponse((TextBlock(text="steered reply"),), StopReason.STOP),
    )
    await loop.prompt(_say("hello"))
    loop.instance.inbox.steer(_say("steer me"), origin="s1")

    await loop.continue_()

    texts = [text_of(m) for m in derive_messages(loop.instance.log)]
    assert texts == ["hello", "hi", "steer me", "steered reply"]


async def test_continue_does_not_double_drain_steering_after_pre_drain() -> None:
    """The pre-drained steering batch must not also be claimed again by this
    same continuation's own initial steering poll."""
    loop = _loop(
        ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP),
        ScriptedResponse((TextBlock(text="steered reply"),), StopReason.STOP),
    )
    await loop.prompt(_say("hello"))
    loop.instance.inbox.steer(_say("steer me"))

    await loop.continue_()

    user_texts = [
        text_of(m) for m in derive_messages(loop.instance.log) if isinstance(m, UserMessage)
    ]
    assert user_texts.count("steer me") == 1


async def test_continue_drains_follow_up_when_last_message_is_assistant_and_no_steering() -> None:
    loop = _loop(
        ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP),
        ScriptedResponse((TextBlock(text="follow-up reply"),), StopReason.STOP),
    )
    await loop.prompt(_say("hello"))
    loop.instance.inbox.followup(_say("follow up"), origin="f1")

    await loop.continue_()

    texts = [text_of(m) for m in derive_messages(loop.instance.log)]
    assert texts == ["hello", "hi", "follow up", "follow-up reply"]


async def test_continue_sends_full_history_when_last_message_is_not_assistant() -> None:
    """Pinned pi's plain continuation (`runAgentLoopContinue`): no new
    message enters, but full history is still sent to the model. Forces the
    first run to pause right after the tool-call turn (a `shouldStopAfterTurn`
    listener), so the transcript's own last message is a tool_result, not
    assistant, when `continue_()` is called."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="continued"),), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: "pong")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.STOP)
    await loop.prompt(_say("hello"))
    before = derive_messages(loop.instance.log)
    assert not isinstance(before[-1], AssistantMessage)

    await loop.continue_()

    after = derive_messages(loop.instance.log)
    assert len(after) == len(before) + 1  # only the new assistant reply, no new user message
    assert text_of(after[-1]) == "continued"
    assert len(adapter.requests[-1].messages) == len(before)  # full prior history was sent


async def test_run_wrapped_defensive_guard_rejects_when_not_idle() -> None:
    """Pinned pi's own `runWithLifecycle` internal guard: a third, distinct
    string, defensive against a caller that bypasses `prompt()`/`continue_()`'s
    own public checks."""
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.set_status(AgentStatus.RUNNING)

    with pytest.raises(AgentActiveError, match=r"^Agent is already processing\.$"):
        await loop._run_wrapped(entering=(), causes=[])


async def test_prepare_next_turn_can_override_system_prompt_run_locally() -> None:
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="second"),), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: "pong")
    original_system = loop.instance.system_prompt

    async def prepare(instance: Any, outcome: Any, next_: Any) -> RunConfigUpdate:
        return RunConfigUpdate(system_prompt="run-local override")

    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, prepare)
    await loop.prompt(_say("go"))

    assert adapter.requests[0].system != adapter.requests[1].system
    # Never persisted back to the certified Layer-07 instance.
    assert loop.instance.system_prompt == original_system


async def test_prepare_next_turn_default_is_a_pure_pass_through() -> None:
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="second"),), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: "pong")

    await loop.prompt(_say("go"))

    assert adapter.requests[0].model == adapter.requests[1].model


async def test_a_pre_step_listener_failure_settles_gracefully() -> None:
    """Pinned pi's `handleRunFailure`: an unexpected exception from
    `AGENT_PRE_STEP` dispatch does not escape `prompt()`."""

    async def boom(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise RuntimeError("listener exploded")

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, boom)

    await loop.prompt(_say("hello"))  # must not raise

    assert loop.instance.status is AgentStatus.IDLE
    assert loop.instance.error_message == "listener exploded"
    kinds = [e.kind for e in loop.instance.log.events]
    assert kinds[-1] == EventKind.AGENT_END
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "failed"
    assistant = [
        m for m in derive_messages(loop.instance.log) if isinstance(m, AssistantMessage)
    ]
    assert assistant and assistant[-1].stop_reason is StopReason.ERROR
    assert assistant[-1].error_message == "listener exploded"


async def test_a_turn_stopping_listener_failure_settles_gracefully() -> None:
    def boom(*args: Any) -> TurnStopping:
        raise RuntimeError("stopping listener exploded")

    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, boom)

    await loop.prompt(_say("hello"))  # must not raise

    assert loop.instance.error_message == "stopping listener exploded"
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "failed"


async def test_a_prepare_next_turn_listener_failure_settles_gracefully() -> None:
    async def boom(instance: Any, outcome: Any, next_: Any) -> RunConfigUpdate:
        raise RuntimeError("prepare listener exploded")

    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, boom)

    await loop.prompt(_say("hello"))  # must not raise

    assert loop.instance.error_message == "prepare listener exploded"
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "failed"


async def test_an_unresolvable_model_still_raises_uncaught() -> None:
    """The failure-settling fallback is narrowly scoped to loop-callback
    dispatch (`AGENT_PRE_STEP`/`AGENT_TURN_STOPPING`/`AGENT_PREPARE_NEXT_TURN`)
    -- a provider/model-resolution failure is a different, pre-existing
    contract (`eager-invalid-model-fails-before-stream.yaml`) and must still
    propagate, not be smuggled into a settled failure turn."""

    class _Boom(Exception):
        pass

    async def boom(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise _Boom("not a loop-callback failure in the pi sense")

    loop = _loop(ScriptedResponse((), StopReason.STOP))

    # Simulate a non-callback failure by making the pre-step machinery itself
    # raise from outside the driver's own three dispatch wrappers: patch
    # `_run_step` directly instead, since that is the real non-callback path
    # (LLM/tool execution) this fallback must not catch.
    async def failing_run_step(decision: Any, reason: Any, snapshot: Any) -> None:
        raise _Boom("model resolution failed")

    loop._run_step = failing_run_step  # type: ignore[method-assign]
    loop.instance.inbox.followup(_say("hello"))

    with pytest.raises(_Boom):
        await loop.run_until_idle()


async def test_a_rejected_follow_up_continuation_ends_the_run() -> None:
    """A listener rejecting the entering messages of a follow-up-triggered
    mid-run continuation (not the run's own initial pre-step) still ends the
    run cleanly, with the rejection detail recorded."""
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    seen_reasons: list[str] = []

    async def veto_second_only(instance: Any, reason: Any, messages: Any, next_: Any) -> Any:
        seen_reasons.append(reason.value)
        if reason.value == "next_turn":
            return Reject(reason="not now")
        return await next_(instance, reason, messages)

    loop.instance.ctx.events.on(AGENT_PRE_STEP, veto_second_only)
    loop.instance.inbox.followup(_say("first"))
    loop.instance.inbox.followup(_say("second"))

    await loop.run_until_idle()

    assert "next_turn" in seen_reasons
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "rejected"
    assert end.data["detail"] == "not now"
