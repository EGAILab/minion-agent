"""Layer 08, PASS 3: remediation for the independent Rust contract rejection
(L08-R001..R008) -- prompt()/continue_() entry points, the complete run-local
RunContext/RunConfig snapshot and whole-context prepareNextTurn, pinned pi's
real handleRunFailure catch boundary, full streaming_message fidelity, the
initial-steering-poll ordering fix, and represented error/aborted terminal
handling."""

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest

from minion_agent.agent.decisions import Enter, Reject, RunConfigUpdate, RunContext, TurnStopping
from minion_agent.agent.events import (
    AGENT_LIFECYCLE_EVENT,
    AGENT_PRE_STEP,
    AGENT_PREPARE_NEXT_TURN,
    AGENT_TURN_STOPPING,
)
from minion_agent.agent.identity import AgentStatus, ThinkingLevel
from minion_agent.agent.instance import AgentActiveError
from minion_agent.agent.projection import (
    AgentEnd,
    AgentStart,
    MessageEnd,
    MessageStart,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    TurnEnd,
)
from minion_agent.llm import (
    AssistantMessage,
    ImageBlock,
    ModelId,
    Request,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    Usage,
    UserMessage,
    text_of,
)
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.errors import UnknownModelError
from minion_agent.llm.stream import (
    StreamChunk,
    StreamDone,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from minion_agent.session import EventKind, derive_messages
from minion_agent.tools.definition import ExecutionMode

from .test_single_turn import _loop, _loop_with_adapter, _register, _say


async def test_prompt_starts_a_run_with_the_given_message() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))

    await loop.prompt(_say("hello"))

    assert [text_of(m) for m in derive_messages(loop.instance.log)] == ["hello", "hi"]
    assert loop.instance.status is AgentStatus.IDLE


async def test_prompt_accepts_a_tuple_of_typed_messages_unchanged() -> None:
    """`L08-R007`: the typed `Message | tuple[Message, ...]` boundary is not
    narrowed by adding the convenience `str` form."""
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))

    await loop.prompt((_say("one"), _say("two")))

    assert [text_of(m) for m in derive_messages(loop.instance.log)][:2] == ["one", "two"]


async def test_prompt_accepts_a_plain_string() -> None:
    """`L08-R007`: pinned pi's `prompt(text: string)` convenience overload."""
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))

    await loop.prompt("hello")

    messages = derive_messages(loop.instance.log)
    assert isinstance(messages[0], UserMessage)
    assert messages[0].content == (TextBlock(text="hello"),)


async def test_prompt_string_with_one_image_orders_text_then_image() -> None:
    """`L08-R007`: pinned pi's `[{type:"text",...}, ...images]` construction --
    text first, then the supplied images, in order."""
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    image = ImageBlock(mime_type="image/png", data=b"Zm9v")

    await loop.prompt("describe this", images=(image,))

    messages = derive_messages(loop.instance.log)
    assert isinstance(messages[0], UserMessage)
    assert messages[0].content == (TextBlock(text="describe this"), image)


async def test_prompt_string_with_multiple_images_preserves_order() -> None:
    """`L08-R007`."""
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    first = ImageBlock(mime_type="image/png", data=b"AAAA")
    second = ImageBlock(mime_type="image/png", data=b"BBBB")

    await loop.prompt("compare these", images=(first, second))

    messages = derive_messages(loop.instance.log)
    assert isinstance(messages[0], UserMessage)
    assert messages[0].content == (TextBlock(text="compare these"), first, second)


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


# -- L08-R001: complete run snapshot and whole-context prepareNextTurn -------


async def test_run_start_snapshot_ignores_a_later_caller_config_change() -> None:
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="second"),), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: "pong")

    async def mutate_mid_run(instance: Any, *args: Any, **kwargs: Any) -> RunConfigUpdate:
        instance.system_prompt = "mutated mid-run"
        return RunConfigUpdate()

    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, mutate_mid_run)
    await loop.prompt(_say("go"))

    # Both requests of this SAME run still used the snapshot taken at run
    # start, not the mid-run mutation.
    assert adapter.requests[0].system == adapter.requests[1].system


async def test_run_start_snapshot_ignores_session_changes_from_outside_the_run() -> None:
    """A message appended to the run-local context is the run's own turn
    output, not a re-derivation from the log -- proven by mutating the log
    directly mid-run (simulating an external append) and confirming the next
    request within the same run does not see it."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="second"),), StopReason.STOP),
    )

    def inject_externally(tool_call_id: str, args: dict[str, object]) -> str:
        from minion_agent.session import EventKind as _EK
        from minion_agent.session import encode_message as _encode

        loop.instance.log.append(_EK.USER_MESSAGE, {"message": _encode(_say("external injection"))})
        return "pong"

    _register(loop, "echo", inject_externally)
    await loop.prompt(_say("go"))

    second_request_texts = [text_of(m) for m in adapter.requests[1].messages]
    assert "external injection" not in second_request_texts


async def test_prepare_next_turn_can_replace_the_whole_context() -> None:
    """`L08-R001`: `RunConfigUpdate.context` replaces `system_prompt`/
    `messages`/`tools` wholesale for the next request only -- pinned pi's own
    `currentContext = nextTurnSnapshot.context ?? currentContext`."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="second"),), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: "pong")
    replacement_message = _say("replacement history")

    async def prepare(
        instance: Any,
        message: Any,
        tool_results: Any,
        context: RunContext,
        new_messages: Any,
        next_: Any,
    ) -> RunConfigUpdate:
        return RunConfigUpdate(
            context=RunContext(
                system_prompt="replaced system prompt",
                messages=[replacement_message],
                tools=context.tools,
            )
        )

    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, prepare)
    await loop.prompt(_say("go"))

    second_request = adapter.requests[1]
    assert [text_of(m) for m in second_request.messages] == ["replacement history"]
    assert second_request.system != adapter.requests[0].system


class _MultiModelAdapter:
    """Test-only adapter serving two model ids, so a `RunConfigUpdate.model`
    replacement (`L08-R001`) has somewhere to actually land -- the certified
    `MockAdapter` only ever serves `mock-1`."""

    provider = "mock"
    api = "mock"
    models = frozenset({"mock-1", "mock-2"})

    def __init__(self, *responses: AssistantMessage) -> None:
        self._responses = list(responses)
        self.requests: list[Request] = []

    def stream(self, request: Request) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        message = self._responses.pop(0)

        async def replay() -> AsyncIterator[StreamChunk]:
            yield StreamStart(partial=message)
            yield StreamDone(message=message, partial=message)

        return replay()


def _reply(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text=text),),
        stop_reason=StopReason.STOP,
        usage=Usage(),
        model="mock-1",
        provider="mock",
        timestamp=0,
    )


async def test_prepare_next_turn_can_replace_model_and_thinking_level() -> None:
    """`L08-R001`: `model`/`thinking_level` are independently optional
    replacements applied to the run-local `RunConfig`, distinct from
    `context`'s own whole-object replacement."""
    from minion_agent.agent.identity import AgentDefinition
    from minion_agent.agent.registry import AgentRegistry
    from minion_agent.llm import LlmService
    from minion_agent.runtime import Context
    from minion_agent.session import SessionService
    from minion_agent.tools.events import declare_tools_events
    from minion_agent.tools.registry import ToolRegistry

    ctx = Context()
    declare_tools_events(ctx.events)
    sessions = SessionService()
    llm = LlmService()
    adapter = _MultiModelAdapter(
        AssistantMessage(
            content=(ToolCallBlock(id="t1", name="echo", arguments={}),),
            stop_reason=StopReason.TOOL_USE,
            usage=Usage(),
            model="mock-1",
            provider="mock",
            timestamp=0,
        ),
        _reply("second"),
    )
    llm.register(adapter)
    registry = AgentRegistry(ctx=ctx, sessions=sessions)
    handle = registry.create("room-a", AgentDefinition(name="ada", model=ModelId("mock", "mock-1")))
    from minion_agent.agent_loop.driver import AgentLoop

    loop = AgentLoop(
        instance=handle.instance, llm=llm, tools=ToolRegistry(), artifacts=sessions.artifacts
    )
    _register(loop, "echo", lambda tool_call_id, args: "pong")
    other_model = ModelId("mock", "mock-2")

    async def prepare(instance: Any, *args: Any, **kwargs: Any) -> RunConfigUpdate:
        return RunConfigUpdate(model=other_model, thinking_level=ThinkingLevel.HIGH)

    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, prepare)
    await loop.prompt(_say("go"))

    assert adapter.requests[0].model == ModelId("mock", "mock-1")
    assert adapter.requests[1].model == other_model


async def test_prepare_next_turn_context_replacement_does_not_persist_to_the_next_run() -> None:
    """The replacement is run-local only -- a second, independent run starts
    from a fresh snapshot again, not the previous run's replaced context."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="first done"),), StopReason.STOP),
        ScriptedResponse((TextBlock(text="second done"),), StopReason.STOP),
    )
    _register(loop, "echo", lambda tool_call_id, args: "pong")

    async def prepare_once(
        instance: Any,
        message: Any,
        tool_results: Any,
        context: RunContext,
        new_messages: Any,
        next_: Any,
    ) -> RunConfigUpdate:
        dispose()
        return RunConfigUpdate(
            context=RunContext(
                system_prompt="only for the first run",
                messages=list(context.messages),
                tools=context.tools,
            )
        )

    dispose = loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, prepare_once)
    await loop.prompt(_say("first"))
    await loop.prompt(_say("second"))

    # The second run's own first request must not carry the first run's
    # replaced system prompt.
    assert adapter.requests[2].system != adapter.requests[1].system


async def test_added_tool_names_already_visible_is_not_duplicated() -> None:
    """A result naming a tool already present in `context.tools` (design spec
    section 7's `added_tool_names`, consumed here per Layer 06's own
    boundary -- see `spec/tools.md`) is a no-op, not a duplicate schema."""
    from minion_agent.tools.result import ToolResult

    loop, adapter = _loop_with_adapter(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="second"),), StopReason.STOP),
    )

    def redeclare_self(tool_call_id: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_call_id="",
            content=(TextBlock(text="pong"),),
            tool_name="echo",
            added_tool_names=("echo",),
        )

    _register(loop, "echo", redeclare_self)
    await loop.prompt(_say("go"))

    second_request_names = [schema.name for schema in adapter.requests[1].tools]
    assert second_request_names.count("echo") == 1


async def test_should_stop_and_prepare_next_turn_receive_pis_full_context() -> None:
    """`L08-R001`: listener signature mirrors pinned pi's
    `ShouldStopAfterTurnContext`/`PrepareNextTurnContext` exactly --
    `message`, `tool_results`, `context`, `new_messages`."""
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    seen: dict[str, Any] = {}

    def observe_stop(
        instance: Any, message: Any, tool_results: Any, context: Any, new_messages: Any
    ) -> None:
        seen["stop_message"] = message
        seen["stop_tool_results"] = tool_results
        seen["stop_context"] = context
        seen["stop_new_messages"] = new_messages
        return None

    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, observe_stop)
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert isinstance(seen["stop_message"], AssistantMessage)
    assert seen["stop_tool_results"] == ()
    assert isinstance(seen["stop_context"], RunContext)
    assert [text_of(m) for m in seen["stop_new_messages"]] == ["hello", "hi"]


# -- L08-R002: pinned pi's real handleRunFailure catch boundary --------------


async def test_a_pre_step_listener_failure_settles_gracefully() -> None:
    async def boom(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise RuntimeError("listener exploded")

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, boom)

    await loop.prompt(_say("hello"))  # must not raise

    assert loop.instance.status is AgentStatus.IDLE
    assert loop.instance.error_message == "listener exploded"
    kinds = [e.kind for e in loop.instance.log.events]
    # `TURN_START` legitimately precedes the failure here: pinned pi's own
    # `runAgentLoop` emits it unconditionally before ever calling into
    # `runLoop`'s body, where a pre-step listener would fire -- the same
    # `TURN_START` this run's own opening already appended (`L08-R006`), not
    # a synthetic one `_settle_run_failure` itself inserts (it inserts none).
    assert kinds == [
        EventKind.AGENT_START,
        EventKind.TURN_START,
        EventKind.ASSISTANT_MESSAGE,
        EventKind.TURN_END,
        EventKind.AGENT_END,
    ]
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "failed"
    assistant = [m for m in derive_messages(loop.instance.log) if isinstance(m, AssistantMessage)]
    assert assistant and assistant[-1].stop_reason is StopReason.ERROR
    assert assistant[-1].error_message == "listener exploded"


async def test_failure_agent_end_messages_is_only_the_failure_message() -> None:
    """`L08-R002`: pinned pi's `agent_end(messages=[failureMessage])`, not
    every message accumulated since `AGENT_START`."""

    async def boom(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise RuntimeError("boom")

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, boom)

    await loop.prompt(_say("hello"))

    from minion_agent.agent.projection import project

    events = project(loop.instance.log)
    end = next(e for e in events if isinstance(e, AgentEnd))
    assert len(end.messages) == 1
    assert end.messages[0].error_message == "boom"


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
    async def boom(instance: Any, *args: Any) -> RunConfigUpdate:
        raise RuntimeError("prepare listener exploded")

    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, boom)

    await loop.prompt(_say("hello"))  # must not raise

    assert loop.instance.error_message == "prepare listener exploded"
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "failed"


async def test_a_post_turn_callback_failure_settles_gracefully() -> None:
    """Post-turn callback failure, distinct from a pre-step failure -- both
    are still "the run executor," pinned pi's own catch boundary."""

    async def boom_after_turn(instance: Any, *args: Any) -> bool:
        raise RuntimeError("post-turn callback exploded")

    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, boom_after_turn)

    await loop.prompt(_say("hello"))

    assert loop.instance.error_message == "post-turn callback exploded"
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "failed"


async def test_an_unresolvable_model_still_raises_uncaught() -> None:
    """`L08-R002`: the failure-settling fallback catches the run executor
    broadly, but explicitly excludes the eager, pre-stream `UnknownModelError`
    -- an unresolvable model is a caller/config bug, reported immediately,
    never smuggled into a settled failure turn
    (`eager-invalid-model-fails-before-stream.yaml`)."""
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.model = ModelId("mock", "no-such-model")

    with pytest.raises(UnknownModelError):
        await loop.prompt(_say("hello"))


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


# -- L08-R002, PASS 4: recovery through the live, listener-bearing seam ------


async def test_a_run_executor_failure_recovers_through_the_live_lifecycle_event_seam() -> None:
    """`L08-R002`, PASS 4: recovery now goes through the SAME live,
    listener-bearing `AGENT_LIFECYCLE_EVENT` seam ordinary progress uses --
    not raw log appends a listener has no way to observe -- proven by a
    passive (non-throwing) listener actually receiving the failure's own
    `message_start`/`message_end`/`turn_end`/`agent_end`, live, during the
    run, not merely reconstructible afterward via offline `project()`."""

    async def boom(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise RuntimeError("run executor failed")

    seen: list[type] = []

    def observe(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart | MessageEnd | TurnEnd | AgentEnd):
            seen.append(type(event))

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, boom)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe)

    await loop.prompt(_say("hello"))  # must not raise -- no listener interrupted it

    assert seen == [MessageStart, MessageEnd, TurnEnd, AgentEnd]


async def test_a_post_turn_callback_failure_recovers_through_the_live_seam() -> None:
    """`L08-R002`, PASS 4/5: same live seam for a failure originating from a
    POST-turn callback (`AGENT_TURN_STOPPING`), distinct from a pre-step
    failure -- both are "the run executor" from pinned pi's own perspective.
    The ordinary turn's own admitted prompt, its own streamed assistant
    reply, and its own `turn_end` already happened live before the failure;
    the same seam then delivers the synthesized failure's own sequence too."""

    def boom(*args: Any) -> TurnStopping:
        raise RuntimeError("stopping listener exploded")

    seen: list[type] = []

    def observe(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart | MessageEnd | TurnEnd | AgentEnd):
            seen.append(type(event))

    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, boom)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe)

    await loop.prompt(_say("hello"))  # must not raise

    assert seen == [
        MessageStart,  # the prompt's own admission
        MessageEnd,
        MessageStart,  # the assistant's own streamed reply (`L08-R002`, PASS 5)
        MessageEnd,
        TurnEnd,  # the ordinary turn's own close
        MessageStart,  # the synthesized failure
        MessageEnd,
        TurnEnd,
        AgentEnd,
    ]


async def test_failure_message_start_listener_failure_interrupts_recovery() -> None:
    """`L08-R002`, PASS 4: a listener throwing during the failure's own
    `message_start` aborts the rest of pinned pi's recovery sequence and
    propagates uncaught, exactly like pinned pi's own bare sequential `await
    this.processEvents(...)` chain -- `_run_wrapped`'s own `finally` (pinned
    pi's `finishRun`) still settles status regardless."""

    async def boom_pre_step(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise RuntimeError("run executor failed")

    seen: list[type] = []

    def boom_on_message_start(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart | MessageEnd | TurnEnd | AgentEnd):
            seen.append(type(event))
        if isinstance(event, MessageStart):
            raise RuntimeError("message_start listener exploded")

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, boom_pre_step)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, boom_on_message_start)

    with pytest.raises(RuntimeError, match="message_start listener exploded"):
        await loop.prompt(_say("hello"))

    assert loop.instance.status is AgentStatus.IDLE
    assert seen == [MessageStart]
    kinds = [e.kind for e in loop.instance.log.events]
    # message_start's own reduce is only `streaming_message = failure`
    # (`PASS 5`, `L08-R002`) -- pinned pi's own reducer does not push onto
    # the transcript until `message_end`, so the failure's own
    # `ASSISTANT_MESSAGE` log entry is NOT yet appended when message_start's
    # listener throws; recovery aborted before reaching that reduce at all.
    assert kinds == [EventKind.AGENT_START, EventKind.TURN_START]
    assert loop.instance.streaming_message is None


async def test_failure_message_end_listener_failure_interrupts_recovery() -> None:
    """`L08-R002`, PASS 4: `message_start`'s own listener succeeds; the
    interruption happens precisely at `message_end`."""

    async def boom_pre_step(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise RuntimeError("run executor failed")

    seen: list[type] = []

    def boom_on_message_end(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart | MessageEnd | TurnEnd | AgentEnd):
            seen.append(type(event))
        if isinstance(event, MessageEnd):
            raise RuntimeError("message_end listener exploded")

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, boom_pre_step)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, boom_on_message_end)

    with pytest.raises(RuntimeError, match="message_end listener exploded"):
        await loop.prompt(_say("hello"))

    assert loop.instance.status is AgentStatus.IDLE
    assert seen == [MessageStart, MessageEnd]
    kinds = [e.kind for e in loop.instance.log.events]
    assert kinds == [EventKind.AGENT_START, EventKind.TURN_START, EventKind.ASSISTANT_MESSAGE]


async def test_failure_turn_end_listener_failure_interrupts_recovery() -> None:
    """`L08-R002`, PASS 4: `message_start`/`message_end` both succeed; the
    interruption happens precisely at the failure's own `turn_end`."""

    async def boom_pre_step(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise RuntimeError("run executor failed")

    seen: list[type] = []

    def boom_on_turn_end(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart | MessageEnd | TurnEnd | AgentEnd):
            seen.append(type(event))
        if isinstance(event, TurnEnd):
            raise RuntimeError("turn_end listener exploded")

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, boom_pre_step)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, boom_on_turn_end)

    with pytest.raises(RuntimeError, match="turn_end listener exploded"):
        await loop.prompt(_say("hello"))

    assert loop.instance.status is AgentStatus.IDLE
    assert seen == [MessageStart, MessageEnd, TurnEnd]
    kinds = [e.kind for e in loop.instance.log.events]
    # turn_end's own durable "reduce" (the TURN_END log entry, carrying the
    # failure-message override) already happened -- but no AGENT_END does.
    assert kinds == [
        EventKind.AGENT_START,
        EventKind.TURN_START,
        EventKind.ASSISTANT_MESSAGE,
        EventKind.TURN_END,
    ]


async def test_failure_agent_end_listener_failure_interrupts_recovery() -> None:
    """`L08-R002`, PASS 4: the entire failure sequence's own listeners
    succeed up through `turn_end`; the interruption happens precisely at
    `agent_end` -- pinned pi's own final recovery event."""

    async def boom_pre_step(instance: Any, reason: Any, messages: Any, next_: Any) -> Enter:
        raise RuntimeError("run executor failed")

    seen: list[type] = []

    def boom_on_agent_end(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart | MessageEnd | TurnEnd | AgentEnd):
            seen.append(type(event))
        if isinstance(event, AgentEnd):
            raise RuntimeError("agent_end listener exploded")

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, boom_pre_step)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, boom_on_agent_end)

    with pytest.raises(RuntimeError, match="agent_end listener exploded"):
        await loop.prompt(_say("hello"))

    assert loop.instance.status is AgentStatus.IDLE
    assert seen == [MessageStart, MessageEnd, TurnEnd, AgentEnd]
    kinds = [e.kind for e in loop.instance.log.events]
    # agent_end's own durable "reduce" (the AGENT_END log entry, carrying the
    # failure-only messages override) already happened before its own
    # listener threw.
    assert kinds == [
        EventKind.AGENT_START,
        EventKind.TURN_START,
        EventKind.ASSISTANT_MESSAGE,
        EventKind.TURN_END,
        EventKind.AGENT_END,
    ]
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "failed"


async def test_a_rejection_during_the_initial_steering_admission_ends_the_turn() -> None:
    """The STEERING-reason pre-step dispatch for the initial claim
    (`L08-R006`'s own second admission stage) can reject independently of the
    prompt's own INITIAL-reason decision -- the prompt's own message was
    already admitted before the rejected steering stage; rejection ends the
    run, it does not retroactively undo what already happened."""

    async def veto_steering_only(instance: Any, reason: Any, messages: Any, next_: Any) -> Any:
        if reason.value == "steering":
            return Reject(reason="no steering allowed")
        return await next_(instance, reason, messages)

    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_PRE_STEP, veto_steering_only)
    loop.instance.inbox.steer(_say("steer me"))

    await loop.prompt(_say("hello"))

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "rejected"
    assert end.data["detail"] == "no steering allowed"
    texts = [text_of(m) for m in derive_messages(loop.instance.log)]
    assert texts == ["hello"]


class _NoStartStreamAdapter:
    """Test-only adapter that never emits its own `StreamStart` chunk at all
    -- some real providers do not -- exercising pinned pi's own defensive
    `!addedPartial` fallback (`L08-R002`, PASS 5): `_run_step` must still
    dispatch a `MessageStart` for the assistant's own reply immediately
    before its `MessageEnd`, even though no live start chunk ever opened it."""

    provider = "mock"
    api = "mock"
    models = frozenset({"mock-1"})

    def __init__(self, message: AssistantMessage) -> None:
        self._message = message

    def stream(self, request: Request) -> AsyncIterator[StreamChunk]:
        async def replay() -> AsyncIterator[StreamChunk]:
            yield StreamDone(message=self._message, partial=self._message)

        return replay()


async def test_a_stream_with_no_start_chunk_still_gets_a_message_start() -> None:
    from minion_agent.agent.identity import AgentDefinition
    from minion_agent.agent.registry import AgentRegistry
    from minion_agent.agent_loop.driver import AgentLoop
    from minion_agent.llm import LlmService
    from minion_agent.runtime import Context
    from minion_agent.session import SessionService
    from minion_agent.tools.registry import ToolRegistry

    message = AssistantMessage(
        content=(TextBlock(text="hi"),),
        stop_reason=StopReason.STOP,
        usage=Usage(),
        model="mock-1",
        provider="mock",
        timestamp=0,
    )
    ctx = Context()
    sessions = SessionService()
    llm = LlmService()
    llm.register(_NoStartStreamAdapter(message))
    registry = AgentRegistry(ctx=ctx, sessions=sessions)
    handle = registry.create("room-a", AgentDefinition(name="ada", model=ModelId("mock", "mock-1")))
    loop = AgentLoop(
        instance=handle.instance, llm=llm, tools=ToolRegistry(), artifacts=sessions.artifacts
    )
    seen: list[type] = []
    streaming_at_message_start: list[Any] = []

    def observe(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart | MessageEnd):
            seen.append(type(event))
        if isinstance(event, MessageStart):
            streaming_at_message_start.append(instance.streaming_message)

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe)

    await loop.prompt(_say("go"))

    assert seen == [MessageStart, MessageEnd, MessageStart, MessageEnd]
    # `L08-R002`, PASS 6: the fallback `MessageStart` (the second one -- the
    # first is the prompt's own admission) reduces (`streaming_message =
    # reply`) BEFORE its own dispatch -- an earlier revision cleared
    # `streaming_message` and appended the transcript entry (message_end's
    # own reduce) before this fallback `MessageStart`'s own dispatch, so a
    # listener observed `None` here instead of the reply.
    assert streaming_at_message_start[1] is message


async def test_turn_end_sets_error_message_even_for_a_non_terminal_reply() -> None:
    """`L08-R002`, PASS 5: `turn_end`'s own reduce (pinned pi: `if role ===
    "assistant" && errorMessage`) applies unconditionally to every turn_end,
    not only a represented `error`/`aborted` terminal one -- an ordinary
    `stop`/`tool_use` reply can still carry a truthy `error_message` (a
    provider-level incidental failure that did not itself prevent
    completion), and `error_message` must still be set from it."""
    loop = _loop(
        ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP, error_message="incidental")
    )

    await loop.prompt(_say("go"))

    assert loop.instance.error_message == "incidental"


# -- L08-R003: full streaming_message fidelity --------------------------------


class _RichStreamAdapter:
    """Test-only adapter emitting `start`/`*_start`/`*_delta`/`*_end` chunks
    for text, thinking, AND tool-call content -- the certified Layer-02/04
    `MockAdapter` only emits text deltas, so this exercises `L08-R003`'s
    full-partial-fidelity requirement (thinking/tool-call, not just text)
    without touching certified adapter code."""

    provider = "mock"
    api = "mock"
    models = frozenset({"mock-1"})

    def __init__(self, message: AssistantMessage) -> None:
        self._message = message
        self.requests: list[Request] = []

    def stream(self, request: Request) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        pending = replace(self._message, stop_reason=StopReason.PENDING)

        async def replay() -> AsyncIterator[StreamChunk]:
            yield StreamStart(partial=pending)
            for index, block in enumerate(self._message.content):
                if isinstance(block, ThinkingBlock):
                    yield ThinkingStart(content_index=index, partial=pending)
                    yield ThinkingDelta(content_index=index, delta=block.thinking, partial=pending)
                    yield ThinkingEnd(content_index=index, thinking=block.thinking, partial=pending)
                elif isinstance(block, ToolCallBlock):
                    yield ToolCallStart(content_index=index, partial=pending)
                    yield ToolCallDelta(content_index=index, delta="{}", partial=pending)
                    yield ToolCallEnd(content_index=index, tool_call=block, partial=pending)
                elif isinstance(block, TextBlock):
                    yield TextStart(content_index=index, partial=pending)
                    yield TextDelta(content_index=index, delta=block.text, partial=pending)
                    yield TextEnd(content_index=index, text=block.text, partial=pending)
            yield StreamDone(message=self._message, partial=self._message)

        return replay()


def _rich_loop(message: AssistantMessage) -> tuple[Any, _RichStreamAdapter]:
    from minion_agent.agent.identity import AgentDefinition
    from minion_agent.agent.registry import AgentRegistry
    from minion_agent.agent_loop.driver import AgentLoop
    from minion_agent.llm import LlmService
    from minion_agent.runtime import Context
    from minion_agent.session import SessionService
    from minion_agent.tools.registry import ToolRegistry

    ctx = Context()
    sessions = SessionService()
    llm = LlmService()
    adapter = _RichStreamAdapter(message)
    llm.register(adapter)
    registry = AgentRegistry(ctx=ctx, sessions=sessions)
    handle = registry.create(
        "room-a", AgentDefinition(name="ada", model=ModelId("mock", "mock-1"), system="be helpful")
    )
    loop = AgentLoop(
        instance=handle.instance, llm=llm, tools=ToolRegistry(), artifacts=sessions.artifacts
    )
    return loop, adapter


async def test_streaming_message_carries_the_full_partial_for_text() -> None:
    from minion_agent.llm import Usage

    message = AssistantMessage(
        content=(TextBlock(text="hello"),),
        stop_reason=StopReason.STOP,
        usage=Usage(),
        model="mock-1",
        provider="mock",
        timestamp=1,
    )
    loop, _adapter = _rich_loop(message)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    from minion_agent.agent.projection import MessageUpdate, project

    updates = [e for e in project(loop.instance.log) if isinstance(e, MessageUpdate)]
    assert any(isinstance(u.event, TextStart) for u in updates)
    assert any(isinstance(u.event, TextDelta) and text_of(u.message) == "hello" for u in updates)
    assert any(isinstance(u.event, TextEnd) for u in updates)


async def test_streaming_message_carries_the_full_partial_for_thinking() -> None:
    from minion_agent.llm import Usage

    message = AssistantMessage(
        content=(ThinkingBlock(thinking="pondering"), TextBlock(text="done")),
        stop_reason=StopReason.STOP,
        usage=Usage(),
        model="mock-1",
        provider="mock",
        timestamp=1,
    )
    loop, _adapter = _rich_loop(message)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    from minion_agent.agent.projection import MessageUpdate, project

    updates = [e for e in project(loop.instance.log) if isinstance(e, MessageUpdate)]
    thinking_updates = [
        u for u in updates if isinstance(u.event, ThinkingStart | ThinkingDelta | ThinkingEnd)
    ]
    assert thinking_updates
    assert any(
        any(isinstance(b, ThinkingBlock) and b.thinking == "pondering" for b in u.message.content)
        for u in thinking_updates
    )


async def test_streaming_message_carries_the_full_partial_for_tool_calls() -> None:
    from minion_agent.llm import Usage

    call = ToolCallBlock(id="t1", name="echo", arguments={})
    message = AssistantMessage(
        content=(call,),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(),
        model="mock-1",
        provider="mock",
        timestamp=1,
    )
    loop, _adapter = _rich_loop(message)
    _register(loop, "echo", lambda tool_call_id, args: "pong")
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, lambda *_: TurnStopping.STOP)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    from minion_agent.agent.projection import MessageUpdate, project

    updates = [e for e in project(loop.instance.log) if isinstance(e, MessageUpdate)]
    toolcall_updates = [
        u for u in updates if isinstance(u.event, ToolCallStart | ToolCallDelta | ToolCallEnd)
    ]
    assert toolcall_updates
    assert any(
        any(isinstance(b, ToolCallBlock) and b.name == "echo" for b in u.message.content)
        for u in toolcall_updates
    )


async def test_streaming_message_clears_after_the_message_finalizes() -> None:
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert loop.instance.streaming_message is None


async def test_message_start_precedes_message_update_for_a_streamed_reply() -> None:
    """`L08-R003`: pinned pi's exact order, `message_start -> message_update*
    -> message_end` -- an earlier revision's canonical scenarios encoded the
    reverse."""
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    from minion_agent.agent.projection import MessageUpdate, project

    kinds = [type(e) for e in project(loop.instance.log)]
    # The entering user message's own MessageStart/MessageEnd pair is fully
    # matched before the assistant reply's stream ever opens, so a
    # MessageStart outnumbering MessageEnd at the moment of the first
    # MessageUpdate can only be the assistant reply's own still-open start --
    # proving it precedes every update, without assuming fixed indices.
    first_update = kinds.index(MessageUpdate)
    starts_before = kinds[:first_update].count(MessageStart)
    ends_before = kinds[:first_update].count(MessageEnd)
    assert starts_before > ends_before


# -- L08-R006: initial steering polled after the first turn opens ------------


async def test_the_initial_steering_claim_happens_after_turn_start() -> None:
    """Offline projection alone cannot prove this (it does not project inbox
    claims); a listener observing live log state at the moment the claimed
    steering message reaches it is the discriminating evidence pinned pi's
    own ordering requires (`L08-R006`)."""
    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    seen_turn_start_already_logged = []

    async def observe(instance: Any, reason: Any, messages: Any, next_: Any) -> Any:
        if reason.value == "initial":
            kinds = [e.kind for e in instance.log.events]
            seen_turn_start_already_logged.append(EventKind.TURN_START in kinds)
        return await next_(instance, reason, messages)

    loop.instance.ctx.events.on(AGENT_PRE_STEP, observe)
    loop.instance.inbox.steer(_say("steer me"))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert seen_turn_start_already_logged == [True]


async def test_the_initial_prompt_lifecycle_precedes_the_steering_claim() -> None:
    """`L08-R006`, PASS 4: `TURN_START` already being logged (proven above)
    is the WEAKER condition the PASS-3 review rejected -- pinned pi's own
    order requires the initial prompt's own COMPLETE message lifecycle
    (`message_start` then `message_end`) to be observable, live, before the
    steering queue is ever claimed at all. This listener-driven trace proves
    exactly that ordering, then confirms both messages still reach the SAME
    single first provider request -- their own lifecycle/claim timing
    differs, but pinned pi never receives them as separate turns."""
    loop, adapter = _loop_with_adapter(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    trace: list[str] = []

    def observe_lifecycle(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart):
            trace.append(f"message_start:{text_of(event.message)}")
        elif isinstance(event, MessageEnd):
            trace.append(f"message_end:{text_of(event.message)}")

    async def observe_claim(instance: Any, reason: Any, messages: Any, next_: Any) -> Any:
        # The exact Minion-only claim marker (this pre-step dispatch's own
        # STEERING reason) need not become pi canonical vocabulary -- what
        # matters is that queue mutation (`_claim_step_input()`, which
        # already ran by the time this STEERING-reason dispatch fires) is
        # proven to occur strictly after the prompt's own lifecycle above.
        if reason.value == "steering":
            trace.append("steering_claim")
        return await next_(instance, reason, messages)

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe_lifecycle)
    loop.instance.ctx.events.on(AGENT_PRE_STEP, observe_claim)
    loop.instance.inbox.steer(_say("steer me"))

    await loop.prompt(_say("go"))

    assert trace == [
        "message_start:go",
        "message_end:go",
        "steering_claim",
        "message_start:steer me",
        "message_end:steer me",
        "message_start:hi",  # the assistant's own streamed reply (`L08-R002`, PASS 5)
        "message_end:hi",
    ]
    # Prompt and steering remain inputs to the SAME first provider request --
    # only their own lifecycle/claim timing differs, never their turn.
    assert [text_of(m) for m in adapter.requests[0].messages] == ["go", "steer me"]


# -- L08-R008: represented error/aborted is immediately terminal -------------


async def test_represented_error_skips_prepare_stop_steering_and_follow_up() -> None:
    loop = _loop(ScriptedResponse((), StopReason.ERROR, error_message="boom"))
    prepared: list[str] = []
    stopped: list[str] = []

    async def observe_prepare(instance: Any, *args: Any, **kwargs: Any) -> RunConfigUpdate:
        prepared.append("called")
        return RunConfigUpdate()

    def observe_stop(*args: Any) -> TurnStopping:
        stopped.append("called")
        return TurnStopping.CONTINUE

    async def queue_steering_at_this_turns_own_turn_end(instance: Any, event: Any) -> None:
        # `L08-R006` moved the initial steering claim to admit BEFORE this
        # turn ever runs at all, so queuing during the INITIAL pre-step no
        # longer proves anything about R008 (that claim happens unconditionally
        # either way). Queuing here instead -- inside this turn's own live
        # `TurnEnd` dispatch, which fires immediately before `_run_step`
        # returns its terminal result to `_run_inner` -- guarantees the
        # message is queued strictly after any admission this turn could
        # possibly have done, so it is only ever at risk of the steady-state
        # poll R008 must prove never runs for a represented error/aborted turn.
        if isinstance(event, TurnEnd):
            instance.inbox.steer(_say("should not be consumed"))

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, queue_steering_at_this_turns_own_turn_end)
    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, observe_prepare)
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, observe_stop)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    assert prepared == []
    assert stopped == []
    assert loop.instance.inbox.has_pending()  # steering was never claimed
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "error"


async def test_represented_aborted_skips_prepare_stop_steering_and_follow_up() -> None:
    loop = _loop(ScriptedResponse((), StopReason.ABORTED, error_message="cancelled"))
    prepared: list[str] = []
    stopped: list[str] = []

    async def observe_prepare(instance: Any, *args: Any, **kwargs: Any) -> RunConfigUpdate:
        prepared.append("called")
        return RunConfigUpdate()

    def observe_stop(*args: Any) -> TurnStopping:
        stopped.append("called")
        return TurnStopping.CONTINUE

    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, observe_prepare)
    loop.instance.ctx.events.on(AGENT_TURN_STOPPING, observe_stop)
    loop.instance.inbox.followup(_say("go"))
    loop.instance.inbox.followup(_say("queued follow-up, should not be consumed"))

    await loop.run_until_idle()

    assert prepared == []
    assert stopped == []
    end = next(e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END)
    assert end.data["reason"] == "aborted"
    # The queued follow-up survives this run untouched -- run_until_idle's own
    # outer pump picks it up as a fresh, separate run afterward.
    texts = [text_of(m) for m in derive_messages(loop.instance.log)]
    assert texts.count("queued follow-up, should not be consumed") == 1


async def test_represented_error_agent_end_messages_uses_the_normal_accumulator() -> None:
    """Distinct from `handleRunFailure` (`L08-R002`): a represented error is a
    NORMAL, successfully-produced message, so `agent_end.messages` is pinned
    pi's usual run-scoped accumulator (`newMessages`), not a synthesized
    single-message override."""
    loop = _loop(ScriptedResponse((), StopReason.ERROR, error_message="boom"))
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    from minion_agent.agent.projection import project

    end = next(e for e in project(loop.instance.log) if isinstance(e, AgentEnd))
    assert [text_of(m) for m in end.messages] == ["go", ""]


async def test_an_agent_start_listener_failure_settles_gracefully() -> None:
    """`L08-R002`, PASS 6: `agent_start`'s own dispatch now lives INSIDE
    `_execute_run`'s own `try`, sharing `_run_inner`'s exception boundary --
    an earlier revision dispatched it BEFORE that `try` even opened, so a
    listener failure here escaped straight past `_run_wrapped`'s own
    `finally`, uncaught, instead of being settled like any other
    run-executor failure."""

    def boom_on_agent_start(instance: Any, event: Any) -> None:
        if isinstance(event, AgentStart):
            raise RuntimeError("agent_start listener exploded")

    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, boom_on_agent_start)

    await loop.prompt(_say("hello"))  # must not raise -- settled gracefully

    assert loop.instance.status is AgentStatus.IDLE
    ends = [e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END]
    assert [e.data["reason"] for e in ends] == ["failed"]


async def test_a_successful_agent_end_listener_failure_settles_gracefully() -> None:
    """`L08-R002`, PASS 6: a listener failing on an ORDINARY, successful
    `agent_end` -- not a recovery one -- is caught the same way, producing a
    SECOND, `failed` `AGENT_END` right after the first, successful one:
    matching pinned Pi's own architecture exactly, `handleRunFailure` has no
    awareness of how far the run had already reduced when its own dispatch
    throws. An earlier revision dispatched the success path's own `agent_end`
    entirely outside `_execute_run`'s `try`, so this failure escaped uncaught
    instead."""

    def boom_on_agent_end(instance: Any, event: Any) -> None:
        if isinstance(event, AgentEnd) and event.reason != "failed":
            raise RuntimeError("agent_end listener exploded")

    loop = _loop(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, boom_on_agent_end)

    await loop.prompt(_say("hello"))  # must not raise -- settled gracefully

    assert loop.instance.status is AgentStatus.IDLE
    ends = [e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END]
    assert [e.data["reason"] for e in ends] == ["completed", "failed"]


async def test_tool_execution_start_listener_failure_prevents_that_calls_own_execution() -> None:
    """`L08-R002`, PASS 6: `ToolExecutionStart`'s own dispatch is now
    genuinely LIVE (Layer-06's additive `on_execution_start` hook, awaited at
    the exact point `tools/execution-start` fires -- before resolution,
    preparation, validation, and the before-hook) -- a listener that throws
    here now genuinely PREVENTS that call's own `execute()` from ever
    running. PASS 5's own capture-and-replay design could not do this: the
    tool's own side effects had already happened by the time a listener
    could object."""

    ran = False

    def echo(call_id: str, args: dict[str, Any]) -> str:
        nonlocal ran
        ran = True
        return "pong"

    def boom_on_start(instance: Any, event: Any) -> None:
        if isinstance(event, ToolExecutionStart):
            raise RuntimeError("tool_execution_start listener exploded")

    call = ToolCallBlock(id="t1", name="echo", arguments={})
    loop = _loop(ScriptedResponse((call,), StopReason.TOOL_USE))
    _register(loop, "echo", echo)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, boom_on_start)

    await loop.prompt(_say("hello"))  # must not raise -- settled gracefully

    assert not ran
    assert loop.instance.status is AgentStatus.IDLE
    ends = [e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END]
    assert [e.data["reason"] for e in ends] == ["failed"]


async def test_sequential_tool_batch_delivers_live_start_end_per_call_in_order() -> None:
    """`L08-R002`, PASS 6: sequential-mode ordering is now the REAL
    `start A, end A, start B, end B` -- PASS 5's own capture-and-replay
    design reordered this to `start A, start B, end A, end B` (batch-wide
    capture, redelivered only after the whole batch settled)."""

    trace: list[str] = []

    def make_tool(name: str) -> Any:
        def fn(call_id: str, args: dict[str, Any]) -> str:
            trace.append(f"execute:{name}")
            return name

        return fn

    def observe(instance: Any, event: Any) -> None:
        if isinstance(event, ToolExecutionStart):
            trace.append(f"start:{event.tool_name}")
        elif isinstance(event, ToolExecutionEnd):
            trace.append(f"end:{event.tool_name}")

    calls = (
        ToolCallBlock(id="t1", name="alpha", arguments={}),
        ToolCallBlock(id="t2", name="beta", arguments={}),
    )
    loop = _loop(
        ScriptedResponse(calls, StopReason.TOOL_USE),
        ScriptedResponse((), StopReason.STOP),
    )
    _register(loop, "alpha", make_tool("alpha"), mode=ExecutionMode.SEQUENTIAL)
    _register(loop, "beta", make_tool("beta"), mode=ExecutionMode.SEQUENTIAL)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe)

    await loop.prompt(_say("hello"))

    assert trace == [
        "start:alpha",
        "execute:alpha",
        "end:alpha",
        "start:beta",
        "execute:beta",
        "end:beta",
    ]


async def test_tool_execution_update_reaches_the_lifecycle_seam() -> None:
    """`L08-R002`, PASS 6: a tool's own live partial-output report
    (`update()`, called SYNCHRONOUSLY by `execute()` per the established
    3-argument calling convention `_wants_update` checks for) now reaches
    `AGENT_LIFECYCLE_EVENT` as a `ToolExecutionUpdate` -- previously missing
    from the union entirely -- delivered before that SAME call's own
    `ToolExecutionEnd` (captured via the existing, certified, synchronous
    `tools/update` EMIT listener and redelivered per call, immediately
    before that call's own live `ToolExecutionEnd` dispatch)."""

    def slow_echo(call_id: str, args: dict[str, Any], update: Any) -> str:
        update("working")
        return "done"

    seen: list[Any] = []

    def observe(instance: Any, event: Any) -> None:
        if isinstance(event, ToolExecutionUpdate | ToolExecutionEnd):
            seen.append(event)

    call = ToolCallBlock(id="t1", name="slow", arguments={"x": 1})
    loop = _loop(
        ScriptedResponse((call,), StopReason.TOOL_USE),
        ScriptedResponse((), StopReason.STOP),
    )
    _register(loop, "slow", slow_echo)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe)

    await loop.prompt(_say("hello"))

    assert [type(e) for e in seen] == [ToolExecutionUpdate, ToolExecutionEnd]
    update_event = seen[0]
    assert (update_event.tool_call_id, update_event.tool_name) == ("t1", "slow")
    assert update_event.arguments == {"x": 1}
    assert update_event.partial_result == "working"


async def test_pending_tool_calls_still_shows_the_call_during_its_own_update_dispatch() -> None:
    """`L08-R002`, PASS 7: a `tool_execution_update` listener observes the call still marked
    pending -- `on_execution_end` (which clears `pending_tool_calls`) only fires once every one of
    that call's own scheduled update dispatches has been joined (`tools/execute.py`,
    `OnExecutionUpdate`), matching pinned Pi's own `tool_execution_update` firing strictly before
    `tool_execution_end`. An earlier revision cleared `pending_tool_calls` for a call BEFORE
    redelivering its own captured updates, so a listener observed it already gone."""

    def slow_echo(call_id: str, args: dict[str, Any], update: Any) -> str:
        update("working")
        return "done"

    pending_snapshots: list[frozenset[str]] = []

    def observe(instance: Any, event: Any) -> None:
        if isinstance(event, ToolExecutionUpdate):
            pending_snapshots.append(frozenset(instance.pending_tool_calls))

    call = ToolCallBlock(id="t1", name="slow", arguments={})
    loop = _loop(
        ScriptedResponse((call,), StopReason.TOOL_USE),
        ScriptedResponse((), StopReason.STOP),
    )
    _register(loop, "slow", slow_echo)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe)

    await loop.prompt(_say("hello"))

    assert pending_snapshots == [frozenset({"t1"})]


async def test_two_lifecycle_listeners_each_suspend_before_the_tool_continues() -> None:
    """`L08-R002`, PASS 9 (contract convergence): pinned Pi's own serial listener loop
    (`for (const listener of listeners) { await listener(event, signal); }`, `agent.ts:544-591`)
    suspends after EVERY listener, even a fully synchronous one, because JS's `await` always defers
    its continuation by at least one microtask turn. With two `AGENT_LIFECYCLE_EVENT` listeners
    registered, a tool calling `update()` must observe listener 1's own effect, then resume its own
    synchronous work, BEFORE listener 2 ever runs -- `[listener-1, tool-continued, listener-2]`, not
    `[listener-1, listener-2, tool-continued]` (the PASS-8 candidate's own observed order, rejected
    by the independent Rust re-review's own focused two-listener probe)."""

    def slow_echo(call_id: str, args: dict[str, Any], update: Any) -> str:
        update("working")
        order.append("tool-continued")
        return "done"

    order: list[str] = []

    def listener_a(instance: Any, event: Any) -> None:
        if isinstance(event, ToolExecutionUpdate):
            order.append("listener-1")

    def listener_b(instance: Any, event: Any) -> None:
        if isinstance(event, ToolExecutionUpdate):
            order.append("listener-2")

    call = ToolCallBlock(id="t1", name="slow", arguments={})
    loop = _loop(
        ScriptedResponse((call,), StopReason.TOOL_USE),
        ScriptedResponse((), StopReason.STOP),
    )
    _register(loop, "slow", slow_echo)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, listener_a)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, listener_b)

    await loop.prompt(_say("hello"))

    assert order == ["listener-1", "tool-continued", "listener-2"]


async def test_tool_execution_update_listener_failure_is_a_genuine_run_failure() -> None:
    """`L08-R002`, PASS 7: pinned Pi's own `tool_execution_update` dispatch
    (`agent-loop.ts:670-711`) lets a listener's own rejection propagate straight out of
    `executePreparedToolCall`, uncaught -- the SAME causal category as a
    `tool_execution_start`/`tool_execution_end` listener failure, not silently absorbed into a
    per-call tool error result. `_finalize`/`tool_execution_end` never run for that call at all."""

    def slow_echo(call_id: str, args: dict[str, Any], update: Any) -> str:
        update("working")
        return "done"

    finalized = False

    def boom_on_update(instance: Any, event: Any) -> None:
        nonlocal finalized
        if isinstance(event, ToolExecutionUpdate):
            raise RuntimeError("update listener exploded")
        if isinstance(event, ToolExecutionEnd):
            finalized = True

    call = ToolCallBlock(id="t1", name="slow", arguments={})
    loop = _loop(ScriptedResponse((call,), StopReason.TOOL_USE))
    _register(loop, "slow", slow_echo)
    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, boom_on_update)

    await loop.prompt(_say("hello"))  # must not raise -- settled gracefully via recovery

    assert not finalized
    assert loop.instance.status is AgentStatus.IDLE
    ends = [e for e in loop.instance.log.events if e.kind == EventKind.AGENT_END]
    assert [e.data["reason"] for e in ends] == ["failed"]


async def test_tool_execution_end_carries_the_finalized_result() -> None:
    """`L08-R002`, PASS 6: `ToolExecutionEnd` now carries `tool_name` and the
    finalized `result` itself (Layer 06's own `ToolResult`), not merely
    `is_error` derived from it -- an earlier revision exposed only
    `tool_call_id`/`is_error`, a real payload reduction pinned Pi does not
    have."""
    call = ToolCallBlock(id="t1", name="echo", arguments={})
    loop = _loop(
        ScriptedResponse((call,), StopReason.TOOL_USE),
        ScriptedResponse((), StopReason.STOP),
    )
    _register(loop, "echo", lambda call_id, args: "pong")

    seen: list[ToolExecutionEnd] = []

    def observe(instance: Any, event: Any) -> None:
        if isinstance(event, ToolExecutionEnd):
            seen.append(event)

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe)

    await loop.prompt(_say("hello"))

    assert len(seen) == 1
    end = seen[0]
    assert end.tool_name == "echo"
    assert end.is_error is False
    assert end.result.tool_call_id == "t1"
    assert end.result.tool_name == "echo"
    assert text_of(end.result.to_message()) == "pong"
