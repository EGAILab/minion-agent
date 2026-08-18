"""agent/pre-step decides what the model sees.

The rewrite case uses the *decision* pattern -- a listener that owns the
outcome returns without delegating. A lone listener cannot rewrite by
delegating with replacement arguments: the waterfall's terminal is a value
fixed at dispatch, so with nothing downstream to receive the replacement the
chain yields the terminal the driver supplied. A replacement reaches the
listeners *after* it, which the delegation test below pins.
"""

from typing import Any

from minion_agent.agent.decisions import Enter, PreStepReason, Reject
from minion_agent.agent.events import AGENT_PRE_STEP
from minion_agent.agent.instance import AgentInstance
from minion_agent.llm import TextBlock, ToolCallBlock, UserMessage, text_of
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind, derive_messages

from .test_single_turn import _loop, _loop_with_adapter


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


async def test_the_terminal_enters_the_claimed_messages() -> None:
    """No listener at all still runs the step with what was claimed."""
    loop = _loop(ScriptedResponse((TextBlock(text="ok"),), StopReason.STOP))
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert text_of(derive_messages(loop.instance.log)[0]) == "hello"


async def test_a_listener_may_rewrite_the_entering_messages() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))

    async def rewrite(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Enter:
        return Enter(messages=(_say("rewritten"),))

    loop.instance.ctx.events.on(AGENT_PRE_STEP, rewrite)
    loop.instance.inbox.followup(_say("original"))

    await loop.run_until_idle()

    assert text_of(derive_messages(loop.instance.log)[0]) == "rewritten"


async def test_a_replacement_reaches_the_listeners_after_it() -> None:
    """The transformation pattern: registration order is application order."""
    loop = _loop(ScriptedResponse((), StopReason.STOP))

    async def transform(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Any:
        return await next_(instance, reason, (_say("transformed"),))

    async def decide(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Enter:
        return Enter(messages=messages)

    loop.instance.ctx.events.on(AGENT_PRE_STEP, transform)
    loop.instance.ctx.events.on(AGENT_PRE_STEP, decide)
    loop.instance.inbox.followup(_say("original"))

    await loop.run_until_idle()

    assert text_of(derive_messages(loop.instance.log)[0]) == "transformed"


async def test_a_rejection_closes_the_turn_with_no_step() -> None:
    loop = _loop()

    async def veto(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Reject:
        return Reject(reason="not now")

    loop.instance.ctx.events.on(AGENT_PRE_STEP, veto)
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    kinds = [event.kind for event in loop.instance.log.events]
    assert EventKind.STEP_START not in kinds
    assert kinds[0] == EventKind.TURN_START
    assert kinds[-1] == EventKind.TURN_END


async def test_a_rejected_turn_still_records_that_it_happened() -> None:
    """The log records the attempt, so a rejection is auditable."""
    loop = _loop()

    async def veto(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Reject:
        return Reject(reason="quiet hours")

    loop.instance.ctx.events.on(AGENT_PRE_STEP, veto)
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.TURN_END)
    assert end.data["reason"] == "rejected"
    assert end.data["detail"] == "quiet hours"


async def test_the_first_step_reports_reason_initial() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    seen: list[str] = []

    async def observe(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Any:
        seen.append(reason.value)
        return await next_()

    loop.instance.ctx.events.on(AGENT_PRE_STEP, observe)
    loop.instance.inbox.followup(_say("hello"))

    await loop.run_until_idle()

    assert seen == [PreStepReason.INITIAL.value]


async def test_steering_is_claimed_at_the_step_boundary() -> None:
    loop = _loop(ScriptedResponse((), StopReason.STOP))
    loop.instance.inbox.followup(_say("first"))
    loop.instance.inbox.steer(_say("steered"))

    await loop.run_until_idle()

    texts = [text_of(m) for m in derive_messages(loop.instance.log)]
    assert "steered" in texts


async def test_injected_context_does_not_start_work_on_its_own() -> None:
    """Silent by design: it waits for something that does wake the driver."""
    loop = _loop()
    loop.instance.inbox.inject(_say("file changed"))

    await loop.run_until_idle()

    assert len(loop.instance.log) == 0


async def test_a_system_override_applies_to_one_step_only() -> None:
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((), StopReason.STOP), ScriptedResponse((), StopReason.STOP)
    )

    async def override(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Enter:
        return Enter(messages=messages, system_override="one-off")

    dispose = loop.instance.ctx.events.on(AGENT_PRE_STEP, override)
    loop.instance.inbox.followup(_say("first"))
    await loop.run_until_idle()

    dispose()
    loop.instance.inbox.followup(_say("second"))
    await loop.run_until_idle()

    assert adapter.requests[0].system == "one-off"
    assert adapter.requests[1].system != "one-off"


async def test_a_rejection_at_a_later_boundary_ends_the_turn() -> None:
    """The decision is asked at every step boundary, not only the first: a
    turn already in flight can still be stopped before its next request."""
    loop = _loop(
        ScriptedResponse((ToolCallBlock(id="t1", name="echo", arguments={}),), StopReason.TOOL_USE),
        ScriptedResponse((TextBlock(text="never reached"),), StopReason.STOP),
    )
    loop.tools.register("echo", lambda args: "ran")

    async def veto_after_the_first(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Any:
        if reason is PreStepReason.INITIAL:
            return await next_()
        return Reject(reason="enough")

    loop.instance.ctx.events.on(AGENT_PRE_STEP, veto_after_the_first)
    loop.instance.inbox.followup(_say("go"))

    await loop.run_until_idle()

    steps = [e for e in loop.instance.log.events if e.kind == EventKind.STEP_START]
    assert len(steps) == 1

    end = next(e for e in loop.instance.log.events if e.kind == EventKind.TURN_END)
    assert end.data["reason"] == "rejected"


async def test_a_history_window_limits_what_the_step_sends() -> None:
    """Pi's per-call max_history_turns: the window applies to one step, and
    the log keeps everything regardless."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((TextBlock(text="one"),), StopReason.STOP),
        ScriptedResponse((TextBlock(text="two"),), StopReason.STOP),
    )

    async def narrow(
        instance: AgentInstance,
        reason: PreStepReason,
        messages: tuple[UserMessage, ...],
        next_: Any,
    ) -> Enter:
        return Enter(messages=messages, history_window=1)

    loop.instance.inbox.followup(_say("first"))
    await loop.run_until_idle()

    loop.instance.ctx.events.on(AGENT_PRE_STEP, narrow)
    loop.instance.inbox.followup(_say("second"))
    await loop.run_until_idle()

    assert len(adapter.requests[0].messages) == 1
    assert len(adapter.requests[1].messages) == 1
    assert text_of(adapter.requests[1].messages[0]) == "second"
    assert len(derive_messages(loop.instance.log)) == 4
