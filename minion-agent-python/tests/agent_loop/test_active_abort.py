"""Layer 09: active abort/cancellation propagation through the real `AgentLoop` seam.

Full behavior matrix, priority ordering, and the discriminating witnesses live at the layer
they are actually decided: `tests/runtime/test_signal.py` (`RunSignal` itself),
`tests/agent/test_instance.py` (`AgentInstance.signal`/`abort()`), `tests/tools/test_execute.py`/
`test_batch.py` (the preflight priority matrix and the sequential/parallel batch algorithms,
including the required A/B/C parallel witness). This file proves the WIRING through the real
`AgentLoop`/`AgentInstance`/`LlmService`/tool-execution seam end to end -- a canonical runner
would otherwise be the only thing exercising this integration, and no canonical scenario exists
yet for this layer (see `assurance/layers/09-active-abort-python.md`)."""

from typing import Any

import pytest

from minion_agent.agent.events import AGENT_LIFECYCLE_EVENT, AGENT_TRANSFORM_CONTEXT
from minion_agent.agent.identity import AgentStatus
from minion_agent.agent.instance import AgentActiveError
from minion_agent.agent.projection import MessageStart
from minion_agent.llm import StopReason, TextBlock, ToolCallBlock, UserMessage
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.runtime import RunAbortController
from minion_agent.tools.decisions import Proceed
from minion_agent.tools.events import TOOLS_PRE_EXECUTE

from .test_single_turn import _loop_with_adapter, _register, _say


async def test_signal_is_none_before_and_after_a_run() -> None:
    """Matches pinned Pi's own `Agent.signal`: `undefined` while idle."""
    loop = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))[0]
    assert loop.instance.signal is None

    await loop.prompt(_say("hello"))

    assert loop.instance.signal is None


async def test_the_active_runs_signal_reaches_the_llm_request() -> None:
    """The SAME per-run signal object reaches `Request.signal`, matching pinned Pi's own
    `streamFunction(model, context, {...config, signal})` (`agent-loop.ts:308-312`)."""
    loop, adapter = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))
    seen_signal: list[Any] = []

    def observe(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart) and seen_signal == []:
            seen_signal.append(instance.signal)

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe)

    await loop.prompt(_say("hello"))

    assert seen_signal[0] is not None
    assert adapter.requests[0].signal is seen_signal[0]


async def test_reset_stays_illegal_while_an_aborted_run_is_still_settling() -> None:
    """`abort()` only requests cancellation -- it does not itself settle the run, so `reset()`
    still rejects until the run has actually finished, exactly as for a non-aborted active run."""
    loop = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))[0]

    def abort_from_pre_step(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart):
            instance.abort()
            with pytest.raises(AgentActiveError):
                instance.reset()

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, abort_from_pre_step)

    await loop.prompt(_say("hello"))  # must not raise -- nothing forces the run to stop

    assert loop.instance.signal is None  # settled normally afterward


async def test_abort_mid_parallel_tool_batch_truncates_through_the_real_loop() -> None:
    """Integration proof of the wiring: `AgentInstance.abort()`, called from B's own before-hook
    DURING preflight (the point pinned Pi's own `prepareToolCall` checks it -- see
    `tests/tools/test_batch.py`'s own unit-level A/B/C witness), reaches `AgentLoop._run_step`'s
    own `execute_batch` call and truncates the batch through the REAL loop, not `execute_batch`
    called directly. Aborting from INSIDE `execute()` instead would be too late to demonstrate
    this: preflight for every call in a parallel batch already completes, sequentially, before
    any call's own `execute()` begins, so all three would already be committed to running
    concurrently by the time an abort from inside one of their bodies fired."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse(
            (
                ToolCallBlock(id="t1", name="a", arguments={}),
                ToolCallBlock(id="t2", name="b", arguments={}),
                ToolCallBlock(id="t3", name="c", arguments={}),
            ),
            StopReason.TOOL_USE,
        ),
        ScriptedResponse((), StopReason.STOP),
    )
    ran: list[str] = []

    async def abort_from_bs_before_hook(
        call: Any, definition: Any, arguments: Any, signal: Any, next_: Any
    ) -> Any:
        if call.name == "b":
            loop.instance.abort()
        return Proceed(arguments=arguments)

    loop.instance.ctx.events.on(TOOLS_PRE_EXECUTE, abort_from_bs_before_hook)

    _register(loop, "a", lambda tool_call_id, args: ran.append("a") or "a")
    _register(loop, "b", lambda tool_call_id, args: ran.append("b") or "b")
    _register(loop, "c", lambda tool_call_id, args: ran.append("c") or "c")

    await loop.prompt(_say("go"))  # must not raise -- abort mid-batch does not end the run

    assert "c" not in ran  # C was never even preflighted
    assert set(ran) == {"a"}  # B's own execute() never runs -- its preflight aborted it
    assert adapter.requests[1].signal is not None
    assert adapter.requests[1].signal.aborted is True  # the second turn's own request sees it


async def test_exception_after_abort_is_settled_as_aborted_not_error() -> None:
    """`L09-R002`: pinned Pi's own `runWithLifecycle` calls `handleRunFailure(error,
    abortController.signal.aborted)`, and the synthesized failure's `stop_reason` is read from
    that CURRENT boolean at settlement time -- causation is deliberately irrelevant. A listener
    calls `instance.abort()` and then raises exactly once (guarded so it does not also fire
    during the recovery dispatch it triggers); the synthesized failure must report
    `StopReason.ABORTED`, not `StopReason.ERROR`, even though the raised exception has nothing to
    do with the abort itself."""
    loop = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))[0]
    fired = False

    def abort_then_raise(instance: Any, event: Any) -> None:
        nonlocal fired
        if isinstance(event, MessageStart) and not fired:
            fired = True
            instance.abort()
            raise RuntimeError("boom-after-abort")

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, abort_then_raise)

    seen: list[Any] = []

    def observe_failure(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart) and event.message.error_message is not None:
            seen.append(event.message)

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe_failure)

    await loop.prompt(_say("hello"))  # must not raise -- the recovery dispatch settles it

    assert seen[0].stop_reason is StopReason.ABORTED
    assert seen[0].error_message == "boom-after-abort"


async def test_an_unrelated_exception_without_abort_is_still_settled_as_error() -> None:
    """Regression paired with the above: without any `abort()` call, the SAME kind of listener
    failure is still classified `error`, matching pinned Pi's own default."""
    loop = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))[0]
    fired = False

    def just_raise(instance: Any, event: Any) -> None:
        nonlocal fired
        if isinstance(event, MessageStart) and not fired:
            fired = True
            raise RuntimeError("boom-no-abort")

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, just_raise)

    seen: list[Any] = []

    def observe_failure(instance: Any, event: Any) -> None:
        if isinstance(event, MessageStart) and event.message.error_message is not None:
            seen.append(event.message)

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, observe_failure)

    await loop.prompt(_say("hello"))

    assert seen[0].stop_reason is StopReason.ERROR
    assert seen[0].error_message == "boom-no-abort"


async def test_transform_context_receives_messages_and_the_active_signal() -> None:
    """`L09-R005`: pinned Pi's own `transformContext(messages, signal)`, invoked immediately
    before every provider request, with the SAME per-run signal every other consumer receives."""
    loop = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))[0]
    seen: list[Any] = []

    async def observe_transform(instance: Any, messages: Any, signal: Any, next_: Any) -> Any:
        seen.append((messages, signal))
        return await next_()

    loop.instance.ctx.events.on(AGENT_TRANSFORM_CONTEXT, observe_transform)

    await loop.prompt(_say("hello"))

    assert len(seen[0][0]) == 1  # the admitted prompt
    assert seen[0][1] is not None  # the active run's own signal, not None


async def test_transform_context_output_is_provider_local_not_persistent() -> None:
    """`L09-R005`'s own required discriminating witness: a transform's own replacement reaches
    THIS request's `Request.messages`, but never the run's own persistent/run-local transcript --
    the NEXT turn's own request starts from the UNTRANSFORMED history again, matching pinned Pi's
    own `streamAssistantResponse` reassigning only its local `messages` variable, never
    `currentContext.messages`."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse(
            (ToolCallBlock(id="t1", name="a", arguments={}),),
            StopReason.TOOL_USE,
        ),
        ScriptedResponse((), StopReason.STOP),
    )
    _register(loop, "a", lambda tool_call_id, args: "a")

    marker = UserMessage(content=(TextBlock(text="INJECTED"),), timestamp=0)

    async def inject_marker(instance: Any, messages: Any, signal: Any, next_: Any) -> Any:
        return (*messages, marker)

    loop.instance.ctx.events.on(AGENT_TRANSFORM_CONTEXT, inject_marker)

    await loop.prompt(_say("go"))

    # First request: the transform's own injected marker is present (provider-local for THAT
    # request).
    assert marker in adapter.requests[0].messages
    # Second request (after the tool call): the transform ran again and injected its OWN marker
    # into that request too, but the marker from the FIRST call never became part of the
    # persistent transcript the second request's own history was built from -- it appears
    # exactly once per request, not accumulating.
    assert adapter.requests[1].messages.count(marker) == 1
    # The durable, offline-visible transcript never contains the injected marker at all.
    assert marker not in loop.instance.messages


async def test_a_transform_listener_cannot_redirect_a_later_listener_to_a_replacement_signal() -> (
    None
):
    """`L09-R006`: the same authoritative-signal witness as `tools/pre-execute`/`tools/
    post-execute`, for `AGENT_TRANSFORM_CONTEXT` -- listener A delegates with a FABRICATED
    replacement signal; listener B must still observe the ORIGINAL (the same object the request
    that follows actually receives), not A's forgery."""
    loop, adapter = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))
    forged = RunAbortController().signal
    seen: list[Any] = []

    async def listener_a(instance: Any, messages: Any, signal: Any, next_: Any) -> Any:
        return await next_(instance, messages, forged)

    async def listener_b(instance: Any, messages: Any, signal: Any, next_: Any) -> Any:
        seen.append(signal)
        return await next_()

    loop.instance.ctx.events.on(AGENT_TRANSFORM_CONTEXT, listener_a)
    loop.instance.ctx.events.on(AGENT_TRANSFORM_CONTEXT, listener_b)

    await loop.prompt(_say("hello"))

    assert seen[0] is not forged  # NOT the forgery listener A delegated with
    assert seen[0] is adapter.requests[0].signal  # the SAME signal the real request received


async def test_the_running_status_observer_sees_a_live_signal_and_the_idle_observer_sees_none() -> (
    None
):
    """`L09-R007`: `AgentInstance.set_status` emits `agent/status`/calls `on_status_change`
    SYNCHRONOUSLY -- an observer reading `instance.signal` (or calling `instance.abort()`) during
    the RUNNING transition must see the run's own real, live signal, not `None`; one reading it
    during the IDLE transition must see `None`, not the just-finished run's stale signal. An
    independent review's own witness found the opposite: the controller was installed/removed
    AFTER each status publish, so a RUNNING observer's own `abort()` call was a no-op (no
    controller existed yet) and an IDLE observer saw the previous run's still-live signal."""
    loop, adapter = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))
    observations: list[tuple[str, bool]] = []

    def on_status_change(status: AgentStatus) -> None:
        if status is AgentStatus.RUNNING:
            observations.append(("running", loop.instance.signal is not None))
            loop.instance.abort()
        elif status is AgentStatus.IDLE:
            observations.append(("idle", loop.instance.signal is None))

    loop.instance.on_status_change = on_status_change

    await loop.prompt(_say("hello"))

    assert observations == [("running", True), ("idle", True)]
    assert adapter.requests[0].signal is not None
    assert adapter.requests[0].signal.aborted is True  # the RUNNING observer's own abort() landed


async def test_a_represented_aborted_terminal_is_unaffected_by_layer_09() -> None:
    """Regression: pinned Pi's own represented-`aborted` short-circuit (already Layer-08-owned,
    `runLoop`'s `stopReason === "aborted"` check) is a scripted terminal, not something Layer 09
    itself produces -- confirming Layer 09's own wiring did not disturb it."""
    loop = _loop_with_adapter(ScriptedResponse((), StopReason.ABORTED, error_message="cancelled"))[
        0
    ]

    await loop.prompt(_say("hello"))

    assert loop.instance.error_message == "cancelled"
    assert loop.instance.signal is None
