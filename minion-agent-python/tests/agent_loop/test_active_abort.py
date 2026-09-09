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

from minion_agent.agent.envelope import ClaimPolicy, InboxTarget
from minion_agent.agent.events import AGENT_LIFECYCLE_EVENT, AGENT_STATUS, AGENT_TRANSFORM_CONTEXT
from minion_agent.agent.identity import AgentStatus
from minion_agent.agent.instance import AgentActiveError
from minion_agent.agent.projection import MessageStart
from minion_agent.llm import StopReason, TextBlock, ToolCallBlock, UserMessage, text_of
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


# -- Layer 09, `L09-R007` convergence: RUNNING/IDLE notification-failure atomicity -------------


async def test_a_raising_on_status_change_on_running_rolls_back_and_propagates() -> None:
    """Convergence witness 1: a synchronous `on_status_change` that raises on `RUNNING` means the
    run never validly started -- the signal is discarded, `status` is forced back to `IDLE`
    WITHOUT re-invoking the same failing callback, and the observer's own exception propagates
    directly out of `prompt()`. A second `prompt()` call must then succeed normally."""
    loop, adapter = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))

    def on_status_change(status: AgentStatus) -> None:
        if status is AgentStatus.RUNNING:
            raise RuntimeError("status-boom")

    loop.instance.on_status_change = on_status_change

    with pytest.raises(RuntimeError, match="status-boom"):
        await loop.prompt(_say("hello"))

    assert loop.instance.status is AgentStatus.IDLE
    assert loop.instance.signal is None
    assert adapter.requests == []

    loop.instance.on_status_change = None
    await loop.prompt(_say("hello again"))  # must not raise AgentActiveError
    assert loop.instance.status is AgentStatus.IDLE


async def test_a_raising_agent_status_listener_on_running_short_circuits_on_status_change() -> None:
    """Convergence witness 2: the same outcome for a raw `AGENT_STATUS` EMIT listener, and proof
    that `EventBus.emit`'s own fail-fast rule means a SEPARATE `on_status_change` callback is
    never reached at all once an earlier listener has already thrown."""
    loop, adapter = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))
    on_status_change_calls: list[AgentStatus] = []

    def emit_listener(instance: Any, status: Any) -> None:
        if status is AgentStatus.RUNNING:
            raise RuntimeError("emit-boom")

    loop.instance.ctx.events.on(AGENT_STATUS, emit_listener)
    loop.instance.on_status_change = on_status_change_calls.append

    with pytest.raises(RuntimeError, match="emit-boom"):
        await loop.prompt(_say("hello"))

    assert loop.instance.status is AgentStatus.IDLE
    assert loop.instance.signal is None
    assert adapter.requests == []
    assert on_status_change_calls == []  # never reached -- emit's own fail-fast rule


async def test_a_raising_on_status_change_on_idle_does_not_hide_a_successful_run() -> None:
    """Convergence witness 3: a run that completes successfully must have its own outcome
    committed regardless of whether the LATER, unrelated IDLE notification itself fails --
    signal/streaming_message/pending_tool_calls/status are all already fully IDLE-consistent by
    the time the notification's own exception is observed, and the exception then propagates."""
    loop = _loop_with_adapter(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))[0]
    fired = False

    def on_status_change(status: AgentStatus) -> None:
        nonlocal fired
        if status is AgentStatus.IDLE and not fired:
            fired = True
            raise RuntimeError("idle-boom")

    loop.instance.on_status_change = on_status_change

    with pytest.raises(RuntimeError, match="idle-boom"):
        await loop.prompt(_say("hello"))

    assert loop.instance.status is AgentStatus.IDLE
    assert loop.instance.signal is None
    assert loop.instance.streaming_message is None
    assert loop.instance.pending_tool_calls == frozenset()

    assert any(text_of(m) == "hi" for m in loop.instance.messages)  # the run's own outcome stands


async def test_a_raising_on_status_change_on_idle_does_not_hide_a_settled_failure() -> None:
    """Convergence witness 4: the same guarantee for a run that settled as a failure via
    `_settle_run_failure` -- the settled failure message is not lost, duplicated, or converted
    into something else by the separate IDLE-notification failure."""
    loop = _loop_with_adapter(ScriptedResponse((), StopReason.STOP))[0]
    run_failed = False

    def fail_the_run(instance: Any, event: Any) -> None:
        nonlocal run_failed
        if isinstance(event, MessageStart) and not run_failed:
            run_failed = True
            raise RuntimeError("run-boom")

    loop.instance.ctx.events.on(AGENT_LIFECYCLE_EVENT, fail_the_run)

    idle_fired = False

    def on_status_change(status: AgentStatus) -> None:
        nonlocal idle_fired
        if status is AgentStatus.IDLE and not idle_fired:
            idle_fired = True
            raise RuntimeError("idle-boom")

    loop.instance.on_status_change = on_status_change

    with pytest.raises(RuntimeError, match="idle-boom"):
        await loop.prompt(_say("hello"))

    assert loop.instance.status is AgentStatus.IDLE
    assert loop.instance.signal is None
    assert loop.instance.streaming_message is None
    assert loop.instance.pending_tool_calls == frozenset()
    failures = [
        m for m in loop.instance.messages if getattr(m, "error_message", None) == "run-boom"
    ]
    assert len(failures) == 1  # settled exactly once -- not lost, not duplicated


# -- Layer 09, `L09-R007` convergence: preclaimed inbox input, RUNNING-failure entry rollback ---


async def test_continue_restores_preclaimed_steering_on_a_running_failure() -> None:
    """Convergence witness 5: `continue_()`'s own steering branch destructively claims from
    `Inbox` BEFORE `_run_wrapped` is ever called. A RUNNING-notification failure must restore
    that claimed envelope -- exact id/message/origin, not a copy -- so it is not lost, and a
    later `continue_()` (with the failing observer removed) consumes it exactly once."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP),
        ScriptedResponse((TextBlock(text="steered reply"),), StopReason.STOP),
    )
    await loop.prompt(_say("hello"))
    envelope = loop.instance.inbox.steer(_say("steer me"), origin="s1")

    def on_status_change(status: AgentStatus) -> None:
        if status is AgentStatus.RUNNING:
            raise RuntimeError("status-boom")

    loop.instance.on_status_change = on_status_change

    with pytest.raises(RuntimeError, match="status-boom"):
        await loop.continue_()

    pending = loop.instance.inbox.pending(InboxTarget.NEXT_STEP)
    assert len(pending) == 1
    assert pending[0].id == envelope.id
    assert pending[0].message == envelope.message
    assert pending[0].origin == envelope.origin
    assert len(adapter.requests) == 1  # only the first prompt() -- the failed continue_() sent none

    loop.instance.on_status_change = None  # remove the failing observer before retrying
    await loop.continue_()

    assert loop.instance.inbox.pending(InboxTarget.NEXT_STEP) == ()

    assert any(text_of(m) == "steer me" for m in loop.instance.messages)


async def test_continue_restores_preclaimed_follow_up_on_a_running_failure() -> None:
    """Convergence witness 6: identical to witness 5, for `continue_()`'s own follow-up branch."""
    loop, adapter = _loop_with_adapter(
        ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP),
        ScriptedResponse((TextBlock(text="follow-up reply"),), StopReason.STOP),
    )
    await loop.prompt(_say("hello"))
    envelope = loop.instance.inbox.followup(_say("follow up"), origin="f1")

    def on_status_change(status: AgentStatus) -> None:
        if status is AgentStatus.RUNNING:
            raise RuntimeError("status-boom")

    loop.instance.on_status_change = on_status_change

    with pytest.raises(RuntimeError, match="status-boom"):
        await loop.continue_()

    pending = loop.instance.inbox.pending(InboxTarget.NEXT_TURN)
    assert len(pending) == 1
    assert pending[0].id == envelope.id
    assert pending[0].message == envelope.message
    assert pending[0].origin == envelope.origin
    assert len(adapter.requests) == 1

    loop.instance.on_status_change = None
    await loop.continue_()

    assert loop.instance.inbox.pending(InboxTarget.NEXT_TURN) == ()

    assert any(text_of(m) == "follow up" for m in loop.instance.messages)


async def test_restored_preclaimed_input_precedes_input_the_failing_observer_itself_enqueues() -> (
    None
):
    """Convergence witness 7: `ClaimPolicy.ALL` claims `A, B`; the RUNNING observer itself enqueues
    `C` at the SAME target and then raises. Restoration must PREPEND the claimed batch ahead of
    `C`, not append behind it or lose it -- the queue afterward reads exactly `A, B, C`."""
    loop = _loop_with_adapter(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))[0]
    loop.next_step_policy = ClaimPolicy.ALL
    await loop.prompt(_say("hello"))
    envelope_a = loop.instance.inbox.steer(_say("A"))
    envelope_b = loop.instance.inbox.steer(_say("B"))

    def on_status_change(status: AgentStatus) -> None:
        if status is AgentStatus.RUNNING:
            loop.instance.inbox.steer(_say("C"))
            raise RuntimeError("status-boom")

    loop.instance.on_status_change = on_status_change

    with pytest.raises(RuntimeError, match="status-boom"):
        await loop.continue_()

    pending = loop.instance.inbox.pending(InboxTarget.NEXT_STEP)
    assert [envelope.message for envelope in pending] == [
        envelope_a.message,
        envelope_b.message,
        pending[2].message,
    ]

    assert [text_of(envelope.message) for envelope in pending] == ["A", "B", "C"]


async def test_run_until_idle_restores_a_preclaimed_follow_up_on_a_running_failure() -> None:
    """Convergence witness 8: `run_until_idle()`'s own pump claims a follow-up batch before each
    `_run_wrapped` call. A RUNNING-notification failure must restore it, propagate out of
    `run_until_idle()` itself rather than swallowing or silently retrying, and leave it available
    for a LATER, separate pump call to drain and process."""
    loop = _loop_with_adapter(ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP))[0]
    envelope = loop.instance.inbox.followup(_say("go"), origin="f1")

    def on_status_change(status: AgentStatus) -> None:
        if status is AgentStatus.RUNNING:
            raise RuntimeError("status-boom")

    loop.instance.on_status_change = on_status_change

    with pytest.raises(RuntimeError, match="status-boom"):
        await loop.run_until_idle()

    pending = loop.instance.inbox.pending(InboxTarget.NEXT_TURN)
    assert len(pending) == 1
    assert pending[0].id == envelope.id
    assert pending[0].message == envelope.message

    loop.instance.on_status_change = None
    await loop.run_until_idle()

    assert loop.instance.inbox.pending(InboxTarget.NEXT_TURN) == ()

    assert any(text_of(m) == "go" for m in loop.instance.messages)


# -- Layer 09, `L09-R011`: a reentrant RUNNING observer must not lose unrelated input -----------


async def test_a_running_observer_that_claims_the_peeked_input_does_not_lose_other_input() -> None:
    """`L09-R011` witness 1: `ONE_AT_A_TIME`, queue `A, B`. A synchronous RUNNING observer itself
    calls `inbox.claim(...)`, removing `A` -- the very envelope this run peeked -- and returns
    normally (no exception). `_run_wrapped`'s own commit must not then delete `B`: it was never
    part of this run's own selected batch, and a count-only commit would have deleted it anyway
    since it was now at the queue's own front. The continued turn is scripted as a represented
    `aborted` terminal so the run returns immediately after it, without reaching Layer 08's own
    separate, already-certified POST-turn steering poll (`_run_step`'s own `_claim_step_input`)
    -- which would otherwise legitimately claim `B` itself moments later for an unrelated reason,
    making this witness observe the wrong thing."""
    loop = _loop_with_adapter(
        ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP),
        ScriptedResponse((), StopReason.ABORTED, error_message="terminal"),
    )[0]
    await loop.prompt(_say("hello"))
    loop.instance.inbox.steer(_say("A"))
    envelope_b = loop.instance.inbox.steer(_say("B"))

    def on_status_change(status: AgentStatus) -> None:
        if status is AgentStatus.RUNNING:
            loop.instance.inbox.claim(InboxTarget.NEXT_STEP, ClaimPolicy.ONE_AT_A_TIME)

    loop.instance.on_status_change = on_status_change

    await loop.continue_()  # must not raise -- the observer returns normally

    pending = loop.instance.inbox.pending(InboxTarget.NEXT_STEP)
    assert len(pending) == 1
    assert pending[0].id == envelope_b.id


async def test_a_running_observer_that_clears_and_enqueues_does_not_lose_the_new_input() -> None:
    """`L09-R011` witness 2: `ClaimPolicy.ALL`, queue `A, B`. A synchronous RUNNING observer
    clears the SAME target entirely and enqueues `C` before returning normally.
    `_run_wrapped`'s own commit must not remove `C` as if it were part of the earlier peek -- a
    count-only commit would have deleted it anyway, since it was the only thing at the queue's
    own front. The continued turn is scripted as a represented `aborted` terminal for the same
    reason as witness 1 above: so the run returns immediately, before Layer 08's own separate
    post-turn steering poll could legitimately claim `C` itself for an unrelated reason."""
    loop = _loop_with_adapter(
        ScriptedResponse((TextBlock(text="hi"),), StopReason.STOP),
        ScriptedResponse((), StopReason.ABORTED, error_message="terminal"),
    )[0]
    loop.next_step_policy = ClaimPolicy.ALL
    await loop.prompt(_say("hello"))
    loop.instance.inbox.steer(_say("A"))
    loop.instance.inbox.steer(_say("B"))

    def on_status_change(status: AgentStatus) -> None:
        if status is AgentStatus.RUNNING:
            loop.instance.inbox.clear(InboxTarget.NEXT_STEP)
            loop.instance.inbox.steer(_say("C"))

    loop.instance.on_status_change = on_status_change

    await loop.continue_()  # must not raise -- the observer returns normally

    pending = loop.instance.inbox.pending(InboxTarget.NEXT_STEP)
    assert len(pending) == 1
    assert text_of(pending[0].message) == "C"


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
