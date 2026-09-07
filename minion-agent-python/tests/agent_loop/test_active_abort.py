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

from minion_agent.agent.events import AGENT_LIFECYCLE_EVENT
from minion_agent.agent.instance import AgentActiveError
from minion_agent.agent.projection import MessageStart
from minion_agent.llm import StopReason, ToolCallBlock
from minion_agent.llm.adapters.mock import ScriptedResponse
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
        call: Any, definition: Any, arguments: Any, next_: Any
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
