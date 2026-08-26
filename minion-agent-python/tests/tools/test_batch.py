"""Batch semantics: pi's contagion rule and pi's two emission orders."""

import asyncio
from typing import Any

from minion_agent.llm import ToolCallBlock, text_of
from minion_agent.runtime import Context
from minion_agent.tools.batch import execute_batch, execute_length_stop_batch
from minion_agent.tools.decisions import Proceed
from minion_agent.tools.definition import ExecutionMode, ToolDefinition
from minion_agent.tools.events import TOOLS_EXECUTION_START, TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult


def _ctx() -> Context:
    ctx = Context()
    declare_tools_events(ctx.events)
    return ctx


def _call(call_id: str, name: str, **arguments: Any) -> ToolCallBlock:
    return ToolCallBlock(id=call_id, name=name, arguments=dict(arguments))


def _tool(name: str, execute: Any, mode: ExecutionMode = ExecutionMode.PARALLEL) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=execute,
        label=name,
        mode=mode,
    )


def _registry(*definitions: ToolDefinition) -> ToolRegistry:
    registry = ToolRegistry()
    for definition in definitions:
        registry.register(definition)
    return registry


async def test_an_empty_batch_does_not_terminate() -> None:
    """The fold is over every result; with no results there is nothing that
    every result agrees on, and an empty batch must not end a turn."""
    outcome = await execute_batch((), registry=_registry(), ctx=_ctx())

    assert outcome.results == ()
    assert not outcome.terminate


async def test_results_come_back_in_assistant_source_order() -> None:
    """Pi's rule: messages follow the order the model asked, whatever the
    order the tools happened to finish in."""
    gate = asyncio.Event()

    async def slow(tool_call_id: str, args: dict[str, Any]) -> str:
        await gate.wait()
        return "slow"

    async def fast(tool_call_id: str, args: dict[str, Any]) -> str:
        gate.set()
        return "fast"

    outcome = await execute_batch(
        (_call("t1", "slow"), _call("t2", "fast")),
        registry=_registry(_tool("slow", slow), _tool("fast", fast)),
        ctx=_ctx(),
    )

    assert [text_of(r.to_message()) for r in outcome.results] == ["slow", "fast"]
    # Completion order is the opposite of source order here (t2 finishes
    # first), which is what makes this test distinguish the two orders
    # instead of passing under an implementation that used one order twice.
    assert outcome.completion_order == ("t2", "t1")


async def test_completion_order_is_recorded_separately() -> None:
    """The other of pi's two orders: `tool_execution_end` follows completion.
    One log carries both, so the batch has to report both."""
    gate = asyncio.Event()

    async def slow(tool_call_id: str, args: dict[str, Any]) -> str:
        await gate.wait()
        return "slow"

    async def fast(tool_call_id: str, args: dict[str, Any]) -> str:
        gate.set()
        return "fast"

    outcome = await execute_batch(
        (_call("t1", "slow"), _call("t2", "fast")),
        registry=_registry(_tool("slow", slow), _tool("fast", fast)),
        ctx=_ctx(),
    )

    assert outcome.completion_order == ("t2", "t1")
    assert outcome.completion_index("t1") == 1
    # Distinguished from source order: results (source order) must still be
    # t1 first even though t2 completed first.
    assert [text_of(r.to_message()) for r in outcome.results] == ["slow", "fast"]


async def test_parallel_tools_actually_overlap() -> None:
    """Otherwise 'parallel' is a label rather than a behavior."""
    started = asyncio.Event()

    async def first(tool_call_id: str, args: dict[str, Any]) -> str:
        started.set()
        return "first"

    async def second(tool_call_id: str, args: dict[str, Any]) -> str:
        await asyncio.wait_for(started.wait(), timeout=1)
        return "second"

    outcome = await execute_batch(
        (_call("t1", "second"), _call("t2", "first")),
        registry=_registry(_tool("second", second), _tool("first", first)),
        ctx=_ctx(),
    )

    assert [text_of(r.to_message()) for r in outcome.results] == ["second", "first"]


async def test_preflight_is_sequential_and_settles_before_any_execute_begins() -> None:
    """`IR-L06-001`: pinned Pi's `executeToolCallsParallel` preflights (resolve/prepare/validate/
    before-hook) every call strictly sequentially, in source order, and starts `execute()`
    concurrently only once every call in the batch has survived preflight -- a barrier a single
    `asyncio.gather` over the whole per-call pipeline cannot express, since that preflights every
    call concurrently too. This test distinguishes the two: `A`'s before-hook awaits a real
    scheduler tick, giving a naive concurrent-preflight implementation a chance to run `B`'s entire
    pipeline -- start through execute -- while `A` is still suspended in preflight. Under the
    correct barrier, `B`'s preflight cannot even begin until `A`'s has fully settled, so
    `tool_execution_start` and the before-hook must interleave as `start_A, before_A, start_B,
    before_B` -- never `start_A, start_B, ...` -- and neither `execute_A` nor `execute_B` may
    appear before both before-hooks have completed.
    """
    ctx = _ctx()
    events: list[str] = []

    ctx.events.on(TOOLS_EXECUTION_START, lambda call_id, name, args: events.append(f"start_{name}"))

    async def traced_before(call: Any, definition: Any, arguments: Any, next_: Any) -> Proceed:
        if call.name == "a":
            await asyncio.sleep(0)
        events.append(f"before_{call.name}")
        return Proceed(arguments=arguments)

    ctx.events.on(TOOLS_PRE_EXECUTE, traced_before)

    def traced_execute(name: str) -> Any:
        def execute(tool_call_id: str, args: dict[str, Any]) -> str:
            events.append(f"execute_{name}")
            return name

        return execute

    outcome = await execute_batch(
        (_call("t1", "a"), _call("t2", "b")),
        registry=_registry(_tool("a", traced_execute("a")), _tool("b", traced_execute("b"))),
        ctx=ctx,
    )

    assert events[:4] == ["start_a", "before_a", "start_b", "before_b"]
    assert sorted(events[4:]) == ["execute_a", "execute_b"]
    assert [text_of(r.to_message()) for r in outcome.results] == ["a", "b"]


async def test_an_immediate_preflight_failure_does_not_block_a_later_calls_preflight() -> None:
    """`IR-L06-001` scenario C: an immediate outcome (here, invalid arguments) finalizes right
    there in the sequential preflight phase -- it must not prevent, delay, or serialize behind it
    the NEXT call's own preflight, which pinned Pi still runs (and, once survived, executes)
    exactly as if the failure had not happened."""
    ctx = _ctx()
    events: list[str] = []
    ctx.events.on(TOOLS_EXECUTION_START, lambda call_id, name, args: events.append(f"start_{name}"))

    async def traced_before(call: Any, definition: Any, arguments: Any, next_: Any) -> Proceed:
        events.append(f"before_{call.name}")
        return Proceed(arguments=arguments)

    ctx.events.on(TOOLS_PRE_EXECUTE, traced_before)

    outcome = await execute_batch(
        (_call("t1", "missing"), _call("t2", "b")),
        registry=_registry(_tool("b", lambda tool_call_id, args: "b")),
        ctx=ctx,
    )

    # "missing" never resolves, so its before-hook never runs at all -- but "b"'s
    # start/preflight/execute must still happen, unblocked by "missing"'s immediate failure.
    assert events == ["start_missing", "start_b", "before_b"]
    assert outcome.results[0].is_error
    assert text_of(outcome.results[1].to_message()) == "b"


async def test_one_sequential_tool_serializes_the_whole_batch() -> None:
    """Pi's contagion rule, deliberately not DSH's grouping (design spec
    section 6). The parallel-safe call does not overlap the exclusive one."""
    live: list[str] = []
    peak = 0

    async def record(name: str) -> str:
        nonlocal peak
        live.append(name)
        peak = max(peak, len(live))
        await asyncio.sleep(0)
        live.remove(name)
        return name

    outcome = await execute_batch(
        (_call("t1", "exclusive"), _call("t2", "shared")),
        registry=_registry(
            _tool(
                "exclusive",
                lambda tool_call_id, args: record("exclusive"),
                ExecutionMode.SEQUENTIAL,
            ),
            _tool("shared", lambda tool_call_id, args: record("shared")),
        ),
        ctx=_ctx(),
    )

    assert peak == 1
    assert [text_of(r.to_message()) for r in outcome.results] == ["exclusive", "shared"]


async def test_a_sequential_tool_serializes_the_batch_from_any_position() -> None:
    """Sibling of `test_one_sequential_tool_serializes_the_whole_batch`, with
    the `SEQUENTIAL` tool last instead of first. Contagion is a property of
    the batch, not of its first element: a scan that only inspected
    `calls[0]` would pass every other test in this suite while missing
    contagion for exactly the common case of a read requested before a
    write."""
    live: list[str] = []
    peak = 0

    async def record(name: str) -> str:
        nonlocal peak
        live.append(name)
        peak = max(peak, len(live))
        await asyncio.sleep(0)
        live.remove(name)
        return name

    outcome = await execute_batch(
        (_call("t1", "shared"), _call("t2", "exclusive")),
        registry=_registry(
            _tool("shared", lambda tool_call_id, args: record("shared")),
            _tool(
                "exclusive",
                lambda tool_call_id, args: record("exclusive"),
                ExecutionMode.SEQUENTIAL,
            ),
        ),
        ctx=_ctx(),
    )

    assert peak == 1
    assert [text_of(r.to_message()) for r in outcome.results] == ["shared", "exclusive"]


async def test_a_sequential_batch_completes_in_source_order() -> None:
    """Proves ordering fidelity within the sequential branch -- that the loop
    awaits calls in the order given -- not that contagion fired in the first
    place. The `peak` tests above are what prove that."""
    outcome = await execute_batch(
        (_call("t1", "a"), _call("t2", "b")),
        registry=_registry(
            _tool("a", lambda tool_call_id, args: "a", ExecutionMode.SEQUENTIAL),
            _tool("b", lambda tool_call_id, args: "b"),
        ),
        ctx=_ctx(),
    )

    assert outcome.completion_order == ("t1", "t2")


async def test_an_unknown_tool_does_not_serialize_the_batch() -> None:
    """It never runs, so it has no exclusivity to spread. Treating an unknown
    name as exclusive would let a model typo serialize everything.

    Strengthened per task-10 ruling: asserting only `results[1].is_error`
    cannot fail under a serializing implementation, since a serialized first
    call ("second") would time out waiting on `started` and *also* produce
    an error -- just at a different index, for a different reason. The
    assertions below on completion order and the first result's success are
    the ones that actually distinguish parallel from serialized execution.
    """
    started = asyncio.Event()

    async def first(tool_call_id: str, args: dict[str, Any]) -> str:
        started.set()
        return "first"

    async def second(tool_call_id: str, args: dict[str, Any]) -> str:
        await asyncio.wait_for(started.wait(), timeout=1)
        return "second"

    outcome = await execute_batch(
        (_call("t1", "second"), _call("t2", "missing"), _call("t3", "first")),
        registry=_registry(_tool("second", second), _tool("first", first)),
        ctx=_ctx(),
    )

    assert outcome.results[1].is_error
    # Strengthened per task-10 ruling: under a serializing implementation,
    # "second" (t1) runs first and blocks on `started` until "first" (t3)
    # sets it -- but "first" cannot run until "second" finishes, so "second"
    # times out and becomes an error, and (being serial) still completes
    # first. Under a genuinely parallel batch, "missing" (t2) and "first"
    # (t3) both resolve without ever suspending, so they complete before
    # "second" (t1) ever wakes up; "missing" is scheduled first and hits no
    # real suspension point either, so it completes before "first" does.
    # These two assertions therefore fail under serialization and pass only
    # under a genuinely parallel batch.
    assert outcome.completion_order[0] == "t2"
    assert not outcome.results[0].is_error


async def test_the_fold_needs_every_result_to_terminate() -> None:
    def terminating(tool_call_id: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(tool_call_id="", content=(), tool_name="stop", terminate=True)

    outcome = await execute_batch(
        (_call("t1", "stop"), _call("t2", "go")),
        registry=_registry(
            _tool("stop", terminating), _tool("go", lambda tool_call_id, args: "go")
        ),
        ctx=_ctx(),
    )

    assert not outcome.terminate


async def test_a_unanimous_batch_terminates() -> None:
    def terminating(tool_call_id: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(tool_call_id="", content=(), tool_name="stop", terminate=True)

    outcome = await execute_batch(
        (_call("t1", "stop"), _call("t2", "stop")),
        registry=_registry(_tool("stop", terminating)),
        ctx=_ctx(),
    )

    assert outcome.terminate


async def test_every_call_gets_a_result_even_when_one_raises() -> None:
    def broken(tool_call_id: str, args: dict[str, Any]) -> str:
        raise RuntimeError("boom")

    outcome = await execute_batch(
        (_call("t1", "ok"), _call("t2", "broken"), _call("t3", "ok")),
        registry=_registry(_tool("ok", lambda tool_call_id, args: "ok"), _tool("broken", broken)),
        ctx=_ctx(),
    )

    assert len(outcome.results) == 3
    assert [r.tool_call_id for r in outcome.results] == ["t1", "t2", "t3"]
    assert outcome.results[1].is_error


async def test_the_run_level_default_serializes_the_batch_even_without_a_sequential_tool() -> None:
    """Pinned Pi's `config.toolExecution === "sequential" || hasSequentialToolCall`
    (`packages/agent/src/agent-loop.ts::executeToolCalls`) -- the run-level default alone can
    force sequential scheduling, independent of any per-tool `execution_mode` (`TOOL-017`)."""
    live: list[str] = []
    peak = 0

    async def record(tool_call_id: str, args: dict[str, Any]) -> str:
        nonlocal peak
        live.append(tool_call_id)
        peak = max(peak, len(live))
        await asyncio.sleep(0)
        live.remove(tool_call_id)
        return tool_call_id

    outcome = await execute_batch(
        (_call("t1", "a"), _call("t2", "b")),
        registry=_registry(_tool("a", record), _tool("b", record)),
        ctx=_ctx(),
        default_mode=ExecutionMode.SEQUENTIAL,
    )

    assert peak == 1
    assert outcome.completion_order == ("t1", "t2")


async def test_the_run_level_default_is_parallel_when_unspecified() -> None:
    """Matches pinned Pi's own default ("Default: parallel",
    `AgentLoopConfig.toolExecution?`) -- omitting `default_mode` must not accidentally
    serialize a batch with no per-tool override."""
    started = asyncio.Event()

    async def first(tool_call_id: str, args: dict[str, Any]) -> str:
        started.set()
        return "first"

    async def second(tool_call_id: str, args: dict[str, Any]) -> str:
        await asyncio.wait_for(started.wait(), timeout=1)
        return "second"

    outcome = await execute_batch(
        (_call("t1", "second"), _call("t2", "first")),
        registry=_registry(_tool("second", second), _tool("first", first)),
        ctx=_ctx(),
    )

    assert [text_of(r.to_message()) for r in outcome.results] == ["second", "first"]


async def test_length_stop_executes_nothing_and_fails_every_call() -> None:
    """Pinned Pi's `failToolCallsFromTruncatedMessage`: zero execution, every call becomes the
    same error result, in source order, `terminate` always `False` (`TOOL-017`). No registry is
    even consulted -- not even an unknown-tool lookup happens for these calls."""
    outcome = await execute_length_stop_batch(
        (_call("t1", "a"), _call("t2", "unknown-entirely")), ctx=_ctx()
    )

    assert len(outcome.results) == 2
    assert [r.tool_call_id for r in outcome.results] == ["t1", "t2"]
    assert all(r.is_error for r in outcome.results)
    assert all("output token limit" in text_of(r.to_message()) for r in outcome.results)
    assert not outcome.terminate
