"""`L08-D001` (`AG-024`, spec/agent.md "Optional prompt assembler"): an optional, synchronous
`assemble(base, tools)` called while each provider request without a per-step override is built,
over the very tool snapshot whose schemas that request carries."""

from __future__ import annotations

import asyncio
from typing import Any

from minion_agent.agent import AGENT_PRE_STEP, AGENT_PREPARE_NEXT_TURN
from minion_agent.agent.decisions import Enter, PreStepReason, RunConfigUpdate, RunContext
from minion_agent.agent_loop.driver import AgentLoop
from minion_agent.llm import TextBlock, ToolCallBlock, UserMessage
from minion_agent.llm.adapters.mock import ScriptedResponse
from minion_agent.llm.messages import StopReason
from minion_agent.session import EventKind, reconstruct_header
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.result import ToolResult

from .test_single_turn import _loop_with_adapter


def _say(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def _tool(name: str, execute: Any = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
        execute=execute or (lambda tool_call_id, args: "ok"),
        label=name,
    )


def _call(name: str, call_id: str = "t1") -> ScriptedResponse:
    return ScriptedResponse(
        (ToolCallBlock(id=call_id, name=name, arguments={}),), StopReason.TOOL_USE
    )


def _done(text: str = "done") -> ScriptedResponse:
    return ScriptedResponse((TextBlock(text=text),), StopReason.STOP)


class _Recorder:
    """An assembler that renders the base and the snapshot's names, recording each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, base: str, tools: tuple[ToolDefinition, ...]) -> str:
        names = tuple(tool.name for tool in tools)
        self.calls.append((base, names))
        return f"{base}\n\nTOOLS: {','.join(names)}"


def _install(loop: AgentLoop, assembler: Any) -> None:
    loop.prompt_assembler = assembler


def _headers(loop: AgentLoop) -> list[Any]:
    return [e for e in loop.instance.log.events if e.kind == EventKind.REQUEST_HEADER]


def _failed(loop: AgentLoop) -> list[Any]:
    return [
        m for m in loop.instance.messages if getattr(m, "stop_reason", None) is StopReason.ERROR
    ]


def _agent_end_reasons(loop: AgentLoop) -> list[str]:
    return [e.data["reason"] for e in loop.instance.log.events if e.kind == EventKind.AGENT_END]


# ---- no assembler: unchanged ----


async def test_without_an_assembler_the_request_is_exactly_the_stored_prompt() -> None:
    loop, adapter = _loop_with_adapter(_done())
    loop.tools.register(_tool("echo"))
    assert loop.prompt_assembler is None
    await loop.prompt(_say("hi"))
    assert adapter.requests[0].system == "be helpful"
    header = _headers(loop)[0]
    assert reconstruct_header(header, loop.artifacts) == {"system_base": "be helpful"}


# ---- run start, between turns, growth ----


async def test_run_start_assembles_from_the_base_and_the_run_start_snapshot() -> None:
    loop, adapter = _loop_with_adapter(_done())
    loop.tools.register(_tool("echo"))
    recorder = _Recorder()
    _install(loop, recorder)
    await loop.prompt(_say("hi"))
    assert recorder.calls == [("be helpful", ("echo",))]
    assert adapter.requests[0].system == "be helpful\n\nTOOLS: echo"
    assert [s.name for s in adapter.requests[0].tools] == ["echo"]


async def test_a_tool_registered_after_run_start_is_in_neither_prompt_nor_schemas() -> None:
    """Registered mid-run into the live registry (from a tool's own execute): the run's snapshot
    excludes it from the next request's prompt and schemas alike (`L08-R001`, extended)."""
    loop, adapter = _loop_with_adapter(_call("echo"), _done())
    recorder = _Recorder()
    _install(loop, recorder)

    def register_late(tool_call_id: str, args: dict[str, Any]) -> str:
        loop.tools.register(_tool("late"))
        return "ok"

    loop.tools.register(_tool("echo", register_late))
    await loop.prompt(_say("hi"))
    assert recorder.calls == [("be helpful", ("echo",)), ("be helpful", ("echo",))]
    assert [s.name for s in adapter.requests[1].tools] == ["echo"]


async def test_a_prepare_next_turn_replacement_drives_the_next_prompt_and_schemas() -> None:
    loop, adapter = _loop_with_adapter(_call("echo"), _done())
    loop.tools.register(_tool("echo"))
    other = _tool("other")
    recorder = _Recorder()
    _install(loop, recorder)

    async def replace(
        instance: Any, message: Any, tool_results: Any, context: RunContext, new: Any, next_: Any
    ) -> RunConfigUpdate:
        return RunConfigUpdate(
            context=RunContext(
                system_prompt="replaced base", messages=list(context.messages), tools=(other,)
            )
        )

    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, replace)
    await loop.prompt(_say("hi"))
    assert recorder.calls[1] == ("replaced base", ("other",))
    assert adapter.requests[1].system == "replaced base\n\nTOOLS: other"
    assert [s.name for s in adapter.requests[1].tools] == ["other"]


async def test_a_listener_reads_and_replaces_the_base_not_the_assembled_text() -> None:
    loop, _ = _loop_with_adapter(_call("echo"), _done())
    loop.tools.register(_tool("echo"))
    _install(loop, _Recorder())
    seen: list[str] = []

    async def observe(
        instance: Any, message: Any, tool_results: Any, context: RunContext, new: Any, next_: Any
    ) -> RunConfigUpdate:
        seen.append(context.system_prompt)
        result: RunConfigUpdate = await next_()
        return result

    loop.instance.ctx.events.on(AGENT_PREPARE_NEXT_TURN, observe)
    await loop.prompt(_say("hi"))
    assert seen[0] == "be helpful"
    assert loop.instance.system_prompt == "be helpful"


async def test_added_tool_names_growth_reaches_both_prompt_and_schemas() -> None:
    loop, adapter = _loop_with_adapter(_call("echo"), _done())
    recorder = _Recorder()
    _install(loop, recorder)

    def add_late(tool_call_id: str, args: dict[str, Any]) -> ToolResult:
        loop.tools.register(_tool("late"))
        return ToolResult(
            tool_call_id="",
            content=(TextBlock(text="ok"),),
            tool_name="echo",
            added_tool_names=("late",),
        )

    loop.tools.register(_tool("echo", add_late))
    await loop.prompt(_say("hi"))
    assert recorder.calls[1] == ("be helpful", ("echo", "late"))
    assert [s.name for s in adapter.requests[1].tools] == ["echo", "late"]


# ---- override ----


async def test_a_per_step_override_is_sent_verbatim_and_bypasses_the_assembler() -> None:
    loop, adapter = _loop_with_adapter(_done())
    loop.tools.register(_tool("echo"))
    recorder = _Recorder()
    _install(loop, recorder)

    async def override(
        instance: Any, reason: PreStepReason, messages: tuple[Any, ...], next_: Any
    ) -> Enter:
        return Enter(messages=messages, system_override="one-off")

    loop.instance.ctx.events.on(AGENT_PRE_STEP, override)
    await loop.prompt(_say("hi"))
    assert recorder.calls == []
    assert adapter.requests[0].system == "one-off"
    assert reconstruct_header(_headers(loop)[0], loop.artifacts) == {"system_base": "one-off"}


# ---- snapshot identity under concurrent registry churn ----


async def test_prompt_and_schemas_agree_on_every_request_under_concurrent_churn() -> None:
    """Another task registers and withdraws tools at every await point of a multi-turn run; each
    request's assembler tools and schemas are identical in membership and order."""
    responses = [_call("echo", f"t{i}") for i in range(6)] + [_done()]
    loop, adapter = _loop_with_adapter(*responses)
    seen: list[tuple[str, ...]] = []

    def assemble(base: str, tools: tuple[ToolDefinition, ...]) -> str:
        seen.append(tuple(t.name for t in tools))
        return base

    _install(loop, assemble)

    async def slow_echo(tool_call_id: str, args: dict[str, Any]) -> str:
        await asyncio.sleep(0)
        return "ok"

    loop.tools.register(_tool("echo", slow_echo))
    stop = asyncio.Event()

    async def churn() -> None:
        k = 0
        while not stop.is_set():
            withdraw = loop.tools.register(_tool(f"churn{k}"))
            await asyncio.sleep(0)
            withdraw()
            k += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(churn())
    try:
        await loop.prompt(_say("hi"))
    finally:
        stop.set()
        await task
    assert len(seen) == len(adapter.requests) == 7
    for names, request in zip(seen, adapter.requests, strict=True):
        assert names == tuple(s.name for s in request.tools)


# ---- failure: nothing published, the run settles as failed ----


async def test_a_raising_assembler_sends_nothing_and_settles_the_run_as_failed() -> None:
    loop, adapter = _loop_with_adapter(_done())

    def boom(base: str, tools: tuple[ToolDefinition, ...]) -> str:
        raise RuntimeError("assembler failed")

    _install(loop, boom)
    await loop.prompt(_say("hi"))
    assert adapter.requests == []
    assert _headers(loop) == []
    failures = _failed(loop)
    assert len(failures) == 1 and "assembler failed" in (failures[0].error_message or "")
    assert _agent_end_reasons(loop) == ["failed"]


async def test_a_non_string_result_is_a_failure_too() -> None:
    loop, adapter = _loop_with_adapter(_done())
    _install(loop, lambda base, tools: None)
    await loop.prompt(_say("hi"))
    assert adapter.requests == [] and _headers(loop) == []
    failures = _failed(loop)
    assert len(failures) == 1 and "not a string" in (failures[0].error_message or "")
    assert _agent_end_reasons(loop) == ["failed"]


async def test_a_later_turn_failure_keeps_the_earlier_request_and_header() -> None:
    loop, adapter = _loop_with_adapter(_call("echo"), _done())
    loop.tools.register(_tool("echo"))
    calls = 0

    def second_fails(base: str, tools: tuple[ToolDefinition, ...]) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second request")
        return f"{base}!"

    _install(loop, second_fails)
    await loop.prompt(_say("hi"))
    assert [r.system for r in adapter.requests] == ["be helpful!"]
    headers = _headers(loop)
    assert len(headers) == 1
    assert reconstruct_header(headers[0], loop.artifacts) == {"system_base": "be helpful!"}
    assert _agent_end_reasons(loop) == ["failed"]
    assert len(_failed(loop)) == 1


# ---- header ----


async def test_the_header_records_the_assembled_text_and_reconstructs_it() -> None:
    loop, adapter = _loop_with_adapter(_done())
    loop.tools.register(_tool("echo"))
    _install(loop, _Recorder())
    await loop.prompt(_say("hi"))
    rebuilt = reconstruct_header(_headers(loop)[0], loop.artifacts)
    assert rebuilt == {"system_base": adapter.requests[0].system}
    assert adapter.requests[0].system == "be helpful\n\nTOOLS: echo"


def test_the_factory_installs_the_assembler_or_leaves_it_absent() -> None:
    from types import SimpleNamespace

    from minion_agent.agent_loop import AgentLoopFactory

    loop, _ = _loop_with_adapter(_done())
    ctx = SimpleNamespace(
        llm=loop.llm,
        tools=loop.tools,
        sessions=SimpleNamespace(artifacts=loop.artifacts),
        registry=SimpleNamespace(has=lambda name: False),
    )
    factory = AgentLoopFactory(ctx)  # type: ignore[arg-type]
    recorder = _Recorder()
    assert (
        factory.for_instance(loop.instance, prompt_assembler=recorder).prompt_assembler is recorder
    )
    assert factory.for_instance(loop.instance).prompt_assembler is None
