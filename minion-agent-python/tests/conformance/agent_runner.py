"""Executes agent conformance scenarios against the composed stack.

Asserts the two observable surfaces section 8 names: the derived Pi event
stream and the log's message projection. Nothing here inspects loop internals.

One adapter is built directly from the scenario rather than through
`mock_adapter_plugin`. The plugin's config model carries text only, and a
scenario needs tool calls, truncation, and post-terminal chunks; registering
two adapters for one model -- as an earlier draft did -- would leave which of
them answers a request dependent on registration order.
"""

from __future__ import annotations

import asyncio
from typing import Any

from minion_agent.agent.envelope import ClaimPolicy
from minion_agent.agent.identity import AgentDefinition
from minion_agent.agent.plugin import agents_plugin
from minion_agent.agent.projection import AgentEnd, event_names, project
from minion_agent.agent_loop import agent_loop_plugin
from minion_agent.llm import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    ContentBlock,
    Cost,
    DeferredHandle,
    DiagnosticError,
    ModelId,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    text_of,
)
from minion_agent.llm.adapters.mock import MockAdapter, ScriptedResponse
from minion_agent.llm.messages import StopReason, Usage
from minion_agent.llm.plugin import llm_plugin
from minion_agent.runtime import Context
from minion_agent.session import EventKind, derive_messages
from minion_agent.session.service import session_plugin
from minion_agent.tools.decisions import AfterToolCallOverride, Block, Proceed
from minion_agent.tools.definition import ExecutionMode, ToolDefinition
from minion_agent.tools.events import (
    TOOLS_EXECUTION_END,
    TOOLS_EXECUTION_START,
    TOOLS_POST_EXECUTE,
    TOOLS_PRE_EXECUTE,
    TOOLS_UPDATE,
)
from minion_agent.tools.execute import register_after_tool_call_hook
from minion_agent.tools.plugin import tools_plugin
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolPartialResult, ToolResult

_ROLE = {
    UserMessage: "user",
    AssistantMessage: "assistant",
    ToolResultMessage: "tool_result",
}


def _block(raw: dict[str, Any]) -> ContentBlock:
    if raw["type"] == "tool_call":
        return ToolCallBlock(
            id=raw["id"],
            name=raw["name"],
            arguments=raw.get("arguments", {}),
            thought_signature=raw.get("thought_signature"),
            namespace=raw.get("namespace"),
        )
    if raw["type"] == "thinking":
        return ThinkingBlock(
            thinking=raw.get("thinking", ""),
            thinking_signature=raw.get("thinking_signature"),
            redacted=raw.get("redacted", False),
        )
    return TextBlock(text=raw.get("text", ""), text_signature=raw.get("text_signature"))


def _usage(raw: dict[str, Any] | None) -> Usage:
    if raw is None:
        return Usage()
    cost = raw.get("cost")
    return Usage(
        input=raw.get("input", 0),
        output=raw.get("output", 0),
        cache_read=raw.get("cache_read", 0),
        cache_write=raw.get("cache_write", 0),
        cache_write_1h=raw.get("cache_write_1h"),
        reasoning=raw.get("reasoning"),
        total_tokens=raw.get("total_tokens", 0),
        cost=Cost(**cost) if cost is not None else Cost(),
    )


def _diagnostic(raw: dict[str, Any]) -> AssistantMessageDiagnostic:
    raw_error = raw.get("error")
    return AssistantMessageDiagnostic(
        type=raw["type"],
        timestamp=raw["timestamp"],
        error=DiagnosticError(**raw_error) if raw_error is not None else None,
        details=raw.get("details"),
    )


def _deferred(raw: dict[str, Any] | None) -> DeferredHandle | None:
    if raw is None:
        return None
    return DeferredHandle(
        provider=raw["provider"],
        model_id=raw["model_id"],
        api=raw["api"],
        id=raw["id"],
        expires_at=raw.get("expires_at"),
        poll_after_ms=raw.get("poll_after_ms"),
        data=raw.get("data"),
    )


def _script(document: dict[str, Any]) -> list[ScriptedResponse]:
    """Translate the scripted provider responses into adapter form."""
    return [
        ScriptedResponse(
            content=tuple(_block(raw) for raw in response.get("content", [])),
            stop_reason=StopReason(response["stop_reason"]),
            usage=_usage(response.get("usage")),
            error_message=response.get("error_message"),
            truncated=response.get("truncated", False),
            chunks_after_terminal=response.get("chunks_after_terminal", 0),
            response_model=response.get("response_model"),
            response_id=response.get("response_id"),
            diagnostics=(
                tuple(_diagnostic(raw) for raw in response["diagnostics"])
                if "diagnostics" in response
                else None
            ),
            deferred=_deferred(response.get("deferred")),
            raw_stop_reason=response.get("raw_stop_reason"),
            end_turn=response.get("end_turn"),
        )
        for response in document["provider_script"]
    ]


def _parameters(spec: dict[str, Any] | None) -> dict[str, Any]:
    """The tool's parameter schema: the actual Layer-05 shared `ToolDefinition.parameters`
    representation -- a plain, object-valued JSON Schema mapping -- passed straight through, not
    a Python-specific shorthand (`L06-R001`; an earlier revision built a Pydantic model from a
    `requires: [...]` shorthand instead, exercising a Python-only validation path rather than the
    approved cross-language schema boundary). Omitted defaults to the explicit empty-object
    schema, matching Layer 05's own no-arguments convention."""
    if spec is None:
        return {"type": "object", "properties": {}}
    return spec


def _prepare_arguments(spec: dict[str, Any] | None) -> Any:
    """A declarative `prepare_arguments` shim: `raises` throws, `set` overwrites/adds keys."""
    if spec is None:
        return None
    raises = spec.get("raises")
    set_fields = spec.get("set", {})

    def prepare(args: dict[str, Any]) -> dict[str, Any]:
        if raises:
            raise RuntimeError(raises)
        return {**args, **set_fields}

    return prepare


def _partial_result(spec: dict[str, Any]) -> ToolPartialResult:
    """A structured partial-output value (`L08-R011`): pinned Pi's own `AgentToolUpdateCallback<T>`
    carries `partialResult: AgentToolResult<T>` -- `content`/`details`/`added_tool_names`/
    `terminate`, with NO nested call identity or error field of its own (an independent Rust
    re-review caught an earlier revision reusing the pipeline-level `ToolResult` here instead,
    observably larger than Pi's own type). `details`/`terminate`/`added_tool_names` are read only
    when the scenario spec actually sets them, left `None` otherwise -- preserving the SAME
    explicit-vs-absent distinction pinned Pi's own optional (`?`) fields carry, not collapsed into
    a concrete default a scenario cannot tell apart from "never set"."""
    added = spec.get("added_tool_names")
    return ToolPartialResult(
        content=(TextBlock(text=spec.get("text", "")),),
        details=spec.get("details", {}),
        added_tool_names=tuple(added) if added is not None else None,
        terminate=spec.get("terminate"),
    )


def _encode_partial(partial: ToolPartialResult) -> dict[str, Any]:
    """Encode a captured `ToolPartialResult` back to the scenario's own structured shorthand
    (`L08-R011`) for plain, language-neutral YAML comparison. `text` is always present (joined
    `TextBlock` content, matching pinned Pi's own `content` field); `details`/`terminate`/
    `added_tool_names` are included ONLY when the tool actually set them (not `None`) -- an
    omitted key in the observed dict means "never set", exactly distinguishing that from an
    explicit falsy/empty value the same way pinned Pi's own optional (`?`) fields do, so a
    scenario asserting the full shape is genuinely discriminating, not merely text-shorthand."""
    encoded: dict[str, Any] = {
        "text": "".join(b.text for b in partial.content if isinstance(b, TextBlock))
    }
    if partial.details:
        encoded["details"] = partial.details
    if partial.terminate is not None:
        encoded["terminate"] = partial.terminate
    if partial.added_tool_names is not None:
        encoded["added_tool_names"] = list(partial.added_tool_names)
    return encoded


def _stub(
    spec: dict[str, Any],
    registry: ToolRegistry,
    name: str,
    late_updates: list[asyncio.Task[None]],
    trace: list[list[str]],
) -> Any:
    """Build a tool body from a declarative stub.

    `late_updates` collects fire-and-forget tasks that call `update()` one cooperative scheduler
    yield after the tool itself has already returned -- deterministic (no wall-clock timing), and
    `run_agent_scenario` drains it after the scenario's own steps finish, so a "late update
    ignored" assertion is never a race.

    `trace` records an `["execute", tool_call_id]` entry the moment this body starts running --
    the `IR-L06-001` execution-trace observation's own record of when a prepared call's `execute()`
    actually began, independent of anything the runner itself decides.
    """
    text = spec.get("result", {}).get("text", "")
    ticks = spec.get("delay_ticks", 0)
    raises = spec.get("raises")
    terminate = spec.get("terminate", False)
    adds = tuple(spec.get("adds_tools", ()))
    updates = tuple(spec.get("emits_updates", ()))
    late_update = spec.get("late_update")

    async def run(tool_call_id: str, args: dict[str, Any], update: Any) -> ToolResult:
        trace.append(["execute", tool_call_id])
        for _ in range(ticks):
            # A scheduler yield, not wall-clock time: deterministic, and
            # reproducible in any language with a cooperative scheduler.
            await asyncio.sleep(0)
        if raises:
            raise RuntimeError(raises)
        for partial in updates:
            update(_partial_result(partial))
        if late_update is not None:

            async def _fire_late(partial: dict[str, Any] = late_update) -> None:
                await asyncio.sleep(0)
                update(_partial_result(partial))

            late_updates.append(asyncio.ensure_future(_fire_late()))
        for added_name in adds:
            registry.register(
                ToolDefinition(
                    name=added_name,
                    description=added_name,
                    parameters={"type": "object", "properties": {}},
                    execute=lambda inner_id, inner: "added",
                    label=added_name,
                )
            )
        return ToolResult(
            tool_call_id="",
            content=(TextBlock(text=text),),
            tool_name=name,
            terminate=terminate,
            added_tool_names=adds,
        )

    return run


def _normalize_block(block: ContentBlock) -> dict[str, Any]:
    """Read a real content-block object's actual fields -- never invent one
    the object doesn't carry."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text, "text_signature": block.text_signature}
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "thinking_signature": block.thinking_signature,
            "redacted": block.redacted,
        }
    if isinstance(block, ToolCallBlock):
        return {
            "type": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments": block.arguments,
            "thought_signature": block.thought_signature,
            "namespace": block.namespace,
        }
    return {"type": "image"}


def _normalize_usage(usage: Usage) -> dict[str, Any]:
    return {
        "input": usage.input,
        "output": usage.output,
        "cache_read": usage.cache_read,
        "cache_write": usage.cache_write,
        "cache_write_1h": usage.cache_write_1h,
        "reasoning": usage.reasoning,
        "total_tokens": usage.total_tokens,
        "cost": {
            "input": usage.cost.input,
            "output": usage.cost.output,
            "cache_read": usage.cost.cache_read,
            "cache_write": usage.cost.cache_write,
            "total": usage.cost.total,
        },
    }


def _normalize_diagnostic(diagnostic: AssistantMessageDiagnostic) -> dict[str, Any]:
    error = diagnostic.error
    normalized_error = (
        None
        if error is None
        else {
            "message": error.message,
            "name": error.name,
            "stack": error.stack,
            "code": error.code,
        }
    )
    return {
        "type": diagnostic.type,
        "timestamp": diagnostic.timestamp,
        "error": normalized_error,
        "details": diagnostic.details,
    }


def _normalize_deferred(handle: DeferredHandle | None) -> dict[str, Any] | None:
    if handle is None:
        return None
    return {
        "provider": handle.provider,
        "model_id": handle.model_id,
        "api": handle.api,
        "id": handle.id,
        "expires_at": handle.expires_at,
        "poll_after_ms": handle.poll_after_ms,
        "data": handle.data,
    }


def _assistant_detail(message: AssistantMessage) -> dict[str, Any]:
    """Normalize a real AssistantMessage the implementation actually
    produced into the canonical dict shape (LLM-F010). Reads every field off
    the object directly; synthesizes nothing the object doesn't carry."""
    return {
        "api": message.api,
        "provider": message.provider,
        "model": message.model,
        "timestamp": message.timestamp,
        "response_model": message.response_model,
        "response_id": message.response_id,
        "stop_reason": message.stop_reason.value,
        "raw_stop_reason": message.raw_stop_reason,
        "end_turn": message.end_turn,
        "error_message": message.error_message,
        "usage": _normalize_usage(message.usage),
        "diagnostics": (
            None
            if message.diagnostics is None
            else [_normalize_diagnostic(d) for d in message.diagnostics]
        ),
        "deferred": _normalize_deferred(message.deferred),
        "content": [_normalize_block(block) for block in message.content],
    }


def _listener(spec: dict[str, Any]) -> Any:
    """Build a declarative listener for the tools pipeline.

    Both branches are total: an action not recognised for the dispatched
    event raises rather than silently delegating. A listener that falls
    through to `next_()` for an unhandled action is a listener that would
    exercise nothing while the scenario still reports green -- exactly the
    "test that cannot fail" failure mode, raised to the vocabulary level.
    """
    action = spec["action"]
    event = spec["event"]
    only = spec.get("only_tool")

    if event == TOOLS_PRE_EXECUTE:

        async def pre(call: Any, definition: Any, arguments: Any, next_: Any) -> Any:
            if only is not None and call.name != only:
                return await next_()
            if action == "block":
                return Block(
                    reason=spec.get("reason", "blocked"),
                    terminate=spec.get("terminate", False),
                )
            if action == "narrow_arguments":
                return Proceed(arguments=spec.get("arguments", {}))
            if action == "abstain":
                return await next_()
            if action == "raise":
                raise RuntimeError(spec.get("message", "before-hook failed"))
            raise ValueError(f"unhandled listener action {action!r} for event {event!r}")

        return pre

    if event == TOOLS_POST_EXECUTE:
        # A hook of this shape must be registered via register_after_tool_call_hook, not
        # ctx.events.on directly: it returns an AfterToolCallOverride (or None), never the whole
        # ToolResult, which is what makes replacing execution identity/added_tool_names
        # structurally impossible through the sanctioned API (L06-R003/L06-R006).
        def post(result: ToolResult) -> AfterToolCallOverride | None:
            if action == "annotate_result":
                label = spec.get("label", "seen")
                return AfterToolCallOverride(
                    content=(TextBlock(text=f"{text_of(result.to_message())}-{label}"),)
                )
            if action == "abstain":
                return None
            if action == "raise":
                raise RuntimeError(spec.get("message", "after-hook failed"))
            raise ValueError(f"unhandled listener action {action!r} for event {event!r}")

        return post

    raise ValueError(f"unhandled listener event {event!r}")


async def run_agent_scenario(document: dict[str, Any]) -> dict[str, Any]:
    """Run the scenario and return its observable surfaces."""
    ctx = Context()
    await ctx.plugin(session_plugin)
    await ctx.plugin(llm_plugin)
    await ctx.plugin(tools_plugin)
    await ctx.plugin(agents_plugin)
    await ctx.plugin(agent_loop_plugin)
    adapter = MockAdapter(_script(document))
    ctx.llm.register(adapter)

    late_updates: list[asyncio.Task[None]] = []
    trace: list[list[str]] = []
    for name, stub in document.get("tools", {}).items():
        ctx.tools.register(
            ToolDefinition(
                name=name,
                description=name,
                parameters=_parameters(stub.get("parameters")),
                execute=_stub(stub, ctx.tools, name, late_updates, trace),
                prepare_arguments=_prepare_arguments(stub.get("prepare_arguments")),
                label=name,
                mode=ExecutionMode(stub.get("execution_mode", "parallel")),
            )
        )

    # IR-L06-001 execution-trace observation: start/end are unconditional, EMIT-mode listeners --
    # trivial. `before` must be a waterfall listener that wraps the ENTIRE tools/pre-execute chain
    # (prepended, so it always runs first and delegates to whatever the scenario itself
    # registered) and records its marker only AFTER `next_()` settles, so it fires exactly once a
    # call's before-hook stage has fully resolved (Proceed or Block), regardless of registered
    # listeners -- never simulating or reordering that resolution, only observing it.
    ctx.events.on(
        TOOLS_EXECUTION_START, lambda call_id, name, args: trace.append(["start", call_id])
    )
    ctx.events.on(TOOLS_EXECUTION_END, lambda call_id, name, result: trace.append(["end", call_id]))

    async def _trace_before(call: Any, definition: Any, arguments: Any, next_: Any) -> Any:
        decision = await next_()
        trace.append(["before", call.id])
        return decision

    ctx.events.on(TOOLS_PRE_EXECUTE, _trace_before, prepend=True)

    for spec in document.get("listeners", []):
        if spec["event"] == TOOLS_POST_EXECUTE:
            register_after_tool_call_hook(ctx, _listener(spec))
        else:
            ctx.events.on(spec["event"], _listener(spec))

    seen_updates: list[dict[str, Any]] = []
    ctx.events.on(
        TOOLS_UPDATE,
        lambda call_id, tool_name, arguments, partial: seen_updates.append(
            {
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "partial": _encode_partial(partial),
            }
        ),
    )

    config = document.get("config", {})
    handle = ctx.agents.create(
        "scenario",
        AgentDefinition(
            name="scenario",
            model=ModelId("mock", config.get("model", "mock-1")),
            system=config.get("system", ""),
        ),
    )
    loop = ctx.agent_loop.for_instance(handle.instance)
    if "next_turn_policy" in config:
        loop.next_turn_policy = ClaimPolicy(config["next_turn_policy"])

    error: Exception | None = None
    for step in document["steps"]:
        for alias in ("followup", "steer", "inject"):
            if alias in step:
                spec = step[alias]
                text = spec if isinstance(spec, str) else spec["text"]
                origin = None if isinstance(spec, str) else spec.get("origin")
                message = UserMessage(content=(TextBlock(text=text),), timestamp=1)
                getattr(handle.instance.inbox, alias)(message, origin=origin)
        if step.get("await_idle"):
            try:
                await loop.run_until_idle()
            except Exception as raised:
                error = raised
                break
        if step.get("continue"):
            # Layer 08, PASS 2: pinned pi's `Agent.continue()` (renamed --
            # `continue` is a Python keyword). A thin call-through, same as
            # `await_idle` above; the runner picks no branch of `continue_()`'s
            # own logic itself.
            try:
                await loop.continue_()
            except Exception as raised:
                error = raised
                break

    if late_updates:
        # Drain deterministically before observing the final state, so a "late update
        # ignored" assertion is never a race against a still-pending background task.
        await asyncio.gather(*late_updates)

    events = project(handle.instance.log)
    messages = derive_messages(handle.instance.log)
    return {
        "events": event_names(events),
        "messages": [
            {
                "role": _ROLE[type(m)],
                "text": text_of(m),
                # IR-L06-004: exposed only for ToolResultMessage, the shape the finding is about --
                # never invented for a message type that carries no such field.
                **({"details": m.details} if isinstance(m, ToolResultMessage) else {}),
            }
            for m in messages
        ],
        "causes": [list(event.causes) for event in events if isinstance(event, AgentEnd)],
        "agent_end_messages": [
            [text_of(m) for m in event.messages] for event in events if isinstance(event, AgentEnd)
        ],
        "assistant_stop_reasons": [
            m.stop_reason.value for m in messages if isinstance(m, AssistantMessage)
        ],
        "assistant_details": [
            _assistant_detail(m) for m in messages if isinstance(m, AssistantMessage)
        ],
        "tool_completion_order": [
            entry.data["message"]["tool_call_id"]
            for entry in sorted(
                (e for e in handle.instance.log.events if e.kind == EventKind.TOOL_RESULT),
                key=lambda e: e.data["completion_index"],
            )
        ],
        "request_tools": [[tool.name for tool in request.tools] for request in adapter.requests],
        "updates": seen_updates,
        "tool_trace": trace,
        "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
    }
