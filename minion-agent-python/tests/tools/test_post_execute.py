"""tools/post-execute transforms results, in registration order."""

from dataclasses import replace
from typing import Any
from unittest.mock import patch

from minion_agent.llm import TextBlock, text_of
from minion_agent.runtime import Context, EventBus
from minion_agent.tools.events import TOOLS_POST_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.result import ToolResult

from .test_execute import _call, _echo, _registry


def _ctx() -> Context:
    ctx = Context()
    declare_tools_events(ctx.events)
    return ctx


async def test_no_listener_returns_the_result_unchanged() -> None:
    """Strengthened: the brief's version can pass even if `_finalize` (and thus
    the `tools/post-execute` dispatch) were never wired into `execute_call` at
    all, since an empty chain and a skipped chain look identical from the
    outside when nothing is registered. A spy on `EventBus.waterfall` proves
    the dispatch actually happens."""
    ctx = _ctx()
    dispatched: list[str] = []
    original = EventBus.waterfall

    async def spy(self: EventBus, name: str, *args: Any, **kwargs: Any) -> Any:
        dispatched.append(name)
        return await original(self, name, *args, **kwargs)

    with patch.object(EventBus, "waterfall", spy):
        result = await execute_call(_call(value="plain"), registry=_registry(_echo()), ctx=ctx)

    assert text_of(result.to_message()) == "plain"
    assert TOOLS_POST_EXECUTE in dispatched


async def test_a_lone_listener_transformation_survives() -> None:
    """The case a fixed terminal loses. One listener transforms and delegates;
    nothing follows it, so the terminal is what returns the transformed value."""
    ctx = _ctx()

    async def audit(result: ToolResult, next_: Any) -> Any:
        return await next_(replace(result, details={**result.details, "audited": True}))

    ctx.events.on(TOOLS_POST_EXECUTE, audit)

    outcome = await execute_call(_call(value="x"), registry=_registry(_echo()), ctx=ctx)

    assert outcome.details == {"audited": True}


async def test_registration_order_equals_application_order() -> None:
    ctx = _ctx()

    def tag(label: str) -> Any:
        async def listener(result: ToolResult, next_: Any) -> Any:
            marked = replace(
                result, content=(TextBlock(text=f"{text_of(result.to_message())}-{label}"),)
            )
            return await next_(marked)

        return listener

    ctx.events.on(TOOLS_POST_EXECUTE, tag("first"))
    ctx.events.on(TOOLS_POST_EXECUTE, tag("second"))

    outcome = await execute_call(_call(value="base"), registry=_registry(_echo()), ctx=ctx)

    assert text_of(outcome.to_message()) == "base-first-second"


async def test_omitted_fields_are_unchanged() -> None:
    """Pi's afterToolCall merge: supplied fields replace, omitted fields stay.

    Strengthened: the brief's version never inspects `outcome.details`, so it
    could not distinguish the listener actually running (and replacing
    `details`) from the listener never running at all -- both leave `content`
    and `is_error` alone. Asserting `details` proves the listener ran while
    the other two assertions prove the fields it omitted were untouched.
    """
    ctx = _ctx()

    async def only_details(result: ToolResult, next_: Any) -> Any:
        return await next_(replace(result, details={"seen": True}))

    ctx.events.on(TOOLS_POST_EXECUTE, only_details)

    outcome = await execute_call(_call(value="kept"), registry=_registry(_echo()), ctx=ctx)

    assert outcome.details == {"seen": True}
    assert text_of(outcome.to_message()) == "kept"
    assert not outcome.is_error


async def test_there_is_no_deep_merge() -> None:
    """A listener that supplies `details` replaces it wholesale. Deep merging
    would make it impossible to remove a key, and pi does not do it."""
    ctx = _ctx()
    definition = _echo(
        execute=lambda args: ToolResult(
            tool_call_id="t1", content=(), details={"original": 1, "keep": 2}
        )
    )

    async def overwrite(result: ToolResult, next_: Any) -> Any:
        return await next_(replace(result, details={"replacement": 3}))

    ctx.events.on(TOOLS_POST_EXECUTE, overwrite)

    outcome = await execute_call(_call(value="x"), registry=_registry(definition), ctx=ctx)

    assert outcome.details == {"replacement": 3}


async def test_a_listener_may_own_the_result_outright() -> None:
    ctx = _ctx()

    async def replace_all(result: ToolResult, next_: Any) -> ToolResult:
        return replace(result, content=(TextBlock(text="redacted"),))

    ctx.events.on(TOOLS_POST_EXECUTE, replace_all)
    ctx.events.on(TOOLS_POST_EXECUTE, replace_all)

    outcome = await execute_call(_call(value="secret"), registry=_registry(_echo()), ctx=ctx)

    assert text_of(outcome.to_message()) == "redacted"


async def test_an_error_result_is_transformed_too() -> None:
    """Failures are the results most worth annotating."""
    ctx = _ctx()

    async def annotate(result: ToolResult, next_: Any) -> Any:
        return await next_(replace(result, details={"failed": result.is_error}))

    ctx.events.on(TOOLS_POST_EXECUTE, annotate)

    outcome = await execute_call(_call("missing"), registry=_registry(), ctx=ctx)

    assert outcome.details == {"failed": True}
