"""tools/post-execute transforms results, in registration order.

`register_after_tool_call_hook` is the only sanctioned registration path (`L06-R003`/`L06-R006`):
a hook receives the current, already-merged `ToolResult` and may return an `AfterToolCallOverride`
(or `None`) -- never the whole result. This is what makes replacing `tool_call_id`/`tool_name`/
`added_tool_names` structurally impossible through the public API, unlike an earlier, uncertified
revision that let a listener return/replace the entire `ToolResult` directly.
"""

from typing import Any
from unittest.mock import patch

from minion_agent.llm import TextBlock, text_of
from minion_agent.runtime import Context, EventBus
from minion_agent.tools.decisions import AfterToolCallOverride
from minion_agent.tools.events import TOOLS_POST_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call, register_after_tool_call_hook
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


async def test_a_lone_hook_transformation_survives() -> None:
    """The case a fixed terminal loses. One hook overrides and the next-listener merge applies
    it, so the terminal is what returns the transformed value."""
    ctx = _ctx()

    def audit(result: ToolResult) -> AfterToolCallOverride:
        return AfterToolCallOverride(details={**result.details, "audited": True})

    register_after_tool_call_hook(ctx, audit)

    outcome = await execute_call(_call(value="x"), registry=_registry(_echo()), ctx=ctx)

    assert outcome.details == {"audited": True}


async def test_a_hook_that_returns_none_leaves_the_result_unchanged() -> None:
    """An after-hook that abstains (returns `None`, matching pinned Pi's own hook being
    optional/absent) makes no change at all -- distinct from returning an override with every
    field left at its `None` default, though both are observably identical."""
    ctx = _ctx()

    def abstain(result: ToolResult) -> None:
        return None

    register_after_tool_call_hook(ctx, abstain)

    outcome = await execute_call(_call(value="unchanged"), registry=_registry(_echo()), ctx=ctx)

    assert text_of(outcome.to_message()) == "unchanged"
    assert outcome.details == {}


async def test_registration_order_equals_application_order() -> None:
    ctx = _ctx()

    def tag(label: str) -> Any:
        def hook(result: ToolResult) -> AfterToolCallOverride:
            return AfterToolCallOverride(
                content=(TextBlock(text=f"{text_of(result.to_message())}-{label}"),)
            )

        return hook

    register_after_tool_call_hook(ctx, tag("first"))
    register_after_tool_call_hook(ctx, tag("second"))

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

    def only_details(result: ToolResult) -> AfterToolCallOverride:
        return AfterToolCallOverride(details={"seen": True})

    register_after_tool_call_hook(ctx, only_details)

    outcome = await execute_call(_call(value="kept"), registry=_registry(_echo()), ctx=ctx)

    assert outcome.details == {"seen": True}
    assert text_of(outcome.to_message()) == "kept"
    assert not outcome.is_error


async def test_there_is_no_deep_merge() -> None:
    """A hook that supplies `details` replaces it wholesale. Deep merging
    would make it impossible to remove a key, and pi does not do it."""
    ctx = _ctx()
    definition = _echo(
        execute=lambda tool_call_id, args: ToolResult(
            tool_call_id="t1", content=(), tool_name="echo", details={"original": 1, "keep": 2}
        )
    )

    def overwrite(result: ToolResult) -> AfterToolCallOverride:
        return AfterToolCallOverride(details={"replacement": 3})

    register_after_tool_call_hook(ctx, overwrite)

    outcome = await execute_call(_call(value="x"), registry=_registry(definition), ctx=ctx)

    assert outcome.details == {"replacement": 3}


async def test_a_hook_cannot_override_execution_identity_or_added_tool_names() -> None:
    """`L06-R003`: pinned Pi's `AfterToolCallResult` has no `tool_call_id`/`tool_name`/
    `added_tool_names` slot at all -- `AfterToolCallOverride` structurally cannot carry them, so
    no hook can replace them, even one that owns every other field. An earlier, uncertified
    revision let a listener return/replace the entire `ToolResult` (proven by the now-removed
    `test_a_listener_may_own_the_result_outright`), which observably could rewrite these; this
    test pins that it no longer can."""
    ctx = _ctx()
    definition = _echo(
        execute=lambda tool_call_id, args: ToolResult(
            tool_call_id="t1",
            content=(TextBlock(text="secret"),),
            tool_name="echo",
            added_tool_names=("alpha",),
        )
    )

    def redact(result: ToolResult) -> AfterToolCallOverride:
        return AfterToolCallOverride(content=(TextBlock(text="redacted"),))

    register_after_tool_call_hook(ctx, redact)
    register_after_tool_call_hook(ctx, redact)

    call = _call(value="secret")
    outcome = await execute_call(call, registry=_registry(definition), ctx=ctx)

    assert text_of(outcome.to_message()) == "redacted"
    assert outcome.tool_call_id == call.id
    assert outcome.tool_name == "echo"
    assert outcome.added_tool_names == ("alpha",)


async def test_an_execute_failure_is_transformed_too() -> None:
    """An outcome that reached `execute()` -- success or failure -- always goes through the
    after-hook (pinned Pi's `finalizeExecutedToolCall` runs uniformly for both; `TOOL-017`).
    Failures are the results most worth annotating."""
    ctx = _ctx()

    def broken(tool_call_id: str, args: dict[str, Any]) -> str:
        raise RuntimeError("boom")

    def annotate(result: ToolResult) -> AfterToolCallOverride:
        return AfterToolCallOverride(details={"failed": result.is_error})

    register_after_tool_call_hook(ctx, annotate)

    outcome = await execute_call(
        _call(value="x"), registry=_registry(_echo(execute=broken)), ctx=ctx
    )

    assert outcome.details == {"failed": True}


async def test_an_unknown_tool_never_reaches_the_after_hook() -> None:
    """Pinned Pi's `finalizeExecutedToolCall` (the after-hook) is invoked only for an outcome
    that actually reached `execute()` -- an unknown-tool lookup never does (`TOOL-017`;
    previously this pipeline ran the after-hook uniformly on every outcome, including this one,
    a genuine `PI_PARITY_DEFECT` this test now pins the fix for)."""
    ctx = _ctx()
    dispatched: list[str] = []

    def annotate(result: ToolResult) -> AfterToolCallOverride:
        dispatched.append("ran")
        return AfterToolCallOverride(details={"failed": result.is_error})

    register_after_tool_call_hook(ctx, annotate)

    outcome = await execute_call(_call("missing"), registry=_registry(), ctx=ctx)

    assert dispatched == []
    assert outcome.details == {}
    assert outcome.is_error


async def test_a_raising_after_hook_replaces_the_entire_prior_result() -> None:
    """Pinned Pi's `finalizeExecutedToolCall`: an after-hook exception replaces the ENTIRE prior
    result -- content, details, usage, terminate all discarded, not merged -- with a plain error
    result. `L06-R002`: the message is the hook's own, with no Python exception-class prefix."""
    ctx = _ctx()
    definition = _echo(
        execute=lambda tool_call_id, args: ToolResult(
            tool_call_id="t1",
            content=(TextBlock(text="success"),),
            tool_name="echo",
            details={"had": "data"},
            terminate=True,
        )
    )

    def exploding(result: ToolResult) -> AfterToolCallOverride:
        raise RuntimeError("annotation service down")

    register_after_tool_call_hook(ctx, exploding)

    outcome = await execute_call(_call(value="x"), registry=_registry(definition), ctx=ctx)

    assert outcome.is_error
    assert text_of(outcome.to_message()) == "annotation service down"
    assert outcome.details == {}
    assert not outcome.terminate
