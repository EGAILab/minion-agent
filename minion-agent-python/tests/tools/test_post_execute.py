"""tools/post-execute transforms results, in registration order.

`register_after_tool_call_hook` is the recommended registration path (`L06-R003`/`L06-R006`): a
hook receives the current, already-merged `ToolResult` and may return an `AfterToolCallOverride`
(or `None`) -- never the whole result, so a hook written against it cannot even attempt to touch
`tool_call_id`/`tool_name`/`added_tool_names`. But `tools/post-execute` remains a public Runtime
event, so a caller may also register a raw listener directly via `ctx.events.on(TOOLS_POST_EXECUTE,
...)`. The actual authoritative boundary is `execute.py::_finalize`'s own unconditional restoration
of those three fields after every dispatch, regardless of registration path -- proven directly in
this file by registering RAW listeners, not only the helper, and confirming identity/
`added_tool_names` survive even an explicit whole-result-replacement attempt.
"""

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import patch

import pytest

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


async def test_a_raw_event_listener_cannot_replace_execution_identity() -> None:
    """`L06-R003`: the authoritative boundary is `_finalize`, not the registration path. A raw
    listener registered directly via `ctx.events.on(TOOLS_POST_EXECUTE, ...)` -- bypassing
    `register_after_tool_call_hook` entirely -- returns a whole `ToolResult` with a different
    `tool_call_id`, `tool_name`, and `added_tool_names`. None of that survives: pinned Pi's
    `AfterToolCallResult` gives a hook no way to touch these fields, and that holds regardless of
    how the hook was registered, not merely for hooks written against the constrained helper."""
    ctx = _ctx()
    definition = _echo(
        execute=lambda tool_call_id, args: ToolResult(
            tool_call_id="t1",
            content=(TextBlock(text="secret"),),
            tool_name="echo",
            added_tool_names=("alpha",),
        )
    )

    async def raw_whole_result_listener(result: ToolResult, next_: Any) -> ToolResult:
        return ToolResult(
            tool_call_id="rewritten",
            content=(TextBlock(text="redacted"),),
            tool_name="rewritten-name",
            added_tool_names=("injected",),
        )

    ctx.events.on(TOOLS_POST_EXECUTE, raw_whole_result_listener)

    call = _call(value="secret")
    outcome = await execute_call(call, registry=_registry(definition), ctx=ctx)

    # The allowed field (content) the raw listener changed is still applied --
    # this closure does not disable direct registration, only its identity authority.
    assert text_of(outcome.to_message()) == "redacted"
    assert outcome.tool_call_id == call.id
    assert outcome.tool_name == "echo"
    assert outcome.added_tool_names == ("alpha",)


async def test_a_raw_event_listener_may_still_change_allowed_fields() -> None:
    """The positive counterpart: direct registration is not disabled, only its identity/
    `added_tool_names` authority. `content` and `terminate` -- Pi-allowed fields -- still change."""
    ctx = _ctx()

    async def raw_listener(result: ToolResult, next_: Any) -> ToolResult:
        from dataclasses import replace

        return replace(result, content=(TextBlock(text="changed"),), terminate=True)

    ctx.events.on(TOOLS_POST_EXECUTE, raw_listener)

    outcome = await execute_call(_call(value="x"), registry=_registry(_echo()), ctx=ctx)

    assert text_of(outcome.to_message()) == "changed"
    assert outcome.terminate


async def test_in_place_mutation_of_the_result_is_structurally_impossible() -> None:
    """`ToolResult` is a frozen dataclass: a listener holding the object cannot mutate
    `tool_call_id`/`tool_name`/`added_tool_names` (or any field) in place at all -- an attempt
    raises `FrozenInstanceError` immediately, which the after-hook failure path (already
    established) converts into a normal error result, exactly like any other after-hook
    exception (`L06-R003`); it is never silently swallowed or, worse, silently successful."""
    ctx = _ctx()

    async def mutating_listener(result: ToolResult, next_: Any) -> Any:
        result.tool_call_id = "mutated"  # type: ignore[misc]
        return await next_()  # pragma: no cover -- the assignment above always raises first

    ctx.events.on(TOOLS_POST_EXECUTE, mutating_listener)

    outcome = await execute_call(_call(value="x"), registry=_registry(_echo()), ctx=ctx)

    assert outcome.is_error
    assert text_of(outcome.to_message()) == "cannot assign to field 'tool_call_id'"
    assert outcome.tool_call_id == "t1"

    # Confirmed directly, independent of the pipeline: mutation itself raises.
    with pytest.raises(FrozenInstanceError):
        outcome.tool_call_id = "mutated"  # type: ignore[misc]


async def test_mixed_raw_and_helper_listeners_share_the_same_authority() -> None:
    """A raw listener and a `register_after_tool_call_hook` listener composed together: both are
    governed by the same authoritative boundary regardless of which one runs first."""
    ctx = _ctx()
    definition = _echo(
        execute=lambda tool_call_id, args: ToolResult(
            tool_call_id="t1", content=(TextBlock(text="x"),), tool_name="echo"
        )
    )

    async def raw_listener(result: ToolResult, next_: Any) -> ToolResult:
        replacement = ToolResult(
            tool_call_id="raw-rewrite", content=(TextBlock(text="raw"),), tool_name="raw-rewrite"
        )
        return await next_(replacement)

    def helper_hook(result: ToolResult) -> AfterToolCallOverride:
        return AfterToolCallOverride(details={"seen": text_of(result.to_message())})

    ctx.events.on(TOOLS_POST_EXECUTE, raw_listener)
    register_after_tool_call_hook(ctx, helper_hook)

    call = _call(value="x")
    outcome = await execute_call(call, registry=_registry(definition), ctx=ctx)

    assert outcome.tool_call_id == call.id
    assert outcome.tool_name == "echo"
    assert outcome.details == {"seen": "raw"}


async def test_middle_listener_failure_skips_later_listeners_with_a_raw_listener_present() -> (
    None
):
    """The failure short-circuit holds even when a raw listener is mixed into the chain: a
    middle failure replaces the accumulated result and later listeners -- raw or helper -- never
    run."""
    ctx = _ctx()
    ran: list[str] = []

    async def first_raw(result: ToolResult, next_: Any) -> ToolResult:
        from dataclasses import replace

        return await next_(replace(result, details={"first": True}))

    def exploding(result: ToolResult) -> AfterToolCallOverride:
        raise RuntimeError("boom")

    def never_runs(result: ToolResult) -> AfterToolCallOverride:
        ran.append("ran")
        return AfterToolCallOverride(details={"should": "not appear"})

    ctx.events.on(TOOLS_POST_EXECUTE, first_raw)
    register_after_tool_call_hook(ctx, exploding)
    register_after_tool_call_hook(ctx, never_runs)

    outcome = await execute_call(_call(value="x"), registry=_registry(_echo()), ctx=ctx)

    assert ran == []
    assert outcome.is_error
    assert text_of(outcome.to_message()) == "boom"
