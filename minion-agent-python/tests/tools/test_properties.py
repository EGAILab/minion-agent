"""Properties that must hold for any batch."""

from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from minion_agent.llm import ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools.batch import execute_batch
from minion_agent.tools.definition import ExecutionMode, ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult

names = st.lists(st.sampled_from(["ok", "slow", "broken", "missing"]), max_size=8)
modes = st.sampled_from([ExecutionMode.PARALLEL, ExecutionMode.SEQUENTIAL])
loop_settings = settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)


def _ctx() -> Context:
    ctx = Context()
    declare_tools_events(ctx.events)
    return ctx


def _registry(mode: ExecutionMode) -> ToolRegistry:
    def broken(args: dict[str, Any]) -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    for name, execute in (("ok", lambda args: "ok"), ("slow", lambda args: "slow")):
        registry.register(
            ToolDefinition(
                name=name, description=name, parameters=None, execute=execute, label=name, mode=mode
            )
        )
    registry.register(
        ToolDefinition(
            name="broken", description="broken", parameters=None, execute=broken, label="broken"
        )
    )
    return registry


def _calls(items: list[str]) -> tuple[ToolCallBlock, ...]:
    return tuple(
        ToolCallBlock(id=f"t{index}", name=name, arguments={}) for index, name in enumerate(items)
    )


@given(names, modes)
@loop_settings
async def test_every_call_gets_exactly_one_result(items: list[str], mode: ExecutionMode) -> None:
    """The invariant the whole subsystem preserves."""
    outcome = await execute_batch(_calls(items), registry=_registry(mode), ctx=_ctx())

    assert len(outcome.results) == len(items)


@given(names, modes)
@loop_settings
async def test_results_are_in_source_order(items: list[str], mode: ExecutionMode) -> None:
    """Strengthened: ids are assigned in *reverse* of source position, so a
    bug that sorted results by `tool_call_id` -- which would coincide with
    source order under the brief's monotonically-increasing `t{index}` ids --
    cannot masquerade as preserving source order here.
    """
    calls = tuple(
        ToolCallBlock(id=f"t{len(items) - 1 - index}", name=name, arguments={})
        for index, name in enumerate(items)
    )

    outcome = await execute_batch(calls, registry=_registry(mode), ctx=_ctx())

    assert [r.tool_call_id for r in outcome.results] == [c.id for c in calls]


@given(names, modes)
@loop_settings
async def test_completion_order_is_a_permutation_of_the_batch(
    items: list[str], mode: ExecutionMode
) -> None:
    """Two orders over the same set: every call completes exactly once."""
    outcome = await execute_batch(_calls(items), registry=_registry(mode), ctx=_ctx())

    assert sorted(outcome.completion_order) == sorted(r.tool_call_id for r in outcome.results)


@given(names, modes)
@loop_settings
async def test_a_batch_without_terminating_results_never_terminates(
    items: list[str], mode: ExecutionMode
) -> None:
    outcome = await execute_batch(_calls(items), registry=_registry(mode), ctx=_ctx())

    assert not outcome.terminate


@given(st.integers(min_value=1, max_value=6))
@loop_settings
async def test_a_unanimous_batch_of_any_size_terminates(size: int) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="stop",
            description="stop",
            parameters=None,
            execute=lambda args: ToolResult(
                tool_call_id="", content=(), tool_name="stop", terminate=True
            ),
            label="stop",
        )
    )

    outcome = await execute_batch(_calls(["stop"] * size), registry=registry, ctx=_ctx())

    assert outcome.terminate


@given(st.integers(min_value=1, max_value=6), st.integers(min_value=1, max_value=6), modes)
@loop_settings
async def test_a_mixed_batch_never_terminates(
    stopping: int, non_stopping: int, mode: ExecutionMode
) -> None:
    """The fold is unanimity (`all`), not "any": strengthens the pair above,
    which -- built from tools that never terminate, and tools that always do
    -- cannot tell an `all` fold from an `any` fold apart, since both agree on
    an all-false or all-true batch. Only a mixed batch separates them.
    """
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="stop",
            description="stop",
            parameters=None,
            execute=lambda args: ToolResult(
                tool_call_id="", content=(), tool_name="stop", terminate=True
            ),
            label="stop",
            mode=mode,
        )
    )
    registry.register(
        ToolDefinition(
            name="go",
            description="go",
            parameters=None,
            execute=lambda args: "go",
            label="go",
            mode=mode,
        )
    )

    outcome = await execute_batch(
        _calls(["stop"] * stopping + ["go"] * non_stopping),
        registry=registry,
        ctx=_ctx(),
    )

    assert not outcome.terminate
