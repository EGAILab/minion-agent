"""A tool may introduce another, available from that transcript point onward.

Registration and declaration are separate. The tool registers the newcomer
through `ctx.tools` -- the same seam a plugin uses -- and `added_tool_names`
records that it did, so the log and the event stream can report it. A result
that only *named* a tool would be a result that could invent capabilities by
assertion.
"""

from dataclasses import replace
from typing import Any

from minion_agent.llm import TextBlock
from minion_agent.runtime import Context
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult, text_result

from .test_execute import _call


def _ctx() -> Context:
    ctx = Context()
    declare_tools_events(ctx.events)
    return ctx


def _definition(name: str, execute: Any) -> ToolDefinition:
    return ToolDefinition(name=name, description=name, parameters=None, execute=execute)


def _loader(registry: ToolRegistry) -> ToolDefinition:
    """A tool that registers `deploy` and declares that it did."""

    def load(args: dict[str, Any]) -> ToolResult:
        registry.register(_definition("deploy", lambda inner: "deployed"))
        return replace(text_result("", "loaded"), added_tool_names=("deploy",))

    return _definition("load", load)


async def test_a_tool_can_introduce_another() -> None:
    registry = ToolRegistry()
    registry.register(_loader(registry))
    assert registry.resolve("deploy") is None

    await execute_call(_call("load"), registry=registry, ctx=_ctx())

    assert registry.resolve("deploy") is not None


async def test_the_new_tool_joins_the_published_schemas() -> None:
    """Which is what makes it available to the *next* request rather than
    only to Python callers."""
    registry = ToolRegistry()
    registry.register(_loader(registry))

    await execute_call(_call("load"), registry=registry, ctx=_ctx())

    assert "deploy" in {schema.name for schema in registry.schemas()}


async def test_the_result_declares_what_it_added() -> None:
    registry = ToolRegistry()
    registry.register(_loader(registry))

    result = await execute_call(_call("load"), registry=registry, ctx=_ctx())

    assert result.added_tool_names == ("deploy",)


async def test_declaring_a_name_does_not_register_anything() -> None:
    """The separation that matters: a result cannot invent a tool by naming
    one. Registration is the act; the declaration only reports it."""
    registry = ToolRegistry()
    registry.register(
        _definition(
            "liar",
            lambda args: replace(text_result("", "trust me"), added_tool_names=("imaginary",)),
        )
    )

    result = await execute_call(_call("liar"), registry=registry, ctx=_ctx())

    assert result.added_tool_names == ("imaginary",)
    assert registry.resolve("imaginary") is None


async def test_an_ordinary_result_adds_nothing() -> None:
    """A tool that never mentions `added_tool_names` gets the empty default
    threaded through by `execute_call`'s rebuild of a returned `ToolResult` --
    not merely because `str` returns can't carry the field at all. Exercising
    the `ToolResult`-return path (not just a bare string) means this fails if
    the rebuild in `execute_call` ever stops propagating the field correctly.
    """
    registry = ToolRegistry()
    registry.register(_definition("plain", lambda args: "ok"))
    registry.register(
        _definition(
            "plain_result",
            lambda args: ToolResult(tool_call_id="", content=(TextBlock(text="ok"),)),
        )
    )

    string_result = await execute_call(_call("plain"), registry=registry, ctx=_ctx())
    result_result = await execute_call(_call("plain_result"), registry=registry, ctx=_ctx())

    assert string_result.added_tool_names == ()
    assert result_result.added_tool_names == ()
