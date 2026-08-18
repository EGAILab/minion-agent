"""Every tool call gets a result, even when it fails."""

from typing import Any

from minion_agent.agent.tools import ToolService
from minion_agent.llm import ToolCallBlock, text_of


def _call(name: str = "echo", **arguments: object) -> ToolCallBlock:
    return ToolCallBlock(id="t1", name=name, arguments=dict(arguments))


async def test_a_registered_tool_runs_and_returns_its_output() -> None:
    service = ToolService()
    service.register("echo", lambda args: str(args["value"]))

    result = await service.execute(_call(value="hello"))

    assert text_of(result) == "hello"
    assert not result.is_error
    assert result.tool_call_id == "t1"


async def test_async_tools_are_awaited() -> None:
    service = ToolService()

    async def slow(args: dict[str, Any]) -> str:
        return "async result"

    service.register("slow", slow)

    assert text_of(await service.execute(_call("slow"))) == "async result"


async def test_an_unknown_tool_yields_an_error_result_not_an_exception() -> None:
    """The model must see a result for every call it made, or the transcript
    stops being coherent."""
    service = ToolService()

    result = await service.execute(_call("missing"))

    assert result.is_error
    assert "missing" in text_of(result)


async def test_a_raising_tool_yields_an_error_result() -> None:
    service = ToolService()

    def broken(args: dict[str, Any]) -> str:
        raise RuntimeError("disk on fire")

    service.register("broken", broken)

    result = await service.execute(_call("broken"))

    assert result.is_error
    assert "disk on fire" in text_of(result)


def test_names_lists_registered_tools() -> None:
    service = ToolService()
    service.register("echo", lambda args: "")

    assert service.names() == frozenset({"echo"})


def test_unregistering_withdraws_the_tool() -> None:
    service = ToolService()
    withdraw = service.register("echo", lambda args: "")

    withdraw()

    assert service.names() == frozenset()


def test_withdrawing_twice_is_harmless() -> None:
    service = ToolService()
    withdraw = service.register("echo", lambda args: "")

    withdraw()
    withdraw()

    assert service.names() == frozenset()
