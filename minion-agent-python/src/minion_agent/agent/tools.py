"""A trivial tool service: enough for the loop to close a round trip.

Plan 4 replaces this with the real registry -- scoped registration, the
`tools/*` waterfalls, batching, execution modes, streaming results, and the
`terminate` fold. What matters here is the contract the loop depends on:

    every tool call produces exactly one tool result

A failure is an error *result*, never an exception. A call without a result
leaves the transcript incoherent, and the model would be asked to continue
from a conversation that does not make sense.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from ..llm import TextBlock, ToolCallBlock, ToolResultMessage

type ToolFn = Callable[[dict[str, Any]], Awaitable[str] | str]


class ToolService:
    """Maps tool names to callables."""

    __service_name__ = "tools"

    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> Callable[[], None]:
        """Register `fn` under `name`; returns a withdrawal handle."""
        self._tools[name] = fn

        def withdraw() -> None:
            if self._tools.get(name) is fn:
                del self._tools[name]

        return withdraw

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    @staticmethod
    def _result(call: ToolCallBlock, text: str, *, is_error: bool) -> ToolResultMessage:
        return ToolResultMessage(
            tool_call_id=call.id,
            content=(TextBlock(text=text),),
            timestamp=0,
            is_error=is_error,
        )

    async def execute(self, call: ToolCallBlock) -> ToolResultMessage:
        """Run `call`, returning a result whether it succeeds or fails."""
        fn = self._tools.get(call.name)
        if fn is None:
            return self._result(call, f"unknown tool {call.name!r}", is_error=True)
        try:
            outcome = fn(call.arguments)
            text = await outcome if inspect.isawaitable(outcome) else outcome
        except Exception as error:  # noqa: BLE001 - surfaced to the model, not raised
            return self._result(call, f"{type(error).__name__}: {error}", is_error=True)
        return self._result(call, str(text), is_error=False)
