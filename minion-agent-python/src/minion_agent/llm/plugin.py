"""Mounting the LLM seam and adapters on the runtime."""

from __future__ import annotations

from pydantic import BaseModel

from ..runtime import Context, plugin
from .adapters.mock import MockAdapter, ScriptedResponse
from .content import TextBlock
from .messages import StopReason
from .service import LlmService


@plugin(name="llm", provides="llm")
async def llm_plugin(ctx: Context, config: None) -> None:
    """Provide the LLM seam. Adapters mount separately and register into it."""
    ctx.provide("llm", LlmService())


class ScriptedResponseConfig(BaseModel):
    """One scripted response, in the shape configuration carries."""

    text: str = ""
    stop_reason: StopReason = StopReason.STOP
    error_message: str | None = None


class MockAdapterConfig(BaseModel):
    script: list[ScriptedResponseConfig] = []


@plugin(name="llm-mock", inject=["llm"], config=MockAdapterConfig)
async def mock_adapter_plugin(ctx: Context, config: MockAdapterConfig) -> None:
    """Register a scripted adapter, withdrawn when this plugin unloads."""
    adapter = MockAdapter(
        [
            ScriptedResponse(
                content=(TextBlock(text=entry.text),) if entry.text else (),
                stop_reason=entry.stop_reason,
                error_message=entry.error_message,
            )
            for entry in config.script
        ]
    )
    ctx.effect(lambda: ctx.llm.register(adapter), "register(mock)")
