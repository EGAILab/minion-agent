"""Messages, stop reasons, and token accounting.

Mirrors Pi's semantics: the same stop-reason vocabulary, the same treatment of
reasoning tokens as a subset of output rather than an extra class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .content import ContentBlock, TextBlock


class StopReason(StrEnum):
    """Why a provider stopped generating."""

    PENDING = "pending"
    """Still streaming; never a settled message's reason."""

    STOP = "stop"
    """The model ended its turn."""

    LENGTH = "length"
    """The output token cap was reached."""

    TOOL_USE = "tool_use"
    """The model requested tools and expects results."""

    ERROR = "error"
    """The request failed. `error_message` says how."""

    ABORTED = "aborted"
    """The caller cancelled."""

    DEFERRED = "deferred"
    """The provider accepted the request but the result is not ready yet;
    see `AssistantMessage.deferred`."""


@dataclass(frozen=True, slots=True)
class Cost:
    """Monetary cost for one request's token accounting, matching `Usage`'s
    token breakdown (design spec section 4)."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one request.

    `reasoning` is a *subset* of `output` where a provider reports it, so it is
    deliberately excluded from `total` — counting it separately would
    double-count the same tokens.

    `total_tokens` is the provider's own reported total, not necessarily equal
    to `total`: a provider may report overhead not broken out in the other
    fields. Until a real (non-mock) adapter supplies it, it defaults to 0.
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    """Subset of `cache_write` written with 1h retention. Only some providers
    (e.g. Anthropic) report this split."""
    reasoning: int | None = None
    total_tokens: int = 0
    cost: Cost = Cost()

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write


@dataclass(frozen=True, slots=True)
class DiagnosticError:
    """A structured error carried by an `AssistantMessageDiagnostic`."""

    message: str
    name: str | None = None
    stack: str | None = None
    code: str | int | None = None


@dataclass(frozen=True, slots=True)
class AssistantMessageDiagnostic:
    """Redacted provider/runtime diagnostics for failures and recoveries
    (design spec section 4)."""

    type: str
    timestamp: int
    error: DiagnosticError | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DeferredHandle:
    """A pending async/deferred provider response, polled or resumed later
    (design spec section 4)."""

    provider: str
    model_id: str
    api: str
    id: str
    expires_at: int | None = None
    poll_after_ms: int | None = None
    data: Any | None = None


@dataclass(frozen=True, slots=True)
class UserMessage:
    """Input from a user or an application."""

    content: tuple[ContentBlock, ...]
    timestamp: int


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """One settled model response. Carries the Pi-visible response
    identity/state needed by provider replay and caller behavior (design
    spec section 4)."""

    content: tuple[ContentBlock, ...]
    stop_reason: StopReason
    usage: Usage
    model: str
    provider: str
    timestamp: int
    error_message: str | None = None
    api: str = "mock"
    """Wire protocol that served this response. Defaults to `"mock"` only
    because the mock adapter is the sole registered adapter today
    (LLM-F003/LLM-F006's disposition) -- real adapters populate this from
    `request.model.api`, which is always known by the time a response
    settles."""
    response_model: str | None = None
    """Concrete model when different from the requested `model` (e.g. a
    router resolving `auto` to a specific model)."""
    response_id: str | None = None
    """Provider-specific response/message identifier, when the upstream API
    exposes one."""
    diagnostics: tuple[AssistantMessageDiagnostic, ...] | None = None
    deferred: DeferredHandle | None = None
    raw_stop_reason: str | None = None
    """The provider's own stop-reason string, preserved alongside the
    normalized `stop_reason`."""
    end_turn: bool | None = None
    """Provider indication of whether the model explicitly ended its turn.
    Preserved for debugging; does not affect agent control flow."""


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """The result of one tool call, linked by `tool_call_id`."""

    tool_call_id: str
    content: tuple[ContentBlock, ...]
    timestamp: int
    tool_name: str
    """The tool that produced this result. Required -- Pi's `ToolResultMessage.toolName`
    is non-optional (`packages/ai/src/types.ts`); the pipeline knows which tool it
    called even when the tool itself does not (LLM-F0-delta finding A)."""
    is_error: bool = False
    details: Any | None = None
    usage: Usage | None = None
    """Usage from the tool execution itself. Not part of main LLM context
    accounting (design spec section 4)."""
    added_tool_names: tuple[str, ...] | None = None
    """Names that became available after this result, for providers with
    native deferred tool loading."""


type Message = UserMessage | AssistantMessage | ToolResultMessage


def text_of(message: Message) -> str:
    """Concatenate a message's text blocks, ignoring every other kind."""
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))
