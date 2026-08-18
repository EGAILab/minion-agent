"""The typed span vocabulary.

Covers the operations the runtime already owns. The vocabulary is
language-neutral, so a second implementation emits the same spans.

Telemetry is an observational projection, never a source of truth: the session
log is semantic truth, runtime events are the extension surface, and no
invariant or conformance case may depend on telemetry contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SpanKind(StrEnum):
    """The operations telemetry describes."""

    TURN = "turn"
    STEP = "step"
    PROVIDER_REQUEST = "provider_request"
    TOOL_EXECUTION = "tool_execution"


@dataclass(frozen=True, slots=True)
class Span:
    """One completed operation."""

    kind: SpanKind
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    error: str | None = None
