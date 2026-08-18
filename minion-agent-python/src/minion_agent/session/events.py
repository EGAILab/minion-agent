"""The session event vocabulary.

Two tiers (design spec section 5). The *surface* subset is what
`derive_messages()` projects into model history; everything else is log-only —
lifecycle, replay fidelity, and the operations that change how derivation
reads the surface.

Model-visible means logged: anything reaching a model request must be
reconstructable from these events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EventKind(StrEnum):
    """Every event a session log can carry."""

    # --- surface: projects into model history ---
    USER_MESSAGE = "user/message"
    ASSISTANT_MESSAGE = "assistant/message"
    TOOL_RESULT = "tool/result"

    # --- log-only: lifecycle ---
    TURN_START = "turn/start"
    TURN_END = "turn/end"
    STEP_START = "step/start"
    STEP_END = "step/end"

    # --- log-only: fidelity and request reconstruction ---
    ASSISTANT_CHUNK = "assistant/chunk"
    TOOL_CALL = "tool/call"
    REQUEST_HEADER = "request/header"

    # --- log-only: operations that change derivation ---
    SESSION_FORKED = "session/forked"
    SESSION_RESET = "session/reset"
    COMPACTION = "compaction"


SURFACE_KINDS: frozenset[EventKind] = frozenset(
    {EventKind.USER_MESSAGE, EventKind.ASSISTANT_MESSAGE, EventKind.TOOL_RESULT}
)
"""Exactly the kinds that project into model history.

Widening this set widens what the model sees, which is why it is stated once
here rather than inferred at each call site.
"""


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """One appended event. `seq` is assigned by the log, never by a caller."""

    seq: int
    kind: EventKind
    data: dict[str, Any]


def is_surface(event: SessionEvent) -> bool:
    """Whether `event` projects into model history."""
    return event.kind in SURFACE_KINDS
