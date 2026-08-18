"""The session event vocabulary.

Two tiers (design spec section 5). The *surface* subset is what
`derive_messages()` projects into model history; everything else is log-only —
lifecycle, replay fidelity, and the operations that change how derivation
reads the surface.

Model-visible means logged: anything reaching a model request must be
reconstructable from these events.

**The namespace is open.** §5 states that plugins may declare session events
that join the surface, so the language-neutral identity of an event is its
*name string*. `EventKind` supplies the core names as constants for ergonomics
and autocompletion — it does not close the namespace, and `"plugin/foo"` is
just as valid an identity as `"user/message"`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

type EventName = str
"""An event's identity. The string is authoritative across languages."""


class EventKind(StrEnum):
    """Core event names.

    Constants, not a closed set: `SessionEvent.kind` is typed `EventName`, and
    any validated name is acceptable. These are the names whose semantics the
    specification defines normatively.
    """

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


CORE_SURFACE_KINDS: frozenset[EventName] = frozenset(
    {EventKind.USER_MESSAGE, EventKind.ASSISTANT_MESSAGE, EventKind.TOOL_RESULT}
)
"""The core kinds that project into model history.

Normative and language-neutral: a second implementation must reproduce exactly
these. A log may carry additional plugin-declared surface names, whose
projections are plugin-scoped rather than cross-language (§5).
"""

SURFACE_KINDS = CORE_SURFACE_KINDS
"""Deprecated alias kept for call sites written before the namespace opened."""

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_-]*)*$")


class InvalidEventNameError(ValueError):
    """An event name that cannot serve as a cross-language identity."""


def validate_event_name(name: str) -> EventName:
    """Return `name` if it is usable as an event identity.

    Validation is about portability, not membership: any well-formed name is
    acceptable, including one no core constant declares. The shape is
    lowercase segments separated by `/`, so a name survives a log, a JSON
    document, and another language's identifier rules unchanged.
    """
    if not isinstance(name, str):
        raise InvalidEventNameError(f"event name must be a string, got {type(name).__name__}")
    if not _NAME_PATTERN.match(name):
        raise InvalidEventNameError(
            f"event name {name!r} is not a valid identity; expected lowercase "
            "segments separated by '/', e.g. 'plugin/foo'"
        )
    return name


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """One appended event. `seq` is assigned by the log, never by a caller."""

    seq: int
    kind: EventName
    data: dict[str, Any]


def is_surface(event: SessionEvent, surface: frozenset[EventName] | None = None) -> bool:
    """Whether `event` projects into model history.

    `surface` defaults to the core set; a log carrying plugin-declared surface
    events passes its own.
    """
    return event.kind in (CORE_SURFACE_KINDS if surface is None else surface)
