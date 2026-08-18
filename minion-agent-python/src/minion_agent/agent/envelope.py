"""Input envelopes: the unit provenance attaches to.

Provenance attaches to *inputs* rather than turns because a turn can have more
than one cause. Under the `all` claim policy one turn consumes several queued
inputs with different origins, so a single `turn.origin` would not be well
defined (design spec section 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..llm import UserMessage

type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class InboxTarget(StrEnum):
    """Which boundary an input waits for."""

    NEXT_TURN = "next-turn"
    """A queued prompt: claimed when a turn opens."""

    NEXT_STEP = "next-step"
    """Steering or injected context: claimed at the next step boundary."""


class ClaimPolicy(StrEnum):
    """How many queued inputs a boundary takes.

    Pi's `steeringMode` and `followUpMode`, one policy per boundary.
    """

    ALL = "all"
    ONE_AT_A_TIME = "one-at-a-time"


@dataclass(frozen=True, slots=True)
class InputEnvelope:
    """One queued input plus the provenance its sender attached.

    `origin` is opaque to the runtime -- never inspected, matched, or
    interpreted -- and must be JSON-safe so it survives the log and a
    reimplementation in another language.
    """

    id: str
    message: UserMessage
    origin: JsonValue = None
