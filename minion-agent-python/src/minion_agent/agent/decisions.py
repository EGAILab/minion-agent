"""Closed decision unions for the loop's extension points.

Pi's hooks are single-valued with precise semantics; multi-listener dispatch
would diffuse them. The answer (design spec section 6) is to type each decision
payload as a closed union and let Plan 1's waterfall short-circuit decide which
listener owns the call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from ..llm import UserMessage


@dataclass(frozen=True, slots=True)
class Reject:
    """Do not run a step. The turn closes having spent none."""

    reason: str = ""


@dataclass(frozen=True, slots=True)
class Enter:
    """Run a step with these messages.

    `system_override` and `history_window` apply to this step alone, which is
    how pi's per-call `system_suffix` and `max_history_turns` are expressed.
    """

    messages: tuple[UserMessage, ...]
    system_override: str | None = None
    history_window: int | None = None


type PreStepDecision = Reject | Enter


class PreStepReason(StrEnum):
    """Why a pre-step is happening.

    Pi's `transformContext` and `prepareNextTurn` fire at different lifecycle
    points -- the first before every request including the first, the second
    only between turns. Collapsing both into one undifferentiated event would
    make `prepareNextTurn` impossible to replicate.
    """

    INITIAL = "initial"
    TOOL_RESULTS = "tool_results"
    STEERING = "steering"
    NEXT_TURN = "next_turn"
    CONTINUATION = "continuation"


class TurnStopping(StrEnum):
    """A listener's opinion on whether the turn should continue."""

    NO_OPINION = "no_opinion"
    CONTINUE = "continue"
    STOP = "stop"


def resolve_stopping(decisions: Iterable[TurnStopping]) -> TurnStopping:
    """First non-`NO_OPINION` decision wins; otherwise continue.

    Order-dependent by design. An order-independent "any STOP wins" rule was
    considered and rejected: it collapses CONTINUE into NO_OPINION and diverges
    from the short-circuit pattern every other decision event uses.
    """
    for decision in decisions:
        if decision is not TurnStopping.NO_OPINION:
            return decision
    return TurnStopping.CONTINUE
