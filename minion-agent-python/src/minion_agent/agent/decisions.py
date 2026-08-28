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

from ..llm import Message, ModelId
from .identity import ThinkingLevel


@dataclass(frozen=True, slots=True)
class Reject:
    """Do not run a step. The turn closes having spent none."""

    reason: str = ""


@dataclass(frozen=True, slots=True)
class Enter:
    """Run a step with these messages.

    `system_override` and `history_window` apply to this step alone, which is
    how pi's per-call `system_suffix` and `max_history_turns` are expressed.

    `messages` is typed as the full `Message` union, not `UserMessage` alone
    (`AG-011`, `L07-R002`): claimed inbox input can now be any pinned-Pi
    `AgentMessage` variant. This is a type-signature widening only -- no step/
    turn timing or decision logic changed.
    """

    messages: tuple[Message, ...]
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


@dataclass(frozen=True, slots=True)
class RunConfigUpdate:
    """Pinned Pi's `AgentLoopTurnUpdate` (`prepareNextTurn`'s return value):
    an optional replacement for `system_prompt`/`model`/`thinking_level`,
    applying to the next provider request only. `None` on any field means
    "keep the current run-local value" -- the terminal `RunConfigUpdate()`
    (all fields `None`) is a pure pass-through, exactly like `Enter`'s own
    "no override" shape. Never persisted back to the certified Layer-07
    `AgentInstance`: pinned Pi's own `prepareNextTurn` only ever affects the
    local `config`/`currentContext` a single `runLoop` call keeps, not
    `Agent._state` (Layer 08, PASS 2)."""

    system_prompt: str | None = None
    model: ModelId | None = None
    thinking_level: ThinkingLevel | None = None
