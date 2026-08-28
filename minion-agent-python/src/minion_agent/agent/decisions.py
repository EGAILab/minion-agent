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
from ..tools import ToolDefinition
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


@dataclass
class RunContext:
    """Pinned Pi's `AgentContext`: the run-local, mutable transcript/tool
    visibility snapshot the provider actually sees on every request within
    one run (Layer 08, PASS 3 -- `L08-R001`). Taken once at run start
    (`Agent.createContextSnapshot()`'s own shallow top-level-array-copy
    semantics exactly: `messages`/`tools` are each a fresh top-level copy,
    never a deep clone), then locally extended in place as the run's own
    turns append messages -- never re-read from the certified Layer-07
    `AgentInstance`/Session/`ToolRegistry` mid-run, so an outside caller
    mutating any of those after this run started does not retroactively
    affect it. `prepareNextTurn` may replace this object wholesale for the
    next request only (`RunConfigUpdate.context`); a replacement is never
    persisted back to `AgentInstance`, matching pinned Pi's own
    `currentContext = nextTurnSnapshot.context ?? currentContext` -- a
    whole-object swap, not a per-field merge."""

    system_prompt: str
    messages: list[Message]
    tools: tuple[ToolDefinition, ...]


@dataclass
class RunConfig:
    """Pinned Pi's own `model`/`reasoning` half of `AgentLoopConfig`, kept
    as an object separate from `RunContext` exactly as pinned Pi's own
    `createContextSnapshot()`/`createLoopConfig()` split does (Layer 08,
    PASS 3 -- `L08-R001`)."""

    model: ModelId
    thinking_level: ThinkingLevel


@dataclass(frozen=True, slots=True)
class RunConfigUpdate:
    """Pinned Pi's `AgentLoopTurnUpdate` (`prepareNextTurn`'s return value):
    an optional WHOLE replacement for `context` (`RunContext`, Layer 08,
    PASS 3 correction -- an earlier revision truncated this to a
    `system_prompt`-only override, contradicting pinned Pi's own
    `context?: AgentContext` field, which can replace `messages`/`tools`
    too), plus independent optional replacements for `model`/
    `thinking_level`. `None` on any field means "keep the current run-local
    value" -- the terminal `RunConfigUpdate()` (all fields `None`) is a pure
    pass-through, exactly like `Enter`'s own "no override" shape. Never
    persisted back to the certified Layer-07 `AgentInstance`: pinned Pi's
    own `prepareNextTurn` only ever affects the local `config`/
    `currentContext` a single `runLoop` call keeps, not `Agent._state`."""

    context: RunContext | None = None
    model: ModelId | None = None
    thinking_level: ThinkingLevel | None = None
