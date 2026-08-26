"""Agent identity: reusable definitions and live instances.

Two things are routinely called "the agent", and section 6 separates them
before anything else is unambiguous:

* An **AgentDefinition** is reusable configuration -- persona, model, policy.
  It holds no conversation state, so many instances can share one.
* An **AgentInstance** (see `instance.py`) is one live execution identity: one
  inbox, one active-turn state, one session log, one lifecycle owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..llm import ModelId

type AgentInstanceId = str


class AgentStatus(StrEnum):
    """Whether an instance is currently working.

    Exactly two states: `agent/status` is the settle signal, and a third value
    would give that signal more than one meaning. This is Minion's adopted
    realization of pinned Pi's `AgentState.isStreaming: boolean` (`AG-014`):
    `RUNNING` <-> `isStreaming === true`, `IDLE` <-> `isStreaming === false` --
    a lossless, direct representation of the same two-value fact, not a
    divergence, so the disposition is `adopted`, not an "architectural
    adaptation" that changes the observable semantic.
    """

    IDLE = "idle"
    RUNNING = "running"


class ThinkingLevel(StrEnum):
    """Requested reasoning level for future turns (pinned Pi's `ThinkingLevel`,
    `packages/agent/src/types.ts`, `AG-014`) -- adopted verbatim, all seven values."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Reusable configuration shared by every instance of a named agent."""

    name: str
    model: ModelId
    system: str = ""
    max_steps: int = 16
    """Upper bound on steps in one turn, so a tool-calling loop cannot run away."""

    @property
    def scope_name(self) -> str:
        """The scope key for registrations shared by all instances of this
        definition. Instances mint children of it (design spec section 3)."""
        return f"agent-definition:{self.name}"
