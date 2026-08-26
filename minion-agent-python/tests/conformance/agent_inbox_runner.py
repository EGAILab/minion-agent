"""Runs a Layer-07 `agent_inbox` scenario against the real `Inbox` primitive.

Thin by construction: every action calls straight through to a real `Inbox` method
(`steer`/`followup`/`inject`/`claim`/`clear`/`has_pending`). This module owns no queue, no FIFO
logic, no claim-policy branching, and no wake bookkeeping of its own -- it only translates the
scenario's declarative actions into real calls and records whatever they returned.
"""

from __future__ import annotations

from typing import Any

from minion_agent.agent.envelope import ClaimPolicy, InboxTarget
from minion_agent.agent.inbox import Inbox
from minion_agent.llm import TextBlock, UserMessage, text_of

_QUEUE_TARGETS = {
    "steering": InboxTarget.NEXT_STEP,
    "follow_up": InboxTarget.NEXT_TURN,
}
_CLAIM_MODES = {
    "all": ClaimPolicy.ALL,
    "one-at-a-time": ClaimPolicy.ONE_AT_A_TIME,
}


def _message(text: str) -> UserMessage:
    return UserMessage(content=(TextBlock(text=text),), timestamp=1)


def run_agent_inbox_scenario(document: dict[str, Any]) -> dict[str, Any]:
    """Run every action in order; return `{observe name: observed value}`."""
    inbox = Inbox()
    observed: dict[str, Any] = {}

    for action in document["agent_inbox"]["actions"]:
        observe = action.get("observe")
        if "steer" in action:
            inbox.steer(_message(action["steer"]["text"]))
        elif "follow_up" in action:
            inbox.followup(_message(action["follow_up"]["text"]))
        elif "inject" in action:
            inbox.inject(_message(action["inject"]["text"]))
        elif "clear" in action:
            queue = action["clear"]["queue"]
            if queue == "all":
                inbox.clear_all()
            else:
                inbox.clear(_QUEUE_TARGETS[queue])
        elif "claim" in action:
            target = _QUEUE_TARGETS[action["claim"]["queue"]]
            mode = _CLAIM_MODES[action["claim"]["mode"]]
            claimed = inbox.claim(target, mode)
            if observe is not None:
                observed[observe] = [text_of(envelope.message) for envelope in claimed]
        elif "has_queued_messages" in action:
            if observe is not None:
                observed[observe] = inbox.has_pending()
        else:
            raise ValueError(f"unhandled agent_inbox action {action!r}")

    return observed
