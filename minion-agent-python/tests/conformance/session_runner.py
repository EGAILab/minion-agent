"""Executes session conformance scenarios.

Log operations in, derived messages out, with no model in play — which is what
makes derivation after fork, reset, and repeated compaction assertable without
a provider.
"""

from __future__ import annotations

from typing import Any

from minion_agent.llm.content import TextBlock
from minion_agent.llm.messages import (
    AssistantMessage,
    Message,
    StopReason,
    ToolResultMessage,
    Usage,
    UserMessage,
    text_of,
)
from minion_agent.session.derive import derive_messages, encode_message
from minion_agent.session.events import CORE_SURFACE_KINDS, EventKind
from minion_agent.session.log import SessionLog
from minion_agent.session.operations import compact, fork, reset

_KIND = {
    "user": EventKind.USER_MESSAGE,
    "assistant": EventKind.ASSISTANT_MESSAGE,
    "tool_result": EventKind.TOOL_RESULT,
}


def _message(role: str, text: str) -> Message:
    """Build the message a role carries.

    A plugin-declared kind encodes as a user message: the payload shape is the
    core vocabulary, and only the event *name* is new.
    """
    content = (TextBlock(text=text),)
    if role not in _KIND or role == "user":
        return UserMessage(content=content, timestamp=1)
    if role == "assistant":
        return AssistantMessage(
            content=content,
            stop_reason=StopReason.STOP,
            usage=Usage(),
            model="mock-1",
            provider="mock",
            timestamp=1,
        )
    return ToolResultMessage(tool_call_id="t1", content=content, timestamp=1)


def _role_of(message: Message) -> str:
    return {
        UserMessage: "user",
        AssistantMessage: "assistant",
        ToolResultMessage: "tool_result",
    }[type(message)]


def run_session_scenario(document: dict[str, Any]) -> list[dict[str, str]]:
    """Apply the scenario's steps and return the derived messages."""
    surface = CORE_SURFACE_KINDS | frozenset(document.get("surface_kinds", ()))
    log = SessionLog("scenario", surface_kinds=surface)
    forks = 0

    for step in document["steps"]:
        if "append" in step:
            spec = step["append"]
            role = spec["role"]
            # A core role maps to its event kind; anything else is the plugin
            # name itself, which is the whole point of an open namespace.
            log.append(
                _KIND.get(role, role),
                {"message": encode_message(_message(role, spec["text"]))},
            )
        elif "fork" in step:
            forks += 1
            log = fork(log, f"fork-{forks}", at=step["fork"].get("at"))
        elif "reset" in step:
            reset(log)
        elif "compact" in step:
            spec = step["compact"]
            compact(log, summary=spec["summary"], keep=spec.get("keep", 0))
        # "derive" is a no-op marker: derivation happens once, at the end.

    return [
        {"role": _role_of(message), "text": text_of(message)} for message in derive_messages(log)
    ]
