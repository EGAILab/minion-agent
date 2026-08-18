"""The agent layer: identity, inbox, decisions, instances, and projection.

This package is the interface. The driver that runs a turn lives in
`agent_loop` and is package-internal -- reachable only through the factory
that package's plugin provides -- so nothing here imports it.
"""

from .decisions import (
    Enter,
    PreStepDecision,
    PreStepReason,
    Reject,
    TurnStopping,
    resolve_stopping,
)
from .envelope import ClaimPolicy, InboxTarget, InputEnvelope, JsonValue
from .events import (
    AGENT_EVENT_MODES,
    AGENT_INBOX_CLAIMED,
    AGENT_INBOX_INSERTED,
    AGENT_PRE_STEP,
    AGENT_STATUS,
    AGENT_TURN_STOPPING,
    declare_agent_events,
)
from .identity import AgentDefinition, AgentInstanceId, AgentStatus
from .inbox import Inbox, NotJsonSafeOriginError
from .instance import AgentInstance, instance_scope_key
from .plugin import agents_plugin, tools_plugin
from .projection import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
    event_names,
    project,
)
from .registry import AgentHandle, AgentRegistry, DuplicateInstanceError
from .tools import ToolFn, ToolService

__all__ = [
    "AGENT_EVENT_MODES",
    "AGENT_INBOX_CLAIMED",
    "AGENT_INBOX_INSERTED",
    "AGENT_PRE_STEP",
    "AGENT_STATUS",
    "AGENT_TURN_STOPPING",
    "AgentDefinition",
    "AgentEnd",
    "AgentEvent",
    "AgentHandle",
    "AgentInstance",
    "AgentInstanceId",
    "AgentRegistry",
    "AgentStart",
    "AgentStatus",
    "ClaimPolicy",
    "DuplicateInstanceError",
    "Enter",
    "Inbox",
    "InboxTarget",
    "InputEnvelope",
    "JsonValue",
    "MessageEnd",
    "MessageStart",
    "NotJsonSafeOriginError",
    "PreStepDecision",
    "PreStepReason",
    "Reject",
    "ToolExecutionEnd",
    "ToolExecutionStart",
    "ToolFn",
    "ToolService",
    "TurnEnd",
    "TurnStart",
    "TurnStopping",
    "agents_plugin",
    "declare_agent_events",
    "event_names",
    "instance_scope_key",
    "project",
    "resolve_stopping",
    "tools_plugin",
]
