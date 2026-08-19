"""The tool subsystem: registry, execution pipeline, and batch semantics.

Sits above `llm` and below `agent`. It owns no session state: the loop writes
the log, and a tool package that appended to it would own state it cannot
reason about.
"""

from .batch import BatchOutcome, execute_batch
from .decisions import Block, PreExecuteDecision, Proceed
from .definition import ExecutionMode, ToolDefinition, ToolFn, ToolUpdate
from .events import (
    TOOLS_EVENT_MODES,
    TOOLS_POST_EXECUTE,
    TOOLS_PRE_EXECUTE,
    TOOLS_REGISTERED,
    TOOLS_UPDATE,
    declare_tools_events,
)
from .execute import execute_call
from .plugin import tools_plugin
from .registry import ToolRegistry, register_tool
from .result import ToolResult, text_result

__all__ = [
    "TOOLS_EVENT_MODES",
    "TOOLS_POST_EXECUTE",
    "TOOLS_PRE_EXECUTE",
    "TOOLS_REGISTERED",
    "TOOLS_UPDATE",
    "BatchOutcome",
    "Block",
    "ExecutionMode",
    "PreExecuteDecision",
    "Proceed",
    "ToolDefinition",
    "ToolFn",
    "ToolRegistry",
    "ToolResult",
    "ToolUpdate",
    "declare_tools_events",
    "execute_batch",
    "execute_call",
    "register_tool",
    "text_result",
    "tools_plugin",
]
