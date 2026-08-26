"""The `tools/*` event vocabulary and its declared dispatch modes."""

from __future__ import annotations

from ..runtime import DispatchMode, EventBus

TOOLS_PRE_EXECUTE = "tools/pre-execute"
"""Waterfall returning `Block | Proceed`. Terminal: `Proceed(validated args)`."""

TOOLS_POST_EXECUTE = "tools/post-execute"
"""Waterfall over the result. Terminal: the result as currently transformed."""

TOOLS_UPDATE = "tools/update"
"""A tool's partial output. Live only -- never logged, never model-visible."""

TOOLS_EXECUTION_START = "tools/execution-start"
"""One call is about to be resolved/prepared/validated/executed. Live only --
matches pinned Pi's `tool_execution_start` (`packages/agent/src/types.ts::AgentEvent`).
Emitted for every tool call in source order, before resolution, regardless of the eventual
outcome (unknown tool, blocked, failed, or successful) -- Layer 06 (`TOOL-017`)."""

TOOLS_EXECUTION_END = "tools/execution-end"
"""One call has produced its final result (`tool_execution_end`). Live only. Emitted for every
tool call exactly once, after `tools/post-execute` has run (when it runs at all) -- in parallel
execution this fires in actual completion order, independent of source order (`TOOL-017`)."""

TOOLS_REGISTERED = "tools/registered"
"""A tool joined the registry. Emitted for `added_tool_names` too."""

TOOLS_EVENT_MODES: dict[str, DispatchMode] = {
    TOOLS_PRE_EXECUTE: DispatchMode.WATERFALL,
    TOOLS_POST_EXECUTE: DispatchMode.WATERFALL,
    TOOLS_UPDATE: DispatchMode.EMIT,
    TOOLS_EXECUTION_START: DispatchMode.EMIT,
    TOOLS_EXECUTION_END: DispatchMode.EMIT,
    TOOLS_REGISTERED: DispatchMode.EMIT,
}


def declare_tools_events(bus: EventBus) -> None:
    """Declare every tools event. Idempotent for matching modes."""
    for name, mode in TOOLS_EVENT_MODES.items():
        bus.declare(name, mode)
