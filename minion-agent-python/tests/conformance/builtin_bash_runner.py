"""Runner for the WP-13.3 `bash` canonical scenarios (`conformance/schema/builtin-bash-scenario.schema.
json`). It builds the real `bash` tool over the real local `ctx.fs` and `ctx.subprocess` in a fresh
working directory and runs each case through the REAL Layer 06 `execute_call` pipeline. It never
performs tool behaviour: it only normalizes the full-output path and the working directory, and reads
the full-output file's raw bytes back.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from minion_agent.execution import LocalFileSystem
from minion_agent.execution.subprocess import LocalSubprocess
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin.bash import create_bash_tool
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

HOST_PLATFORM = "win32" if os.name == "nt" else "linux"
_FULL_OUTPUT_IN_TEXT = re.compile(r"Full output: (.+?)\]")
_SPECIAL_TIMEOUTS = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}


def _utf16(text: str) -> bytes:
    return text.encode("utf-16-le", "surrogatepass")


def _units_length(text: str) -> int:
    return len(_utf16(text)) // 2


def _units_slice(text: str, start: int, end: int | None = None) -> str:
    data = _utf16(text)
    chunk = data[start * 2 :] if end is None else data[start * 2 : end * 2]
    return chunk.decode("utf-16-le", "surrogatepass")


async def run_builtin_bash_scenario(document: dict[str, Any]) -> dict[str, Any] | None:
    """Returns `{observed, expected}` for the host's platform, or `None` when the scenario has no
    expectation for it."""
    spec = document["builtin_bash"]
    expected = spec["expect"].get(HOST_PLATFORM)
    if expected is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        cwd = str(root / "missing") if spec.get("missing_cwd") else str(root)
        tool = create_bash_tool(LocalFileSystem(str(root)), LocalSubprocess(cwd))
        registry = ToolRegistry()
        registry.register(tool)
        ctx = Context()
        declare_tools_events(ctx.events)
        controller = RunAbortController()
        if spec.get("signal") == "pre_aborted":
            controller.abort()
        arguments: dict[str, Any] = {"command": spec["command"]}
        if "timeout" in spec:
            timeout = spec["timeout"]
            arguments["timeout"] = _SPECIAL_TIMEOUTS.get(timeout, timeout)
        timer = None
        if "abort_after_ms" in spec:
            timer = asyncio.get_running_loop().call_later(
                spec["abort_after_ms"] / 1000, controller.abort
            )
        try:
            result = await execute_call(
                ToolCallBlock(id="call-1", name="bash", arguments=arguments),
                registry=registry,
                ctx=ctx,
                signal=controller.signal,
            )
        finally:
            if timer is not None:
                timer.cancel()
        first = result.content[0]
        assert isinstance(first, TextBlock)
        text = first.text
        details = dict(result.details)
        match = _FULL_OUTPUT_IN_TEXT.search(text)
        full_path = details.get("fullOutputPath") or (match.group(1) if match else None)
        full_output = None
        if full_path:
            data = Path(full_path).read_bytes()
            full_output = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            Path(full_path).unlink()
            text = text.replace(full_path, "<FULL_OUTPUT>")
        text = text.replace(cwd, "<CWD>")
        content_matches = None
        if "truncation" in details:
            truncation = dict(details["truncation"])
            shown = truncation.pop("content")
            content_matches = text.startswith(shown) and (shown == "" or text[len(shown) :].startswith("\n\n[Showing"))
            details = {"truncation": truncation, "fullOutputPath": "<FULL_OUTPUT>"}
        observed = {
            "is_error": result.is_error,
            "text": text,
            "text_length": _units_length(text),
            "text_head": _units_slice(text, 0, 200),
            "text_tail": _units_slice(text, max(0, _units_length(text) - 400)),
            "details": details,
            "full_output": full_output,
            "content_matches": content_matches,
        }
    return {"observed": observed, "expected": expected}
