"""Thin runner for `spawn_nul` scenarios (`conformance/schema/spawn-nul-scenario.schema.json`;
L12-D011, spec/execution.md section 21). A case constructs the real `LocalSubprocess` with its
`provider_cwd`, installs a CONFIGURED `baseline` as a fixture (the live baseline otherwise), and
calls `spawn` once; a tool case runs the real `find`, `grep` or `bash` tool through the Layer-06
execute pipeline. The runner only decodes inputs and compares; it performs no validation itself."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import Err, LocalFileSystem
from minion_agent.execution import subprocess as subprocess_module
from minion_agent.execution.spawn_nul import from_units
from minion_agent.execution.subprocess import LocalSubprocess, SpawnOptions
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin.bash import create_bash_tool
from minion_agent.tools.builtin.find import create_find_tool
from minion_agent.tools.builtin.grep import create_grep_tool
from minion_agent.tools.builtin.search_engines import EngineStore
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

HOST_PLATFORM = "win32" if os.name == "nt" else "linux"
SPAWN_NUL_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "spawn-nul"


def text(value: Any) -> str:
    """A `{"utf16": [...]}` string, or a `{linux, win32}` pair of them, for this host."""
    if "utf16" not in value:
        value = value[HOST_PLATFORM]
    return from_units(value["utf16"])


def pairs(value: list[list[Any]] | None) -> list[tuple[str, str]] | None:
    return None if value is None else [(text(k), text(v)) for k, v in value]


async def run_spawn_case(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Returns {"code", "message", "nul_input"}; a started process is terminated and reported."""
    baseline = pairs(case["baseline"])
    if baseline is not None:
        monkeypatch.setattr(subprocess_module, "local_baseline", lambda: list(baseline))
    controller = RunAbortController()
    if case["pre_aborted"]:
        controller.abort()
    overlay = pairs(case["env"])
    provider = LocalSubprocess(text(case["provider_cwd"]))
    result = await provider.spawn(
        [text(arg) for arg in case["argv"]],
        SpawnOptions(
            cwd=None if case["cwd"] is None else text(case["cwd"]),
            env=None if overlay is None else dict(overlay),
            inherit_env=case["inherit_env"],
            signal=controller.signal,
        ),
    )
    if not isinstance(result, Err):
        await result.value.terminate()
        return {"code": "STARTED", "message": None, "nul_input": False}
    return {
        "code": str(result.error.code),
        "message": result.error.message,
        "nul_input": result.error.nul_input,
    }


async def run_tool_case(case: dict[str, Any], workdir: Path, store: EngineStore) -> dict[str, Any]:
    """Runs one tool case in `workdir` (fresh, empty, .git-free); returns is_error and the text."""
    fs, sub = LocalFileSystem(str(workdir)), LocalSubprocess(str(workdir))
    factory = {"find": create_find_tool, "grep": create_grep_tool}.get(case["tool"])
    registry = ToolRegistry()
    registry.register(factory(fs, sub, store) if factory else create_bash_tool(fs, sub))
    ctx = Context()
    declare_tools_events(ctx.events)
    controller = RunAbortController()
    if case["pre_aborted"]:
        controller.abort()
    arguments = {k: text(v) for k, v in case["arguments"][HOST_PLATFORM].items()}
    result = await execute_call(
        ToolCallBlock(id="call-1", name=case["tool"], arguments=arguments),
        registry=registry,
        ctx=ctx,
        signal=controller.signal,
    )
    first = result.content[0]
    assert isinstance(first, TextBlock)
    return {"is_error": result.is_error, "text": first.text}
