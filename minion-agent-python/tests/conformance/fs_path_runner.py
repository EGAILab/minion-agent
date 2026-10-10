"""Runner for `fs_path_domain` scenarios (`conformance/schema/fs-path-domain-scenario.schema.json`,
L12-D001 /
`EXEC-002`, `EXEC-003`: the filesystem path JavaScript-string domain through the Layer-12 `ctx.fs`
seam).

Thin by design: each case runs in a fresh working directory that is already its own realpath; each
step calls ONE
real `LocalFileSystem` operation (or `resolve()` for `target_key`) with the step's path, decoded
from UTF-16 code
units, and only OBSERVES the result -- path results as components relative to the working directory,
strings as
code units, errors as `{code, path}`. A `file_url_tail` path is the working directory's own file URL
plus the tail
(fixture construction). The runner never projects, normalizes or resolves on the provider's behalf.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import Any

from minion_agent.execution import LocalFileSystem
from minion_agent.execution.result import Ok
from minion_agent.runtime.signal import RunAbortController, RunSignal


def string(code_units: list[int]) -> str:
    return struct.pack(f"<{len(code_units)}H", *code_units).decode("utf-16-le", "surrogatepass")


def units(value: str) -> list[int]:
    data = value.encode("utf-16-le", "surrogatepass")
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def observe_path(cwd: str, value: str) -> dict[str, Any]:
    rel = os.path.relpath(value, cwd)
    if rel == ".":
        return {"components": []}
    if rel.startswith("..") or os.path.isabs(rel):
        return {"outside": True}
    return {"components": [units(part) for part in rel.split(os.sep)]}


def observe_error(cwd: str, error: Any) -> dict[str, Any]:
    path = error.path
    return {
        "error": str(error.code),
        "path": observe_path(cwd, path) if isinstance(path, str) else None,
    }


PLATFORM = "win32" if sys.platform == "win32" else "linux"


def applies(case: dict[str, Any]) -> bool:
    """An error-origin case may be declared for some platforms only (`platform_note` says why)."""
    return PLATFORM in case.get("platforms", ["linux", "win32"])


def _aborted_signal() -> RunSignal:
    controller = RunAbortController()
    controller.abort()
    return controller.signal


def path_argument(cwd: str, path: dict[str, Any]) -> str:
    if "utf16" in path:
        return string(path["utf16"])
    return f"{Path(cwd).as_uri()}/{string(path['file_url_tail'])}"


def _fixture_target(cwd: str, p: str) -> str:
    """L12-D007 fixture steps build an OS condition natively; their target must stay inside the
    case directory (Owner containment rule, 2026-10-10). String concatenation, never
    `os.path.join`, which would let an `X:` component (a drive) replace the base."""
    target = os.path.abspath(cwd + os.sep + p)
    base = os.path.normcase(cwd)
    if os.path.commonpath([base, os.path.normcase(target)]) != base:
        raise AssertionError(f"fixture target {p!r} escapes the case directory")
    return target


def _hold(op: str, target: str) -> Any:  # pragma: no cover -- win32-only fixture
    """`hold_exclusive`: a FileShare.None handle; `lock_range`: a shared handle with bytes 0-63
    locked. Both are held until the case ends, so every provider call meets the sharing / lock
    violation."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    k32.CreateFileW.restype = wintypes.HANDLE
    share = 0 if op == "hold_exclusive" else 3  # none / read+write (no delete), as the schema says
    handle = k32.CreateFileW(target, 0xC0000000, share, None, 3, 0x80, None)  # RW, OPEN_EXISTING
    if handle == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), f"{op} fixture open failed", target)
    if op == "lock_range" and not k32.LockFile(handle, 0, 0, 64, 0):
        raise OSError(ctypes.get_last_error(), "lock_range fixture lock failed", target)
    return lambda: k32.CloseHandle(handle)


async def run_case(case: dict[str, Any], root: Path) -> list[Any]:
    releases: list[Any] = []
    try:
        return await _run_steps(case, root, releases)
    finally:
        for release in releases:  # pragma: no cover -- win32-only fixtures
            release()


async def _run_steps(case: dict[str, Any], root: Path, releases: list[Any]) -> list[Any]:
    cwd = os.path.realpath(root)
    fs = LocalFileSystem(cwd)
    observed: list[Any] = []
    for step in case["steps"]:
        p = path_argument(cwd, step["path"])
        op = step["op"]
        if op == "make_symlink":
            _fixture_target(cwd, string(step["to"]["utf16"]))
            os.symlink(string(step["to"]["utf16"]), _fixture_target(cwd, p))
            observed.append({"ok": None})
            continue
        if op in ("hold_exclusive", "lock_range"):  # pragma: no cover -- win32-only fixtures
            releases.append(_hold(op, _fixture_target(cwd, p)))
            observed.append({"ok": None})
            continue
        # L12-D006: `aborted` hands the operation an already-aborted signal (fixture construction).
        signal = _aborted_signal() if step.get("aborted") else None
        if op == "write_file":
            r: Any = await fs.write_file(p, string(step["content"]), signal=signal)
            o: Any = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "read_text_file":
            r = await fs.read_text_file(p, signal=signal)
            o = {"ok": units(r.value)} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "list_dir":
            r = await fs.list_dir(p, signal=signal)
            o = (
                {"names": sorted(units(i.name) for i in r.value)}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "canonical_path":
            r = await fs.canonical_path(p)
            o = observe_path(cwd, r.value) if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "absolute_path":
            r = await fs.absolute_path(p)
            assert isinstance(r, Ok)
            o = (
                {"last": units(os.path.basename(r.value))}
                if step.get("observe") == "last_component"
                else observe_path(cwd, r.value)
            )
        elif op == "target_key":
            r = await fs.resolve(p)
            o = (
                observe_path(cwd, r.value.target_key)
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "exists":
            r = await fs.exists(p)
            o = {"ok": r.value} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "file_info":
            r = await fs.file_info(p)
            o = (
                {"kind": str(r.value.kind), "name": units(r.value.name)}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "append_file":
            r = await fs.append_file(p, string(step["content"]))
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "read_text_lines":
            r = await fs.read_text_lines(p, max_lines=step.get("max_lines"), signal=signal)
            o = (
                {"ok": [units(line) for line in r.value]}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "read_binary_file":
            r = await fs.read_binary_file(p, signal=signal)
            o = {"ok": list(r.value)} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "create_dir":
            r = await fs.create_dir(p, recursive=step["recursive"])
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "remove":
            r = await fs.remove(
                p, recursive=step.get("recursive", False), force=step.get("force", False)
            )
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "rename_file":
            r = await fs.rename_file(p, path_argument(cwd, step["to"]), signal=signal)
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "list_dir_raw":
            r = await fs.list_dir_raw(p)
            o = (
                {"names": sorted(units(name) for name in r.value)}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "probe_dir_entry":
            r = await fs.probe_dir_entry(p)
            o = (
                {"kind": str(r.value.kind), "name": units(r.value.name)}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "check_readable":
            r = await fs.check_readable(p)
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "check_read_write":
            r = await fs.check_read_write(p)
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        else:  # pragma: no cover -- the schema closes the op set
            raise AssertionError(f"unknown op {op}")
        observed.append(o)
    return observed


def expected(step: dict[str, Any]) -> Any:
    """`expect`, or this platform's answer where pinned Node itself differs by platform."""
    return step["expect"] if "expect" in step else step["expect_by_platform"][PLATFORM]


def check(case: dict[str, Any], observed: list[Any]) -> None:
    for index, (step, got) in enumerate(zip(case["steps"], observed, strict=True)):
        want = expected(step)
        assert got == want, (case["id"], index, step.get("op", step.get("tool")), got, want)


async def run_tool_case(case: dict[str, Any], root: Path) -> list[Any]:
    """One `fs_path_tools` case: each step is ONE real built-in tool call
    (`write`/`read`/`ls`/`edit` over the real
    `LocalFileSystem`), observed as `{is_error, text}` with the text as code units."""
    from minion_agent.llm import TextBlock, ToolCallBlock
    from minion_agent.runtime import Context
    from minion_agent.tools.builtin import (
        create_edit_tool,
        create_ls_tool,
        create_read_tool,
        create_write_tool,
    )
    from minion_agent.tools.events import declare_tools_events
    from minion_agent.tools.execute import execute_call
    from minion_agent.tools.registry import ToolRegistry

    fs = LocalFileSystem(os.path.realpath(root))
    registry = ToolRegistry()
    for create in (create_write_tool, create_read_tool, create_ls_tool, create_edit_tool):
        registry.register(create(fs))
    observed = []
    for index, step in enumerate(case["steps"]):
        ctx = Context()
        declare_tools_events(ctx.events)
        call = ToolCallBlock(
            id=f"c{index}", name=step["tool"], arguments=_decode(step["arguments"])
        )
        result = await execute_call(call, registry=registry, ctx=ctx)
        first = result.content[0] if result.content else None
        body = first.text if isinstance(first, TextBlock) else ""
        observed.append({"is_error": result.is_error, "text": {"utf16": units(body)}})
    return observed


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if "utf16" in value:
            return string(value["utf16"])
        return {key: _decode(item) for key, item in value.items()}
    return value
