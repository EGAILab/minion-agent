"""CE-L13-WP131-02 revision 6 witnesses that need a live race or a real host permission state.

`L13-WP131-I001` (R-I1): the outcome is decided by ORDER at the work's settle point --
`W-I1`..`W-I4`.
`L13-WP131-C012` (G1 on `EXEC-008`): real-host rows of the access-site rule -- `W-G6`, `W-G11`,
`W-G12`. The scripted per-code rows are canonical scenarios (`conformance/agent/builtin-read-*`)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import stat as _stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import Err, FsError, FsErrorCode, LocalFileSystem
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin import create_ls_tool, create_read_tool
from minion_agent.tools.builtin._signal import race_abort
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

_WINDOWS = os.name == "nt"
_POSIX_ROOT = not _WINDOWS and os.geteuid() == 0


class _Gate:
    """The real local ctx.fs; the `blocked` operation waits for `release`, then returns the real
    answer or `fail_with`. Every call is logged."""

    def __init__(self, root: Path, blocked: str, fail_with: FsErrorCode | None = None) -> None:
        self._local = LocalFileSystem(str(root))
        self.blocked = blocked
        self.fail_with = fail_with
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._local, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            if name == self.blocked:
                self.entered.set()
                await self.release.wait()
                if self.fail_with is not None:
                    return Err(FsError(self.fail_with, "scripted", args[0] if args else None))
            return await method(*args, **kwargs)

        return call


async def _run(tool: Any, arguments: dict[str, Any], signal: Any) -> tuple[bool, str]:
    registry = ToolRegistry()
    registry.register(tool)
    ctx = Context()
    declare_tools_events(ctx.events)
    result = await execute_call(
        ToolCallBlock(id="t", name=tool.name, arguments=arguments),
        registry=registry,
        ctx=ctx,
        signal=signal,
    )
    first = result.content[0]
    assert isinstance(first, TextBlock)
    return result.is_error, first.text


async def _abort_and_release_in_one_step(
    tool: Any, arguments: dict[str, Any], gate: _Gate
) -> tuple[bool, str]:
    """Codex's I001 witness shape: the work is blocked; `abort()` then `release()` happen in the
    same synchronous step, so the work can settle before any poll observes the abort."""
    controller = RunAbortController()
    call = asyncio.ensure_future(_run(tool, arguments, controller.signal))
    await asyncio.wait_for(gate.entered.wait(), timeout=5)
    controller.abort()
    gate.release.set()
    return await asyncio.wait_for(call, timeout=5)


# ---------------------------------------------------------------------------
# I001 -- outcome decided by order at the settle point (R-I1)
# ---------------------------------------------------------------------------


async def test_w_i1_read_abort_then_release_in_one_step_is_operation_aborted(
    tmp_path: Path,
) -> None:
    """`W-I1`: abort, then the blocked content read settles before any poll -> `Operation aborted`
    (Pi: the abort listener rejected first; `if (!aborted) resolve` never delivers the result)."""
    (tmp_path / "a.txt").write_text("hello")
    gate = _Gate(tmp_path, blocked="read_binary_file")
    tool = create_read_tool(gate)  # type: ignore[arg-type]
    assert await _abort_and_release_in_one_step(tool, {"path": "a.txt"}, gate) == (
        True,
        "Operation aborted",
    )


async def test_w_i2_ls_abort_then_release_in_one_step_is_operation_aborted(tmp_path: Path) -> None:
    """`W-I2`, `ls`: the same ordering at the enumeration seam (ls.ts: the promise is already
    rejected when the listing completes)."""
    (tmp_path / "a.txt").write_text("x")
    gate = _Gate(tmp_path, blocked="list_dir_raw")
    tool = create_ls_tool(gate)  # type: ignore[arg-type]
    assert await _abort_and_release_in_one_step(tool, {}, gate) == (True, "Operation aborted")


async def test_w_i3_failure_after_the_abort_is_operation_aborted(tmp_path: Path) -> None:
    """`W-I3`: the work fails AFTER the abort -> `Operation aborted` (read.ts:
    `if (!aborted) reject(error)`)."""
    (tmp_path / "a.txt").write_text("hello")
    gate = _Gate(tmp_path, blocked="read_binary_file", fail_with=FsErrorCode.NOT_FOUND)
    tool = create_read_tool(gate)  # type: ignore[arg-type]
    assert await _abort_and_release_in_one_step(tool, {"path": "a.txt"}, gate) == (
        True,
        "Operation aborted",
    )


async def test_w_i3_failure_before_any_abort_keeps_its_own_error(tmp_path: Path) -> None:
    """`W-I3`, other half: with no abort, the same failure is the read site's own error."""
    (tmp_path / "a.txt").write_text("hello")
    gate = _Gate(tmp_path, blocked="read_binary_file", fail_with=FsErrorCode.NOT_FOUND)
    gate.release.set()
    tool = create_read_tool(gate)  # type: ignore[arg-type]
    absolute = str(tmp_path / "a.txt")
    assert await _run(tool, {"path": "a.txt"}, RunAbortController().signal) == (
        True,
        f"Cannot read {absolute}: no such file or directory",
    )


async def test_w_i4_result_settled_before_the_abort_is_kept() -> None:
    """`W-I4`: the work settles, THEN the abort lands before the caller resumes -> the result
    stands (Pi's promise was already resolved). Guards against re-checking after settling."""
    controller = RunAbortController()
    loop = asyncio.get_running_loop()

    async def work() -> str:
        loop.call_soon(controller.abort)  # runs after the work returns, before race_abort resumes
        return "result"

    assert await race_abort(work(), controller.signal) == "result"
    assert controller.signal.aborted


# ---------------------------------------------------------------------------
# C012 under G1 -- real host rows
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _unreadable(path: Path) -> Iterator[None]:
    """A deny-`RD` ACE on Windows (`FILE_READ_DATA` / `FILE_LIST_DIRECTORY`), mode 000 on POSIX.
    Always restored."""
    if _WINDOWS:
        user = os.environ["USERNAME"]
        subprocess.run(
            ["icacls", str(path), "/deny", f"{user}:(RD)"], check=True, capture_output=True
        )
        try:
            yield
        finally:
            subprocess.run(["icacls", str(path), "/reset"], check=True, capture_output=True)
    else:
        original = _stat.S_IMODE(os.stat(path).st_mode)
        os.chmod(path, 0o000)
        try:
            yield
        finally:
            os.chmod(path, original)


class _Log:
    def __init__(self, root: Path) -> None:
        self._local = LocalFileSystem(str(root))
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._local, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return await method(*args, **kwargs)

        return call


@pytest.mark.skipif(_POSIX_ROOT, reason="root bypasses POSIX permission bits")
@pytest.mark.parametrize("kind", ["file", "directory"])
async def test_w_g6_w_g12_real_unreadable_target_fails_the_access_check(
    tmp_path: Path, kind: str
) -> None:
    """`W-G6` (unreadable file) / `W-G12` (unreadable directory, `CE13-C003`): the real host's
    permission state fails `check_readable` -> `Cannot access`, and no content read is attempted.
    On Windows this is `EXEC-008`'s certified readability disposition (disclosed `W-1`)."""
    target = tmp_path / "target"
    if kind == "file":
        target.write_text("x")
    else:
        target.mkdir()
    fs = _Log(tmp_path)
    tool = create_read_tool(fs)  # type: ignore[arg-type]
    with _unreadable(target):
        observed = await _run(tool, {"path": "target"}, RunAbortController().signal)
    assert observed == (True, f"Cannot access {target}: permission denied")
    assert "read_binary_file" not in fs.calls
    assert fs.calls[0] == "check_readable"


@pytest.mark.parametrize("via_symlink", [False, True])
async def test_w_g11_real_readable_directory_is_the_read_site(
    tmp_path: Path, via_symlink: bool
) -> None:
    """`W-G11`: a readable directory passes the access check and fails the content read, so it is
    the READ site (Pi: access ok, readFile EISDIR). On Windows the certified Python
    `read_binary_file` classifies a directory read as `permission_denied` -- disclosed divergence
    `W-3` (`L12-WINDOWS-DIRECTORY-READ`, minion-agent#67; not worked around)."""
    (tmp_path / "sub").mkdir()
    path = "sub"
    if via_symlink:
        os.symlink(tmp_path / "sub", tmp_path / "link", target_is_directory=True)
        path = "link"
    fs = _Log(tmp_path)
    tool = create_read_tool(fs)  # type: ignore[arg-type]
    observed = await _run(tool, {"path": path}, RunAbortController().signal)
    phrase = "permission denied" if _WINDOWS else "is a directory"
    assert observed == (True, f"Cannot read {tmp_path / path}: {phrase}")
    assert fs.calls[:2] == ["check_readable", "read_binary_file"]
