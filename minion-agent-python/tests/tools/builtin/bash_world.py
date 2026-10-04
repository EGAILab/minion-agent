"""Test execution worlds for the WP-13.3 `bash` witnesses.

`WorldFs` answers existence from a scripted set (or the real local filesystem) and keeps the real
temp area for the full-output file. `WorldSubprocess` wraps the REAL `LocalSubprocess`: it declares
any platform and environment snapshot, records every spawn and termination, and maps the selected
shell or the `where`/`which` lookup to a controlled Python child -- so every shell-selection branch
runs on either host. Neither performs bash behaviour.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from minion_agent.execution import (
    EnvSnapshot,
    Err,
    FsError,
    FsErrorCode,
    LocalFileSystem,
    Ok,
    Platform,
)
from minion_agent.execution.filesystem import DirEntryProbe, DirEntryProbeKind
from minion_agent.execution.subprocess import LocalSubprocess, Process, SpawnOptions
from minion_agent.execution.world import ExecutionWorldIdentity

PY = sys.executable


def child(code: str) -> list[str]:
    """argv for a Python child running `code`."""
    return [PY, "-c", code]


class WorldFs:
    """`ctx.fs` with scripted existence. `existing=None` defers to the real local filesystem."""

    def __init__(
        self,
        root: str,
        *,
        existing: set[str] | None = None,
        probe_error: FsErrorCode | None = None,
        file_info_error: FsErrorCode | None = None,
        create_temp_error: FsErrorCode | None = None,
        append_error: FsErrorCode | None = None,
        append_gate: asyncio.Event | None = None,
        world: ExecutionWorldIdentity | None = None,
    ) -> None:
        self._local = LocalFileSystem(root)
        self.cwd = root
        self.execution_world = world or ExecutionWorldIdentity.local()
        self.existing = existing
        self.probe_error = probe_error
        self.file_info_error = file_info_error
        self.create_temp_error = create_temp_error
        self.append_error = append_error
        self.append_gate = append_gate
        """When set, every `append_file` blocks until the event is set (a slow, conforming fs)."""
        self.append_started = asyncio.Event()
        self.appends_completed = 0
        self.calls: list[str] = []

    def _scripted(self, path: str, error: FsErrorCode | None) -> Any:
        if error is not None:
            return Err(FsError(error, error.value))
        if self.existing is None:
            return None
        if path in self.existing:
            return Ok(DirEntryProbe(path.rsplit("/", 1)[-1], path, DirEntryProbeKind.FILE))
        return Err(FsError(FsErrorCode.NOT_FOUND, "missing"))

    async def probe_dir_entry(self, path: str, signal: Any = None) -> Any:
        self.calls.append(f"probe_dir_entry {path}")
        scripted = self._scripted(path, self.probe_error)
        return scripted if scripted is not None else await self._local.probe_dir_entry(path)

    async def file_info(self, path: str, signal: Any = None) -> Any:
        self.calls.append(f"file_info {path}")
        scripted = self._scripted(path, self.file_info_error)
        return scripted if scripted is not None else await self._local.file_info(path)

    async def create_temp_file(self, prefix: str = "", suffix: str = "", signal: Any = None) -> Any:
        self.calls.append(f"create_temp_file {prefix} {suffix}")
        if self.create_temp_error is not None:
            return Err(FsError(self.create_temp_error, "scripted"))
        return await self._local.create_temp_file(prefix, suffix)

    async def append_file(self, path: str, content: Any, signal: Any = None) -> Any:
        self.append_started.set()
        if self.append_gate is not None:
            await self.append_gate.wait()
        if self.append_error is not None:
            return Err(FsError(self.append_error, "scripted"))
        result = await self._local.append_file(path, content)
        self.appends_completed += 1
        return result


@dataclass
class Spawned:
    argv: list[str]
    options: SpawnOptions
    process: Process | None = None


@dataclass
class WorldSubprocess:
    """A `ctx.subprocess` over the real `LocalSubprocess`, with a declared world."""

    cwd: str
    platform: Platform = Platform.POSIX
    environment: list[tuple[str, str]] = field(default_factory=list)
    shell_program: Callable[[list[str]], list[str]] | None = None
    """Maps the bash spawn's argv (shell, args..., command) to the argv actually run."""
    lookup_program: list[str] | None = None
    """The argv run for `where bash.exe` / `which bash`; None leaves the real lookup."""
    world: ExecutionWorldIdentity = field(default_factory=ExecutionWorldIdentity.local)
    spawns: list[Spawned] = field(default_factory=list)
    terminates: list[str] = field(default_factory=list)
    on_terminate: Callable[[Process], Any] | None = None

    @property
    def execution_world(self) -> ExecutionWorldIdentity:
        return self.world

    def base_env(self) -> EnvSnapshot:
        return EnvSnapshot(self.environment, self.platform)

    async def spawn(self, argv: Sequence[str], options: SpawnOptions | None = None) -> Any:
        opts = options or SpawnOptions()
        record = Spawned(list(argv), opts)
        self.spawns.append(record)
        run = list(argv)
        if run[:1] in (["where"], ["which"]):
            if self.lookup_program is not None:
                run = self.lookup_program
        elif self.shell_program is not None:
            run = self.shell_program(run)
        result = await LocalSubprocess(self.cwd).spawn(run, opts)
        if isinstance(result, Ok):
            record.process = _Recorded(result.value, self)
            return Ok(record.process)
        return result


class _Recorded:
    """A `Process` proxy recording `terminate()` calls (lifecycle delegation witness)."""

    def __init__(self, process: Process, owner: WorldSubprocess) -> None:
        self._process = process
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._process, name)

    async def terminate(self) -> None:
        self._owner.terminates.append(" ".join(self._owner.spawns[-1].argv[:1]))
        if self._owner.on_terminate is not None:
            await self._owner.on_terminate(self._process)
        await self._process.terminate()
