"""`TEST-BASH-ABORT-TIMING` (#197, WP133-D001): the `builtin-bash-abort-during` scenario's abort
trigger, under a deterministic slow shell start-up (the mechanism of the full-suite flake).

A 600 ms delay before the spawn stands in for Git Bash start-up under load. With the original
400 ms wall-clock trigger the abort lands before `printf` runs, so `before` is lost (the observed
flake); with the `abort_on_output` trigger the abort waits for the command's own output and the
scenario's expectation holds."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from minion_agent.execution import Ok
from minion_agent.execution.subprocess import ExitStatus, SpawnOptions
from minion_agent.runtime import RunAbortController
from minion_agent.tools.builtin import bash_shell

from . import builtin_bash_runner
from .builtin_bash_runner import HOST_PLATFORM, run_builtin_bash_scenario

SCENARIO = (
    Path(__file__).resolve().parents[3]
    / "conformance"
    / "agent"
    / "builtin-bash"
    / "builtin-bash-abort-during.yaml"
)
SLOW_START_S = 0.6


def _scenario() -> dict[str, Any]:
    return yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))


@pytest.fixture
def slow_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtin_bash_runner.LocalSubprocess.spawn

    async def spawn(self: Any, argv: Any, options: Any = None) -> Any:
        await asyncio.sleep(SLOW_START_S)
        return await real(self, argv, options)

    monkeypatch.setattr(builtin_bash_runner.LocalSubprocess, "spawn", spawn)


def test_the_scenario_uses_the_output_trigger() -> None:
    spec = _scenario()["builtin_bash"]
    assert spec["abort_on_output"] == "before"
    assert "abort_after_ms" not in spec


async def test_output_trigger_holds_under_slow_start(slow_spawn: None) -> None:
    outcome = await run_builtin_bash_scenario(_scenario())
    assert outcome is not None
    assert outcome["observed"]["is_error"] is True
    assert outcome["observed"]["text"] == outcome["expected"]["text"]


class _ScriptedStream:
    """An in-memory `ReadableStream`: delivers its chunks, then EOF. No native process."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read_chunk(self) -> Any:
        return Ok(self._chunks.pop(0) if self._chunks else None)

    async def close(self) -> None:
        self._chunks.clear()


class _ScriptedProcess:
    def __init__(self, stdout: list[bytes]) -> None:
        self.stdout = _ScriptedStream(stdout)
        self.stderr = _ScriptedStream([])

    async def wait(self) -> Any:
        return Ok(ExitStatus(exit_code=0))

    async def terminate(self) -> None:
        return None


class _ScriptedSubprocess:
    """Records every spawn; each returns a scripted process emitting `stdout`."""

    def __init__(self, stdout: list[bytes]) -> None:
        self.stdout = stdout
        self.spawns: list[list[str]] = []

    async def spawn(self, argv: Any, options: Any = None) -> Any:
        self.spawns.append(list(argv))
        return Ok(_ScriptedProcess(self.stdout))


async def _drain(process: Any) -> bytes:
    data = b""
    while True:
        chunk = (await process.stdout.read_chunk()).value
        if chunk is None:
            return data
        data += chunk


async def test_lookup_output_never_triggers_the_abort() -> None:
    # WP133D001-R001: the certified shell lookup (`where`/`which`, spawned WITHOUT the call's
    # signal) is not the command; a lookup path containing the marker must not abort the call.
    controller = RunAbortController()
    inner = _ScriptedSubprocess([b"C:\\before\\Git\\bin\\bash.exe\r\n"])
    tap = builtin_bash_runner._AbortOnOutput(inner, b"before", controller)
    found = await bash_shell.lookup(tap, ["where", "bash.exe"])
    await asyncio.sleep(0)  # any scheduled abort would have run by now
    assert found == "C:\\before\\Git\\bin\\bash.exe\r\n"  # ordinary lookup collection kept
    assert inner.spawns == [["where", "bash.exe"]]
    assert controller.signal.aborted is False


async def test_command_output_triggers_the_abort_after_delivery() -> None:
    # The command spawn carries the call's own signal; a marker split across chunks still fires,
    # and only after the tool has taken in the completing chunk (scheduled on the next loop turn).
    controller = RunAbortController()
    inner = _ScriptedSubprocess([b"befo", b"re\n"])
    tap = builtin_bash_runner._AbortOnOutput(inner, b"before", controller)
    spawned = await tap.spawn(["bash", "-c", "x"], SpawnOptions(signal=controller.signal))
    process = spawned.value
    assert (await process.stdout.read_chunk()).value == b"befo"
    assert (await process.stdout.read_chunk()).value == b"re\n"
    assert controller.signal.aborted is False  # delivered first, abort scheduled after
    await asyncio.sleep(0)
    assert controller.signal.aborted is True


async def test_control_tapping_every_spawn_is_caught_by_the_lookup_witness() -> None:
    # Control: the pre-remediation tap watched every spawn. Restored, the lookup witness's own
    # assertion must fail for its intended reason -- the lookup output aborts the call.
    class _AllSpawns(builtin_bash_runner._AbortOnOutput):
        def is_command(self, options: Any) -> bool:
            return True

    controller = RunAbortController()
    inner = _ScriptedSubprocess([b"C:\\before\\Git\\bin\\bash.exe\r\n"])
    found = await bash_shell.lookup(_AllSpawns(inner, b"before", controller), ["where", "bash.exe"])
    await asyncio.sleep(0)
    assert found is not None
    assert controller.signal.aborted is True


async def test_wall_clock_trigger_loses_the_output_under_slow_start(slow_spawn: None) -> None:
    # The pre-WP133-D001 trigger: the flake's mechanism, reproduced deterministically.
    document = copy.deepcopy(_scenario())
    spec = document["builtin_bash"]
    del spec["abort_on_output"]
    spec["abort_after_ms"] = 400
    outcome = await run_builtin_bash_scenario(document)
    assert outcome is not None
    assert outcome["observed"]["text"] != outcome["expected"]["text"]
    assert "before" not in outcome["observed"]["text"]
    assert HOST_PLATFORM in ("win32", "linux")
