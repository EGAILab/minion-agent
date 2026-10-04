"""`TOOL-034`/`TOOL-035`: the `bash` tool's binding witnesses (spec/tools.md WP-13.3 "Witnesses and
evidence" item 3). The shell is a controlled Python child mapped in by `WorldSubprocess` unless a
test runs the host's real bash; the canonical scenarios (`test_builtin_bash_conformance.py`) cover
the real-shell rows."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import FsErrorCode, Platform
from minion_agent.execution.world import ExecutionWorldIdentity
from minion_agent.llm import ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin import bash as bash_module
from minion_agent.tools.builtin.bash import (
    BASH_DESCRIPTION,
    BASH_PARAMETERS,
    IncompatibleExecutionWorldError,
    create_bash_tool,
    resolve_timeout_ms,
    scalar_command,
    scheduled_delay_ms,
    session_environment,
)
from minion_agent.tools.builtin.paths import BuiltinToolError
from minion_agent.tools.definition import ToolExecutionContext
from minion_agent.tools.events import TOOLS_UPDATE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

from .bash_world import PY, WorldFs, WorldSubprocess

SHELL = "/fake/bash"


def _echo_shell(code: str) -> Any:
    """Map the bash spawn to a Python child running `code`; the command is argv[-1]."""

    def mapped(argv: list[str]) -> list[str]:
        return [PY, "-c", code, *argv[1:]]

    return mapped


def _tool(tmp_path: Path, code: str, **world: Any) -> tuple[Any, WorldSubprocess, WorldFs]:
    fs = WorldFs(str(tmp_path), existing={SHELL, str(tmp_path)}, **world.pop("fs", {}))
    subprocess = WorldSubprocess(str(tmp_path), shell_program=_echo_shell(code), **world)
    return create_bash_tool(fs, subprocess, shell_path=SHELL), subprocess, fs  # type: ignore[arg-type]


async def _run(
    tool: Any,
    arguments: dict[str, Any],
    *,
    signal: Any = None,
    context: ToolExecutionContext | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    try:
        result = await tool.execute("c", arguments, signal, context=context)
    except BuiltinToolError as error:
        return True, str(error), {}
    return False, result.content[0].text, result.details


# ---- definition and factory ----


def test_definition_strings_are_pinned_pi() -> None:
    assert BASH_DESCRIPTION == (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        "Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, "
        "full output is saved to a temp file. Optionally provide a timeout in seconds."
    )
    assert BASH_PARAMETERS == {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, no default timeout)",
            },
        },
        "required": ["command"],
    }


def test_factory_surface_is_owner_q3(tmp_path: Path) -> None:
    fs = WorldFs(str(tmp_path))
    subprocess = WorldSubprocess(str(tmp_path))
    tool = create_bash_tool(fs, subprocess, shell_path=None, expose_session_environment=False)  # type: ignore[arg-type]
    assert (tool.name, tool.label, tool.wants_signal, tool.wants_context) == (
        "bash",
        "bash",
        True,
        True,
    )
    with pytest.raises(TypeError):
        create_bash_tool(fs, subprocess, command_prefix="x")  # type: ignore[arg-type, call-arg]
    with pytest.raises(TypeError):
        create_bash_tool(fs, subprocess, spawn_hook=lambda c: c)  # type: ignore[arg-type, call-arg]


def test_incompatible_worlds_fail_activation(tmp_path: Path) -> None:
    fs = WorldFs(str(tmp_path), world=ExecutionWorldIdentity("remote-a"))
    with pytest.raises(IncompatibleExecutionWorldError) as caught:
        create_bash_tool(fs, WorldSubprocess(str(tmp_path)))  # type: ignore[arg-type]
    pair = caught.value.error.incompatible_pairs[0]
    assert (pair.left, pair.right) == ("fs", "subprocess")


# ---- environment (Owner Q1, WP-12.E4 composition, L0506-D004 context) ----

_ENV_DUMP = "import json, os, sys; sys.stdout.write(json.dumps(dict(os.environ), sort_keys=True))"
_STALE = [
    ("MINION_SESSION_ID", "stale"),
    ("MINION_SESSION_FILE", "stale"),
    ("MINION_PROVIDER", "stale"),
    ("MINION_MODEL", "stale"),
    ("MINION_REASONING_LEVEL", "stale"),
    ("PI_SESSION_ID", "pi-untouched"),
    ("UNRELATED", "kept"),
]


async def _environment(tmp_path: Path, **kwargs: Any) -> dict[str, str]:
    context = kwargs.pop("context", None)
    environment = kwargs.pop("environment", _STALE)
    platform = kwargs.pop("platform", Platform.POSIX)
    fs = WorldFs(str(tmp_path), existing={SHELL, str(tmp_path)})
    world = WorldSubprocess(
        str(tmp_path),
        platform=platform,
        environment=environment,
        shell_program=_echo_shell(_ENV_DUMP),
    )
    tool = create_bash_tool(fs, world, shell_path=SHELL, **kwargs)  # type: ignore[arg-type]
    await _run(tool, {"command": "env"}, context=context)
    spawn = world.spawns[-1]
    assert spawn.options.inherit_env is False
    assert spawn.options.env is not None
    return spawn.options.env


async def test_session_variables_replace_stale_ones(tmp_path: Path) -> None:
    context = ToolExecutionContext("sess-1", provider="openai", model="gpt", reasoning_level="off")
    env = await _environment(tmp_path, context=context)
    assert env["MINION_SESSION_ID"] == "sess-1"
    assert (env["MINION_PROVIDER"], env["MINION_MODEL"]) == ("openai", "gpt")
    assert env["MINION_REASONING_LEVEL"] == "off"
    assert "MINION_SESSION_FILE" not in env  # absent today, never injected as "" or "none"
    assert (env["PI_SESSION_ID"], env["UNRELATED"]) == ("pi-untouched", "kept")


async def test_disabled_or_absent_context_removes_and_injects_nothing(tmp_path: Path) -> None:
    for kwargs in ({"expose_session_environment": False, "context": ToolExecutionContext("s")}, {}):
        env = await _environment(tmp_path, **kwargs)  # type: ignore[arg-type]
        assert not [name for name in env if name.startswith("MINION_")]
        assert env["UNRELATED"] == "kept"


def test_partial_context_injection_rules() -> None:
    assert session_environment(ToolExecutionContext("s", provider="p")) == {
        "MINION_SESSION_ID": "s"
    }
    assert session_environment(ToolExecutionContext("s", reasoning_level="")) == {
        "MINION_SESSION_ID": "s"
    }
    assert session_environment(ToolExecutionContext("s", session_file="/f")) == {
        "MINION_SESSION_ID": "s",
        "MINION_SESSION_FILE": "/f",
    }
    assert session_environment(None) == {}


async def test_case_variant_survives_exact_removal_and_injection_wins_on_windows(
    tmp_path: Path,
) -> None:
    """Owner Q1/C003: removal is by exact spelling; on WINDOWS the injected upper-case name wins
    Node's arbitration, and with nothing injected the variant survives."""
    variant = [("Minion_Session_Id", "variant"), ("UNRELATED", "kept")]
    injected = await _environment(
        tmp_path, platform=Platform.WINDOWS, environment=variant, context=ToolExecutionContext("s")
    )
    assert injected.get("MINION_SESSION_ID") == "s" and "Minion_Session_Id" not in injected
    survived = await _environment(tmp_path, platform=Platform.WINDOWS, environment=variant)
    assert survived.get("Minion_Session_Id") == "variant"
    posix = await _environment(tmp_path, environment=variant)
    assert posix.get("Minion_Session_Id") == "variant"


# ---- command projection (WP133-AUD-R001) ----

_ARGV_DUMP = "import json, sys; sys.stdout.write(json.dumps([ord(c) for c in sys.argv[-1]]))"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("a\ud800b", [0x61, 0xFFFD, 0x62]),
        ("a\udc80b", [0x61, 0xFFFD, 0x62]),
        ("a\ude00\ud83db", [0x61, 0xFFFD, 0xFFFD, 0x62]),
        ("a\ud83d\ude00b", [0x61, 0x1F600, 0x62]),
        ("a\U0001f600b", [0x61, 0x1F600, 0x62]),
    ],
)
def test_command_scalar_projection(command: str, expected: list[int]) -> None:
    assert [ord(char) for char in scalar_command(command)] == expected


async def test_projection_reaches_argv(tmp_path: Path) -> None:
    tool, world, _ = _tool(tmp_path, _ARGV_DUMP)
    _, text, _ = await _run(tool, {"command": "x\ud800y"})
    assert json.loads(text) == [0x78, 0xFFFD, 0x79]
    assert world.spawns[-1].argv == [SHELL, "-c", "x\ufffdy"]


async def test_legacy_wsl_shell_reads_the_projected_command_from_stdin(tmp_path: Path) -> None:
    wsl = "C:/Windows/System32/bash.exe"
    reader = "import sys; sys.stdout.write(sys.stdin.buffer.read().hex())"
    fs = WorldFs(str(tmp_path), existing={wsl, str(tmp_path)})
    world = WorldSubprocess(str(tmp_path), shell_program=_echo_shell(reader))
    tool = create_bash_tool(fs, world, shell_path=wsl)  # type: ignore[arg-type]
    _, text, _ = await _run(tool, {"command": "e\udc80"})
    assert bytes.fromhex(text) == "e\ufffd".encode()
    assert world.spawns[-1].argv == [wsl, "-s"]


async def test_stdin_child_that_never_reads_still_times_out(tmp_path: Path) -> None:
    wsl = "C:/Windows/System32/bash.exe"
    fs = WorldFs(str(tmp_path), existing={wsl, str(tmp_path)})
    world = WorldSubprocess(str(tmp_path), shell_program=_echo_shell("import time; time.sleep(30)"))
    tool = create_bash_tool(fs, world, shell_path=wsl)  # type: ignore[arg-type]
    started = time.monotonic()
    failed, text, _ = await _run(tool, {"command": "x" * 200000, "timeout": 0.3})
    assert failed and text == "Command timed out after 0.3 seconds"
    assert time.monotonic() - started < 10


# ---- pre-spawn order (pinned exec) ----


async def test_invalid_timeout_before_abort(tmp_path: Path) -> None:
    tool, world, _ = _tool(tmp_path, "pass")
    controller = RunAbortController()
    controller.abort()
    failed, text, _ = await _run(tool, {"command": "x", "timeout": 0}, signal=controller.signal)
    assert failed and text == "Invalid timeout: must be a finite number of seconds"
    assert world.spawns == []


async def test_abort_before_shell_selection(tmp_path: Path) -> None:
    fs = WorldFs(str(tmp_path), existing=set())
    world = WorldSubprocess(str(tmp_path))
    tool = create_bash_tool(fs, world, shell_path="/missing")  # type: ignore[arg-type]
    controller = RunAbortController()
    controller.abort()
    failed, text, _ = await _run(tool, {"command": "x"}, signal=controller.signal)
    assert failed and text == "Command aborted"
    assert fs.calls == []


async def test_shell_error_before_cwd_check(tmp_path: Path) -> None:
    fs = WorldFs(str(tmp_path), existing=set())
    world = WorldSubprocess(str(tmp_path / "missing"))
    tool = create_bash_tool(fs, world, shell_path="/missing")  # type: ignore[arg-type]
    failed, text, _ = await _run(tool, {"command": "x"})
    assert failed and text == "Custom shell path not found: /missing"


async def test_abort_arriving_during_shell_selection_is_command_aborted(tmp_path: Path) -> None:
    """Step 3 is reachable when the abort arrives after preflight: here during the cwd check."""
    controller = RunAbortController()
    fs = WorldFs(str(tmp_path), existing={SHELL, str(tmp_path)})
    original = fs.probe_dir_entry

    async def abort_then_probe(path: str, signal: Any = None) -> Any:
        controller.abort()
        return await original(path)

    fs.probe_dir_entry = abort_then_probe  # type: ignore[method-assign]
    world = WorldSubprocess(str(tmp_path), shell_program=_echo_shell("import time; time.sleep(5)"))
    tool = create_bash_tool(fs, world, shell_path=SHELL)  # type: ignore[arg-type]
    failed, text, _ = await _run(tool, {"command": "x"}, signal=controller.signal)
    assert failed and text == "Command aborted"


# ---- cwd check (CE-WP133-01) ----


async def test_cwd_check_uses_followed_probe_on_posix_and_file_info_on_windows(
    tmp_path: Path,
) -> None:
    for platform, operation in (
        (Platform.POSIX, "probe_dir_entry"),
        (Platform.WINDOWS, "file_info"),
    ):
        fs = WorldFs(str(tmp_path), existing={SHELL})
        world = WorldSubprocess(str(tmp_path / "gone"), platform=platform)
        tool = create_bash_tool(fs, world, shell_path=SHELL)  # type: ignore[arg-type]
        failed, text, _ = await _run(tool, {"command": "x"})
        assert failed
        assert text == (
            f"Working directory does not exist: {tmp_path / 'gone'}\nCannot execute bash commands."
        )
        assert fs.calls[-1] == f"{operation} {tmp_path / 'gone'}"


@pytest.mark.parametrize(
    ("platform", "operation", "fs_kwargs"),
    [
        (Platform.POSIX, "probe_dir_entry", {"probe_error": FsErrorCode.NOT_SUPPORTED}),
        (Platform.WINDOWS, "file_info", {"file_info_error": FsErrorCode.NOT_SUPPORTED}),
    ],
)
async def test_not_supported_is_a_prerequisite_error(
    tmp_path: Path, platform: Platform, operation: str, fs_kwargs: dict[str, Any]
) -> None:
    fs = WorldFs(str(tmp_path), existing={SHELL, str(tmp_path)}, **fs_kwargs)
    world = WorldSubprocess(str(tmp_path), platform=platform)
    tool = create_bash_tool(fs, world, shell_path=SHELL)  # type: ignore[arg-type]
    if operation == "probe_dir_entry":
        # the shell check itself uses probe_dir_entry: not_supported reads as "not found" there
        failed, text, _ = await _run(tool, {"command": "x"})
        assert failed and text == f"Custom shell path not found: {SHELL}"
        return
    failed, text, _ = await _run(tool, {"command": "x"})
    assert failed and text == f"bash requires a filesystem provider that supports {operation}"


# ---- timer (WP133-CON-R003) ----


@pytest.mark.parametrize(
    ("seconds", "delay"),
    [
        (0.0005, 1),
        (0.000999, 1),
        (0.001, 1),
        (0.0019, 1),
        (0.0025, 2),
        (0.25, 250),
        (2147483.647, 2147483647),
    ],
)
def test_scheduled_delay_is_node_normalized(seconds: float, delay: int) -> None:
    assert scheduled_delay_ms(seconds * 1000) == delay


async def test_timer_is_scheduled_for_the_normalized_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed through a controllable timer, never by wall-clock latency."""
    delays: list[float] = []
    loop = asyncio.get_running_loop()
    original = loop.call_later

    def recording(delay: float, callback: Any, *args: Any) -> Any:
        delays.append(delay)
        return original(delay, callback, *args)

    monkeypatch.setattr(loop, "call_later", recording)
    tool, _, _ = _tool(tmp_path, "pass")
    await _run(tool, {"command": "x", "timeout": 0.0019})
    assert 0.001 in delays and 0.0019 not in delays


def test_timeout_validation() -> None:
    assert resolve_timeout_ms(None) is None
    assert resolve_timeout_ms(2147483.647) == 2147483647.0
    for bad in (0, -0.0, -1, math.inf, -math.inf, math.nan):
        with pytest.raises(
            BuiltinToolError, match=r"^Invalid timeout: must be a finite number of seconds$"
        ):
            resolve_timeout_ms(bad)
    with pytest.raises(
        BuiltinToolError, match=r"^Invalid timeout: maximum is 2147483\.647 seconds$"
    ):
        resolve_timeout_ms(2147483.648)


# ---- run, settlement and classification ----


async def test_signal_is_classified_before_the_timeout(tmp_path: Path) -> None:
    """Classification checks the signal FIRST: when the timeout fired AND the call was aborted
    before the result, the outcome is "Command aborted", not the timeout."""
    controller = RunAbortController()

    async def abort_too(_process: Any) -> None:
        controller.abort()  # the timer's terminate() also aborts the call

    tool, world, _ = _tool(
        tmp_path,
        "import sys, time; sys.stdout.write('p'); sys.stdout.flush(); time.sleep(30)",
        on_terminate=abort_too,
    )
    failed, text, _ = await _run(tool, {"command": "x", "timeout": 0.3}, signal=controller.signal)
    assert world.terminates  # the timer fired
    assert failed and text == "p\n\nCommand aborted"


async def test_timeout_during_output(tmp_path: Path) -> None:
    tool, world, _ = _tool(
        tmp_path,
        "import sys, time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(30)",
    )
    failed, text, _ = await _run(tool, {"command": "x", "timeout": 0.5})
    assert failed and text == "partial\n\nCommand timed out after 0.5 seconds"
    assert world.terminates == [SHELL]  # the certified Layer-12 tree kill


async def test_abort_during_the_post_exit_grace_is_aborted(tmp_path: Path) -> None:
    """Classification checks the signal FIRST: an abort after a clean exit, while the 100 ms grace
    is still waiting on a descendant, is "aborted"."""
    controller = RunAbortController()
    started = tmp_path / "descendant-started"
    # Event-driven, not timed: the parent writes 'a' and exits only once the descendant is running;
    # the descendant then writes every 20 ms (well inside the 100 ms grace) for 2 s, so the abort
    # issued after it started lands after the parent's clean exit and inside the grace.
    descendant = (
        "import pathlib, sys, time\n"
        f"pathlib.Path({str(started)!r}).write_text('x')\n"
        "for _ in range(100):\n"
        "    time.sleep(0.02); sys.stdout.write('b'); sys.stdout.flush()\n"
    )
    code = (
        "import os, pathlib, subprocess, sys, time\n"
        "sys.stdout.write('a'); sys.stdout.flush()\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}], stdin=subprocess.DEVNULL,"
        " stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno(),"
        " start_new_session=(os.name != 'nt'))\n"
        f"while not pathlib.Path({str(started)!r}).exists():\n"
        "    time.sleep(0.01)\n"
        "os._exit(0)\n"
    )
    tool, _, _ = _tool(tmp_path, code)

    async def abort_once_the_descendant_runs() -> None:
        while not started.exists():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)  # the parent has exited; the descendant keeps the grace armed
        controller.abort()

    trigger = asyncio.ensure_future(abort_once_the_descendant_runs())
    failed, text, _ = await _run(tool, {"command": "x"}, signal=controller.signal)
    await trigger
    assert failed and text.startswith("ab") and text.endswith("\n\nCommand aborted")


async def test_detached_descendant_holding_the_pipes_settles_after_the_grace(
    tmp_path: Path,
) -> None:
    code = (
        "import os, subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'],"
        " stdin=subprocess.DEVNULL, stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno(),"
        " start_new_session=(os.name != 'nt'))\n"
        "sys.stdout.write('x'); sys.stdout.flush(); os._exit(0)\n"
    )
    tool, _, _ = _tool(tmp_path, code)
    started = time.monotonic()
    failed, text, _ = await _run(tool, {"command": "x"})
    assert (failed, text) == (False, "x")
    assert time.monotonic() - started < 3


async def test_a_background_job_survives_settlement_and_meets_released_pipes(
    tmp_path: Path,
) -> None:
    """spec/execution.md section 16.5 #12 (`npm run dev &`): the foreground shell completes and
    the result settles; the background descendant is NOT killed (it records that it is alive) and
    bash has released its read ends (the descendant's next write fails); its late output is not
    part of the result."""
    alive = tmp_path / "alive"
    outcome = tmp_path / "outcome"
    descendant = (
        "import pathlib, sys, time\n"
        "time.sleep(0.8)\n"
        f"pathlib.Path({str(alive)!r}).write_text('yes')\n"
        "try:\n"
        "    sys.stdout.write('late\\n' * 1000); sys.stdout.flush(); result = 'wrote'\n"
        "except OSError:\n"
        "    result = 'closed'\n"
        f"pathlib.Path({str(outcome)!r}).write_text(result)\n"
    )
    code = (
        "import os, subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}], stdin=subprocess.DEVNULL,"
        " stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno(),"
        " start_new_session=(os.name != 'nt'))\n"
        "sys.stdout.write('x'); sys.stdout.flush(); os._exit(0)\n"
    )
    tool, _, _ = _tool(tmp_path, code)
    failed, text, _ = await _run(tool, {"command": "npm run dev &"})
    assert (failed, text) == (False, "x")
    deadline = time.monotonic() + 10
    while not outcome.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert alive.read_text() == "yes"
    assert outcome.read_text() == "closed"


async def test_output_written_shortly_after_exit_is_kept(tmp_path: Path) -> None:
    """Settlement waits for the grace, not `wait()` alone: a descendant writing shortly after the
    parent exits is included. Event-driven: the parent exits only once the descendant runs, and
    the descendant writes 30 ms later -- inside the 100 ms grace whatever its startup time."""
    started = tmp_path / "descendant-started"
    descendant = (
        "import pathlib, sys, time\n"
        f"pathlib.Path({str(started)!r}).write_text('x')\n"
        "time.sleep(0.03); sys.stdout.write('late'); sys.stdout.flush()\n"
    )
    code = (
        "import os, pathlib, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}], stdin=subprocess.DEVNULL,"
        " stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno(),"
        " start_new_session=(os.name != 'nt'))\n"
        f"while not pathlib.Path({str(started)!r}).exists():\n"
        "    time.sleep(0.005)\n"
        "os._exit(0)\n"
    )
    tool, _, _ = _tool(tmp_path, code)
    failed, text, _ = await _run(tool, {"command": "x"})
    assert (failed, text) == (False, "late")


async def test_external_kill_without_exit_code_is_success(tmp_path: Path) -> None:
    tool, _, _ = _tool(tmp_path, "pass")

    async def no_code(self: Any) -> int | None:
        return None

    tool_result = await _run(tool, {"command": "x"})
    assert tool_result[0] is False


# ---- error paths ----


async def test_spawn_failure_names_the_shell(tmp_path: Path) -> None:
    fs = WorldFs(str(tmp_path), existing={str(tmp_path / "no-shell"), str(tmp_path)})
    world = WorldSubprocess(str(tmp_path))
    tool = create_bash_tool(fs, world, shell_path=str(tmp_path / "no-shell"))  # type: ignore[arg-type]
    failed, text, _ = await _run(tool, {"command": "x"})
    assert failed and text == f"Failed to start the shell {tmp_path / 'no-shell'}"


_BIG = "import sys; sys.stdout.write('y' * 60000)"


@pytest.mark.parametrize(
    ("fs_kwargs", "cause"),
    [
        ({"create_temp_error": FsErrorCode.PERMISSION_DENIED}, "permission denied"),
        ({"append_error": FsErrorCode.UNKNOWN}, "unknown filesystem error"),
    ],
)
async def test_full_output_file_failure(
    tmp_path: Path, fs_kwargs: dict[str, Any], cause: str
) -> None:
    tool, _, _ = _tool(tmp_path, _BIG, fs=fs_kwargs)
    failed, text, _ = await _run(tool, {"command": "x"})
    assert failed and text == f"Cannot write the full-output file: {cause}"


async def test_full_output_file_holds_the_raw_bytes(tmp_path: Path) -> None:
    raw = b"\xff" * 30000 + b"z" * 30000
    tool, _, _ = _tool(
        tmp_path, "import sys; sys.stdout.buffer.write(bytes([255]) * 30000 + b'z' * 30000)"
    )
    failed, _, details = await _run(tool, {"command": "x"})
    assert not failed
    path = details["fullOutputPath"]
    assert Path(path).name.startswith("minion-bash-") and path.endswith(".log")
    assert Path(path).read_bytes() == raw
    assert details["truncation"]["truncated"] is True
    Path(path).unlink()


async def test_pipe_failure_ends_that_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minion_agent.execution import Err
    from minion_agent.execution.errors import SubprocessError, SubprocessErrorCode
    from minion_agent.execution.subprocess import ReadableStream

    async def broken(self: Any) -> Any:
        return Err(SubprocessError(SubprocessErrorCode.PIPE_ERROR, "broken"))

    monkeypatch.setattr(ReadableStream, "read_chunk", broken)
    tool, _, _ = _tool(tmp_path, "import sys; sys.stdout.write('lost')")
    failed, text, _ = await _run(tool, {"command": "x"})
    assert (failed, text) == (False, "(no output)")


# ---- through Layer 06: zero partial updates ----


async def test_zero_partial_updates_with_the_final_result(tmp_path: Path) -> None:
    tool, _, _ = _tool(
        tmp_path,
        "import sys, time; sys.stdout.write('a'); sys.stdout.flush(); time.sleep(0.3);"
        " sys.stdout.write('b')",
    )
    registry = ToolRegistry()
    registry.register(tool)
    ctx = Context()
    declare_tools_events(ctx.events)
    updates: list[Any] = []
    ctx.events.on(TOOLS_UPDATE, lambda *args: updates.append(args))
    result = await execute_call(
        ToolCallBlock(id="c1", name="bash", arguments={"command": "x"}), registry=registry, ctx=ctx
    )
    assert result.content[0].text == "ab"  # type: ignore[union-attr]
    assert updates == []


def test_module_constants() -> None:
    assert bash_module.EXIT_STDIO_GRACE_S == 0.1
    assert bash_module.MAX_TIMEOUT_MS == 2_147_483_647
    assert (bash_module.FULL_OUTPUT_PREFIX, bash_module.FULL_OUTPUT_SUFFIX) == (
        "minion-bash-",
        ".log",
    )
