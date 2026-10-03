"""`TOOL-034` shell selection and the `where`/`which` lookup (spec/tools.md WP-13.3 "Shell selection"
and "The lookup"; `WP133-CON-R005`/`R006`, `CE-WP133-02`, `DIV-001`).

Expectations come from the pinned-Pi authority rows in `data/wp133/` (Minion's expectation for the
lifecycle rows, `minionExpected`). The lookup programs are Python children reproducing the probe's
Node programs; the 5000 ms limit is patched down (timings scaled with it) to keep the suite fast --
`test_lookup_constants` pins the real values.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import Platform
from minion_agent.tools.builtin import bash_shell
from minion_agent.tools.builtin.bash_shell import (
    ShellConfig,
    ShellNotFoundError,
    is_legacy_wsl_bash_path,
    js_trim,
    select_shell,
    shell_config,
)

from .bash_world import WorldFs, WorldSubprocess, child

DATA = Path(__file__).parent / "data" / "wp133"
HOST = "win32" if os.name == "nt" else "linux"
LIMIT = 1.0  # the patched lookup limit, standing for 5000 ms


def _load(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture
def short_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bash_shell, "LOOKUP_TIMEOUT_S", LIMIT)


def test_lookup_constants() -> None:
    assert bash_shell.LOOKUP_TIMEOUT_S == 5.0
    assert bash_shell.LOOKUP_OUTPUT_BUDGET == 1024 * 1024


def _units(selected: str | None) -> str | None:
    if selected is None:
        return None
    return " ".join(format(ord(char), "04X") for char in selected)


# ---- the lookup's output budget and decoding (lookup-win32.json) ----

BUDGET_ROWS = {
    "stdoutAtBudget", "stdoutOverBudget", "stderrAtBudget", "stderrOverBudget",
    "combinedAtBudget", "combinedOverBudget", "bomThenPath", "invalidByteInPath",
}  # fmt: skip


def _budget_program(row: dict[str, Any], path: bytes, name: str) -> list[str]:
    prefix = path
    if name == "bomThenPath":
        prefix = b"\xef\xbb\xbf" + path
    elif name == "invalidByteInPath":
        prefix = path[:3] + b"\xff" + path[3:]
    stdout_fill = row["stdoutBytes"] - len(prefix)
    return child(
        "import sys\n"
        f"sys.stdout.buffer.write(bytes.fromhex({prefix.hex()!r}) + b'x' * {stdout_fill})\n"
        "sys.stdout.flush()\n"
        f"sys.stderr.buffer.write(b'y' * {row['stderrBytes']})\n"
        "sys.stderr.flush()\n"
    )


@pytest.mark.parametrize("name", sorted(BUDGET_ROWS))
async def test_lookup_budget_and_decoding_rows(name: str, tmp_path: Path) -> None:
    data = _load("lookup-win32.json")
    row = data["rows"][name]
    path = b"C:/valid/bash.exe\n"
    world = WorldSubprocess(str(tmp_path), lookup_program=_budget_program(row, path, name))
    found = await bash_shell.lookup(world, ["which", "bash"])
    first = None if found is None else bash_shell._LINE_SPLIT.split(js_trim(found))[0] or None
    assert _units(first) == row["selected"]


async def test_lookup_budget_is_combined_not_per_stream(tmp_path: Path) -> None:
    """The per-stream control: 18 + 1048559 bytes over two streams exceed the one budget."""
    world = WorldSubprocess(
        str(tmp_path),
        lookup_program=child(
            "import sys\nsys.stdout.write('/valid/bash\\n'); sys.stdout.flush()\n"
            "sys.stderr.buffer.write(b'y' * (1048576 - 12 + 1)); sys.stderr.flush()\n"
            "import time; time.sleep(2)\n"
        ),
    )
    assert await bash_shell.lookup(world, ["which", "bash"]) is None


# ---- the lookup lifecycle (lookup-lifecycle-<host>.json, minionExpected) ----

LIFECYCLE_PATH = "C:/valid/bash.exe" if HOST == "win32" else "/valid/bash"


def _program(parent: list[tuple[float, str]], descendant: list[tuple[float, str]] | None) -> list[str]:
    """Python equivalents of lookup_lifecycle_probe.mjs's step lists; times scaled to LIMIT."""

    def steps(items: list[tuple[float, str]]) -> str:
        out = []
        for delay, action in items:
            out.append(f"time.sleep({delay})")
            if action == "path":
                out.append(f"sys.stdout.write({LIFECYCLE_PATH + chr(10)!r}); sys.stdout.flush()")
            elif action == "flood":
                out.append(
                    "try:\n    sys.stderr.buffer.write(b'y' * 2097152); sys.stderr.flush()\n"
                    "except OSError:\n    pass"
                )
            elif action.startswith("exit:"):
                out.append(f"os._exit({int(action[5:])})")
            elif action.startswith("trap:"):
                _, code, delay_ms = action.split(":")
                out.append(
                    "if hasattr(signal, 'SIGTERM') and os.name != 'nt':\n"
                    f"    signal.signal(signal.SIGTERM, lambda *a: (time.sleep({int(delay_ms) / 1000}), os._exit({code})))"
                )
            elif action == "end":
                out.append("os._exit(0)")
        return "\n".join(out)

    head = "import os, signal, subprocess, sys, time\n"
    spawn = ""
    if descendant is not None:
        code = head + steps(descendant)
        spawn = (
            f"subprocess.Popen([sys.executable, '-c', {code!r}], stdin=subprocess.DEVNULL,"
            " stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno(),"
            " start_new_session=(os.name != 'nt'))\n"
        )
    return child(head + spawn + steps(parent))


S = LIMIT / 5.0  # one probe millisecond, scaled
LIFECYCLE: dict[str, list[str]] = {
    "parentPathExit0": _program([(0, "path"), (0, "exit:0")], None),
    "parentPathExit1": _program([(0, "path"), (0, "exit:1")], None),
    "descendantPathAfterExit": _program([(0, "exit:0")], [(0.25, "path"), (0, "end")]),
    "descendantPathAfterExitParentExit1": _program([(0, "exit:1")], [(0.25, "path"), (0, "end")]),
    "overflowWhileAlive": _program([(0, "path"), (0, "flood"), (0, "exit:0")], None),
    "overflowAfterExit": _program([(0, "path"), (0, "exit:0")], [(0.25, "flood"), (0, "end")]),
    "overflowAfterExitParentExit1": _program(
        [(0, "path"), (0, "exit:1")], [(0.25, "flood"), (0, "end")]
    ),
    "timeoutWhileAlive": _program([(0, "path"), (LIMIT + 0.4, "exit:0")], None),
    "timeoutAfterExit": _program([(0, "path"), (0, "exit:0")], [(LIMIT + 0.4, "end")]),
    "descendantPathAfterTimeout": _program([(0, "exit:0")], [(LIMIT + 0.4, "path"), (0, "end")]),
    "descendantPathThenOverflow": _program(
        [(0, "exit:0")], [(0.25, "path"), (0.1, "flood"), (0, "end")]
    ),
    "trapExit0OverflowWhileAlive": _program(
        [(0, "trap:0:0"), (0, "path"), (0, "flood"), (LIMIT + 1, "exit:3")], None
    ),
    "trapExit0TimeoutWhileAlive": _program([(0, "trap:0:0"), (0, "path"), (LIMIT + 1, "exit:3")], None),
    "trapExit7TimeoutWhileAlive": _program([(0, "trap:7:0"), (0, "path"), (LIMIT + 1, "exit:3")], None),
    "trapDelayedExit0TimeoutWhileAlive": _program(
        [(0, "trap:0:300"), (0, "path"), (LIMIT + 1, "exit:3")], None
    ),
}


@pytest.mark.parametrize("name", sorted(LIFECYCLE))
async def test_lookup_lifecycle_rows(name: str, tmp_path: Path, short_limit: None) -> None:
    """Minion's rule on every row (Pi's own, except the DIV-001 rows): settlement at exit plus EOF
    with no idle grace, the combined budget, the timer, and an interruption of an unexited lookup
    by the certified `terminate()` -- never a direct `SIGTERM`."""
    expected = _load(f"lookup-lifecycle-{HOST}.json")["rows"][name]["minionExpected"]
    world = WorldSubprocess(str(tmp_path), lookup_program=LIFECYCLE[name])
    config = await select_shell(WorldFs(str(tmp_path), existing=set()), world, None)
    assert config.shell == (expected if expected is not None else "sh")


async def test_div001_interruption_is_terminate_not_sigterm(
    tmp_path: Path, short_limit: None
) -> None:
    """DIV-001: a live lookup is interrupted through the certified `terminate()` (recorded), so a
    lookup that would handle `SIGTERM` and exit 0 is not selected."""
    world = WorldSubprocess(str(tmp_path), lookup_program=LIFECYCLE["trapExit0TimeoutWhileAlive"])
    config = await select_shell(WorldFs(str(tmp_path), existing=set()), world, None)
    assert world.terminates == ["which"]
    assert config.shell == "sh"


async def test_c002_natural_exit_racing_the_kill_keeps_its_code(
    tmp_path: Path, short_limit: None
) -> None:
    """`CE-WP133-02-C002`: the lookup completes naturally with 0 after the interruption decision
    and before the kill reaches it -- the real code stands and the path is selected."""
    world = WorldSubprocess(
        str(tmp_path),
        lookup_program=child(
            "import sys\nsys.stdout.write('/valid/bash\\n'); sys.stdout.flush()\n"
            "sys.stdin.buffer.read(1); sys.exit(0)\n"
        ),
    )

    async def natural_exit_first(process: Any) -> None:
        await process._proc.stdin.drain() if process._proc.stdin else None  # noqa: SLF001
        if process._proc.stdin is not None:  # noqa: SLF001
            process._proc.stdin.write(b"x")  # noqa: SLF001
        await process._proc.wait()  # noqa: SLF001

    world.on_terminate = natural_exit_first
    from minion_agent.execution.subprocess import SpawnOptions, StdioMode

    original = world.spawn

    async def spawn_with_stdin(argv: Any, options: SpawnOptions | None = None) -> Any:
        opts = options or SpawnOptions()
        return await original(argv, SpawnOptions(inherit_env=opts.inherit_env, stdin=StdioMode.PIPED))

    world.spawn = spawn_with_stdin  # type: ignore[method-assign]
    config = await select_shell(WorldFs(str(tmp_path), existing=set()), world, None)
    assert world.terminates == ["which"]
    assert config.shell == "/valid/bash"


# ---- the selection matrix over a declared world ----


async def test_custom_shell_path(tmp_path: Path) -> None:
    fs = WorldFs(str(tmp_path), existing={"/opt/bash"})
    world = WorldSubprocess(str(tmp_path))
    assert await select_shell(fs, world, "/opt/bash") == ShellConfig("/opt/bash", ("-c",), "argv")
    with pytest.raises(ShellNotFoundError, match="^Custom shell path not found: /missing$"):
        await select_shell(fs, world, "/missing")


async def test_empty_shell_path_falls_through_to_discovery(tmp_path: Path) -> None:
    fs = WorldFs(str(tmp_path), existing={"/bin/bash"})
    assert (await select_shell(fs, WorldSubprocess(str(tmp_path)), "")).shell == "/bin/bash"


WINDOWS_ENV = [("ProgramFiles", "C:\\PF"), ("ProgramFiles(x86)", "C:\\PF86")]


async def test_windows_git_bash_candidates_in_order(tmp_path: Path) -> None:
    first = "C:\\PF\\Git\\bin\\bash.exe"
    second = "C:\\PF86\\Git\\bin\\bash.exe"
    world = WorldSubprocess(str(tmp_path), platform=Platform.WINDOWS, environment=WINDOWS_ENV)
    both = WorldFs(str(tmp_path), existing={first, second})
    assert (await select_shell(both, world, None)).shell == first
    only_x86 = WorldFs(str(tmp_path), existing={second})
    assert (await select_shell(only_x86, world, None)).shell == second
    assert world.spawns == []


async def test_windows_baseline_lookup_is_the_native_comparison(tmp_path: Path) -> None:
    world = WorldSubprocess(
        str(tmp_path), platform=Platform.WINDOWS, environment=[("PROGRAMFILES", "C:\\Up")]
    )
    fs = WorldFs(str(tmp_path), existing={"C:\\Up\\Git\\bin\\bash.exe"})
    assert (await select_shell(fs, world, None)).shell == "C:\\Up\\Git\\bin\\bash.exe"


async def test_windows_where_lookup_first_line_must_exist(tmp_path: Path) -> None:
    lookup = child("import sys; sys.stdout.write('\\ufeff  D:\\\\b\\\\bash.exe\\r\\nE:\\\\other\\r\\n')")
    world = WorldSubprocess(
        str(tmp_path), platform=Platform.WINDOWS, environment=WINDOWS_ENV, lookup_program=lookup
    )
    found = WorldFs(str(tmp_path), existing={"D:\\b\\bash.exe"})
    assert (await select_shell(found, world, None)).shell == "D:\\b\\bash.exe"
    assert world.spawns[-1].argv == ["where", "bash.exe"]
    assert world.spawns[-1].options.inherit_env is True
    missing = WorldFs(str(tmp_path), existing=set())
    with pytest.raises(ShellNotFoundError) as caught:
        await select_shell(missing, world, None)
    assert str(caught.value) == (
        "No bash shell found. Options:\n"
        "  1. Install Git for Windows: https://git-scm.com/download/win\n"
        "  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n"
        "  3. Set shellPath in settings.json\n\n"
        "Searched Git Bash in:\n  C:\\PF\\Git\\bin\\bash.exe\n  C:\\PF86\\Git\\bin\\bash.exe"
    )


async def test_windows_without_program_files_lists_no_candidates(tmp_path: Path) -> None:
    world = WorldSubprocess(
        str(tmp_path), platform=Platform.WINDOWS, lookup_program=child("import sys; sys.exit(1)")
    )
    with pytest.raises(ShellNotFoundError) as caught:
        await select_shell(WorldFs(str(tmp_path), existing=set()), world, None)
    assert str(caught.value).endswith("Searched Git Bash in:\n")


async def test_posix_which_is_trusted_and_sh_is_the_silent_fallback(tmp_path: Path) -> None:
    fs = WorldFs(str(tmp_path), existing=set())
    trusted = WorldSubprocess(
        str(tmp_path), lookup_program=child("import sys; sys.stdout.write('/nowhere/bash\\n')")
    )
    assert (await select_shell(fs, trusted, None)).shell == "/nowhere/bash"
    blank = WorldSubprocess(str(tmp_path), lookup_program=child("import sys; sys.stdout.write(' \\n')"))
    assert await select_shell(fs, blank, None) == ShellConfig("sh", ("-c",), "argv")
    failed = WorldSubprocess(str(tmp_path), lookup_program=child("import sys; sys.exit(1)"))
    assert (await select_shell(fs, failed, None)).shell == "sh"
    unspawnable = WorldSubprocess(str(tmp_path), lookup_program=[str(tmp_path / "no-such-program")])
    assert (await select_shell(fs, unspawnable, None)).shell == "sh"


@pytest.mark.parametrize(
    ("path", "legacy"),
    [
        ("C:/Windows/System32/bash.exe", True),
        ("c:\\windows\\sysnative\\bash.exe", True),
        ("\u212a:\\Windows\\System32\\bash.exe", True),
        ("C:\\Windows\\System32\\bash.exe\n", False),
        ("C:\\Program Files\\Git\\bin\\bash.exe", False),
        ("/bin/bash", False),
    ],
)
def test_legacy_wsl_bash_uses_stdin(path: str, legacy: bool) -> None:
    assert is_legacy_wsl_bash_path(path) is legacy
    config = shell_config(path)
    assert config.transport == ("stdin" if legacy else "argv")
    assert config.args == (("-s",) if legacy else ("-c",))


def test_js_trim_set() -> None:
    assert js_trim("\ufeff\u00a0\u2028x\u3000\t") == "x"
    assert js_trim("\u200bx") == "\u200bx"  # ZERO WIDTH SPACE is not ECMAScript whitespace


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc descriptor path")
async def test_descriptor_path_shell_is_selected_by_followed_existence(tmp_path: Path) -> None:
    """CE-WP133-01: Pi's existsSync accepts /proc/self/fd/N of an unlinked file (realpath would
    fail); probe_dir_entry agrees."""
    from minion_agent.execution import LocalFileSystem

    target = tmp_path / "shell"
    target.write_text("#!/bin/sh\n")
    handle = os.open(target, os.O_RDONLY)
    try:
        target.unlink()
        path = f"/proc/{os.getpid()}/fd/{handle}"
        fs = LocalFileSystem(str(tmp_path))
        assert (await select_shell(fs, WorldSubprocess(str(tmp_path)), path)).shell == path  # type: ignore[arg-type]
    finally:
        os.close(handle)
