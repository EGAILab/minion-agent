"""Thin runner for the L12-D012 `path-resolution` corpus (conformance/agent/path-resolution/;
schema conformance/schema/path-resolution-scenario.schema.json; record 12-l12-d012-path-colon.md
section 9). It never resolves a path itself: every observation is the real `ctx.fs.absolute_path`
seam (`LocalFileSystem`).

- baseline cases run in-process, after the case's preconditions are VERIFIED through the binding's
  own lookups (the inherited per-drive entry via `_libuv_win32.inherited_drive_cwd`, the process
  cwd's drive); an unmet precondition REFUSES -- the expectation is never rewritten;
- host cases (win32) run the seam in a CHILD Python process whose environment block is this
  process's ordinary variables (CPython's `os.environ` holds no `=`-prefixed entry) plus exactly the
  case's `=<drive>:` entry, and whose cwd is a contained sandbox directory; `cwd_drive` / `cwd_tail`
  template parts are filled from the cwd the child reports;
- L12-D013: a host case's `cwd_form` `extended` runs the child at `\\\\?\\` + the sandbox path; the
  runner REFUSES unless the child observes exactly that spelling, and fills templates from the
  ordinary sandbox path.

No case path is ever passed to a filesystem operation or used as a cwd.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import minion_agent
from minion_agent.execution.filesystem import LocalFileSystem
from minion_agent.execution.result import Ok

PATH_RESOLUTION_DIR = (
    Path(__file__).resolve().parents[3] / "conformance" / "agent" / "path-resolution"
)
DOCUMENT: dict[str, Any] = json.loads(
    (PATH_RESOLUTION_DIR / "path-resolution.json").read_text(encoding="utf-8")
)
HOST_PLATFORM = "win32" if sys.platform == "win32" else "linux"
SRC = str(Path(minion_agent.__file__).resolve().parents[1])

_CHILD = """
import asyncio, json, os, sys
from minion_agent.execution.filesystem import LocalFileSystem
provider_cwd, ordinary, input_parts = json.loads(sys.argv[1])
case = {"upper": ordinary[0].upper(), "lower": ordinary[0].lower()}
path = "".join(value if kind == "text" else case[value] for kind, value in input_parts)
result = asyncio.run(LocalFileSystem(provider_cwd).absolute_path(path))
print(json.dumps({"cwd": os.getcwd(), "path": path, "value": result.value}))
"""
EXTENDED_PREFIX = "\\\\?\\"


def text(value: dict[str, Any]) -> str:
    units: list[int] = value["utf16"]
    return b"".join(u.to_bytes(2, "little") for u in units).decode("utf-16-le", "surrogatepass")


def instantiate(value: dict[str, Any], cwd: str) -> str:
    """A text value, or a template whose parts are filled from `cwd` (the child's own cwd)."""
    if "utf16" in value:
        return text(value)
    out = []
    for part in value["parts"]:
        if "utf16" in part:
            out.append(text(part))
        elif "cwd_drive" in part:
            out.append(cwd[0].upper() if part["cwd_drive"] == "upper" else cwd[0].lower())
        else:
            out.append(cwd[2:])
    return "".join(out)


def precondition_refusal(
    case: dict[str, Any],
    *,
    inherited: Callable[[str], str | None],
    process_cwd: str,
) -> str | None:
    """Why this host state does not meet the case's preconditions, or None. An inherited entry that
    is present but EMPTY is present (absent and empty stay distinct)."""
    pre = case["preconditions"]
    for drive in pre.get("absent_entries", []):
        if inherited(f"{drive}:") is not None:
            return f"inherited ={drive}: entry is present"
    for drive in pre.get("process_cwd_drive_not", []):
        if process_cwd[1:2] == ":" and process_cwd[0].upper() == drive:
            return f"process cwd {process_cwd!r} is on {drive}:"
    return None


def real_inherited(device: str) -> str | None:
    if sys.platform != "win32":  # pragma: no cover - POSIX has no per-drive cwd entries
        return None
    from minion_agent.execution import _libuv_win32

    return _libuv_win32.inherited_drive_cwd(device)


def process_cwd() -> str:
    """The runner's own process cwd (module-level, so the refusal controls can script it)."""
    return os.getcwd()


async def run_baseline_case(case: dict[str, Any]) -> str:
    refusal = precondition_refusal(case, inherited=real_inherited, process_cwd=process_cwd())
    if refusal is not None:
        raise AssertionError(f"PRECONDITION REFUSED: {refusal}")
    result = await LocalFileSystem(text(case["provider_cwd"])).absolute_path(text(case["input"]))
    assert isinstance(result, Ok)
    return result.value


def host_child_env(case: dict[str, Any], cwd: str) -> dict[str, str]:
    """The child's environment block: ordinary variables, no `=`-prefixed entry, plus exactly the
    case's per-drive entry when it is present."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("=")}
    env["PYTHONPATH"] = SRC
    host = case["host"]
    letter = cwd[0].upper() if host["drive"] == "cwd" else host["drive"]
    if host["entry"] is not None:
        env[f"={letter}:"] = instantiate(host["entry"], cwd)
    return env


def child_cwd_spelling(case: dict[str, Any], ordinary: str) -> str:
    """The child's cwd for the case's `cwd_form` (L12-D013): the ordinary sandbox path, or the
    four units `\\\\?\\` followed by it."""
    extended = case["host"].get("cwd_form", "ordinary") == "extended"
    return EXTENDED_PREFIX + ordinary if extended else ordinary


def run_host_case(case: dict[str, Any], sandbox: Path) -> tuple[str, str]:
    """(observed, expected) for a host case, from a child run in `sandbox` (a contained, existing,
    empty directory), spelled as the case's `cwd_form`. Templates are filled from the ORDINARY
    sandbox path; a child that does not observe exactly the case's cwd spelling REFUSES."""
    cwd = str(sandbox)
    refusal = precondition_refusal(case, inherited=lambda _: None, process_cwd=cwd)
    if refusal is not None:
        raise AssertionError(f"PRECONDITION REFUSED (child cwd): {refusal}")
    spelling = child_cwd_spelling(case, cwd)
    parts: list[tuple[str, str]] = []
    value = case["input"]
    for part in value.get("parts", [value]):
        parts.append(("text", text(part)) if "utf16" in part else ("drive", part["cwd_drive"]))
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps([text(case["provider_cwd"]), cwd, parts])],
        env=host_child_env(case, cwd),
        cwd=spelling,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    observed = json.loads(done.stdout)
    if spelling == cwd:
        same = os.path.normcase(observed["cwd"]) == os.path.normcase(cwd)
    else:
        same = observed["cwd"] == spelling
    if not same:
        raise AssertionError(f"PRECONDITION REFUSED (cwd form): child observed {observed['cwd']!r}")
    # An ordinary child's reported cwd fills the templates, as before L12-D013; an extended child's
    # is the namespace spelling, so its templates come from the ordinary sandbox path.
    return observed["value"], instantiate(case["expect"], observed["cwd"] if spelling == cwd else cwd)
