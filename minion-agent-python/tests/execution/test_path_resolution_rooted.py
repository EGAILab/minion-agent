"""L12-D013 (record `12-l12-d013-rooted.md`; spec/execution.md section 3.2, rooted and UNC-root
inputs on win32): a device-less rooted or incomplete-UNC input takes the PROCESS cwd's device, a
complete UNC root is its own device, and neither the provider `cwd` nor an inherited `=X:` entry
plays any part.

Every expectation is pinned Node's recorded observation (`data/node_rooted_cwd.json`, frozen from
the native scoping characterization `rooted-probe-2.json` by minion-agent-docs
assurance/layers/data/l12-d013/impl/build_fixture.py), never a value the binding computes. The UNC
process cwd (`\\\\localhost\\E$\\...`) is covered here at the pure seam with an injected cwd, as the
scoping review requires (record section 7.2, item 4); it is never used as a real cwd."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import filesystem
from minion_agent.execution.filesystem import node_win32_resolve, resolve_local_path
from minion_agent.execution.subprocess import LocalSubprocess, SpawnOptions, StdioMode

FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "data" / "node_rooted_cwd.json").read_text(encoding="utf-8")
)
WIN = pytest.mark.skipif(sys.platform != "win32", reason="the win32 resolver branch")
PROVIDERS = ["C:\\x", "E:\\y", "\\\\srv\\share\\x", "\\\\?\\C:\\x"]


def text(units: list[int]) -> str:
    return b"".join(u.to_bytes(2, "little") for u in units).decode("utf-16-le", "surrogatepass")


INPUTS = [text(i) for i in FIXTURE["inputs"]]
ROWS = [
    (form["label"], text(form["cwd"]), source, text(node))
    for form in FIXTURE["cwd_forms"]
    for source, node in zip(INPUTS, form["node"], strict=True)
]


def _unused(*_: Any) -> Any:
    raise AssertionError("a host operand the rooted branch must not read was read")


@pytest.mark.parametrize(("form", "cwd", "source", "node"), ROWS)
def test_port_alone_reproduces_node(form: str, cwd: str, source: str, node: str) -> None:
    """`win32.resolve(input)` alone, with Node's own process cwd: never an inherited entry."""
    assert node_win32_resolve([source], inherited=_unused, process_cwd=lambda: cwd) == node


@WIN
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(("form", "cwd", "source", "node"), ROWS)
def test_resolver_ignores_the_provider_and_entries(
    provider: str, form: str, cwd: str, source: str, node: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real `resolve_local_path` (M1-M3): every provider of the characterization gives Node's
    process-cwd result, and no inherited `=X:` entry is consulted."""
    monkeypatch.setattr(filesystem, "_native_inherited_view", _unused)
    monkeypatch.setattr(filesystem, "_native_process_cwd_view", lambda: cwd)
    assert resolve_local_path(provider, source) == node


@pytest.mark.skipif(sys.platform == "win32", reason="the POSIX normalization (M4)")
@pytest.mark.parametrize(
    ("provider", "source", "node"),
    [("/x", "//leaf", "/leaf"), ("//x", "a", "/x/a"), ("//x", "..", "/")],
)
def test_posix_result_has_one_leading_slash(provider: str, source: str, node: str) -> None:
    """M4 in both branches: pinned Node's `path.posix.resolve(provider, source)` (pure; the values
    are recorded by the gate's own Node check), where POSIX `normpath` keeps two leading `/`."""
    assert resolve_local_path(provider, source) == node


def test_fixture_is_the_frozen_characterization() -> None:
    labels = [f["label"] for f in FIXTURE["cwd_forms"]]
    assert labels == ["drive_root", "extended", "plain", "unc_admin"]
    assert len(INPUTS) == 25 and len(ROWS) == 100
    by_form = {f["label"]: f for f in FIXTURE["cwd_forms"]}
    assert text(by_form["unc_admin"]["cwd"]).startswith("\\\\localhost\\E$\\")
    assert text(by_form["extended"]["cwd"]) == "\\\\?\\" + text(by_form["plain"]["cwd"])


@WIN
async def test_exec005_rooted_spawn_cwd_uses_the_process_device(tmp_path: Path) -> None:
    """EXEC-005 consumer witness: a rooted spawn `cwd` resolves through the same rule. The provider
    cwd is on C: and never accessed; the rooted spelling of the contained `tmp_path` must reach
    this process's own drive, where merged Python's M1 would spawn in `C:` instead."""
    drive = os.getcwd()[:2]
    if drive.upper() == "C:" or str(tmp_path)[:2].upper() != drive.upper():
        raise AssertionError(f"PRECONDITION REFUSED: process cwd {drive} must be tmp_path's, not C:")
    rooted = str(tmp_path)[2:]
    assert rooted.startswith("\\") and not rooted.startswith("\\\\")
    sp = LocalSubprocess(cwd="C:\\never-used")
    result = await sp.spawn(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        SpawnOptions(stdout=StdioMode.PIPED, cwd=rooted),
    )
    process = result.value  # type: ignore[union-attr]
    assert process.stdout is not None
    output = b""
    while True:
        chunk = await process.stdout.read_chunk()
        if chunk.value is None:  # type: ignore[union-attr]
            break
        output += chunk.value  # type: ignore[union-attr]
    await process.wait()
    assert os.path.samefile(output.decode().strip(), str(tmp_path))
