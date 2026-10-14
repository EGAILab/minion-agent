"""CE-L12D012-01 (L12D012-PY-I002; record `12-l12-d012-convergence-ce01.md` sections 7
and 8): the win32 drive-relative branch's selected value is Node's own string view of a
NATIVE operand -- the inherited per-drive entry, else the process cwd -- indexed by UTF-16
code units; a LOGICAL caller argument stays raw through it.

Every expectation is Node's recorded observation (`data/node_selected_value.json`, frozen
from the native pinned-Pi characterization by minion-agent-docs
assurance/layers/data/l12-d012/ce01/build_fixture.mjs), never a value the binding computes.
W1/W4/W5's real-binding witnesses run the real `LocalFileSystem` in child processes with
crafted environment blocks (and, for W4, a contained `tmp_path` directory used only as the
child's cwd)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import filesystem
from minion_agent.execution.filesystem import (
    LocalFileSystem,
    node_native_string_view,
    node_win32_resolve,
)

FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "data" / "node_selected_value.json").read_text(encoding="utf-8")
)
ENTRY = {r["label"]: r for r in FIXTURE["entry_rows"]}
CWD_NAMES = {r["label"]: r for r in FIXTURE["cwd_names"]}
LOGICAL = {r["label"]: r for r in FIXTURE["logical_rows"]}
SRC = str(Path(filesystem.__file__).resolve().parents[2])
WIN = pytest.mark.skipif(sys.platform != "win32", reason="the per-drive entry exists only on win32")


def text(units: list[int] | None) -> str | None:
    if units is None:
        return None
    return b"".join(u.to_bytes(2, "little") for u in units).decode("utf-16-le", "surrogatepass")


def units(value: str) -> list[int]:
    return list(memoryview(value.encode("utf-16-le", "surrogatepass")).cast("H"))


def _port(entry_view: str | None, input_path: str, process_cwd: str = "E:\\host") -> str:
    def inherited(device: str) -> str | None:
        return entry_view if device.lower() in ("a:", "e:") else None

    return node_win32_resolve(
        ["C:\\x", input_path], inherited=inherited, process_cwd=lambda: process_cwd
    )


# --- W2: the pure port, with NODE's recorded view of each entry (all platforms) ---------------


@pytest.mark.parametrize("label", sorted(ENTRY))
def test_port_with_node_view_of_the_entry(label: str) -> None:
    row = ENTRY[label]
    assert units(_port(text(row["node_entry"]), "a:leaf")) == row["node"]


@pytest.mark.parametrize("label", sorted(LOGICAL))
def test_port_keeps_the_logical_tail_raw(label: str) -> None:
    row = LOGICAL[label]
    observed = _port(text(row["node_entry"]), text(row["logical"]) or "")
    assert units(observed) == row["node"]


# --- W3: the native-string view against section 15's executed authority -------------------------


@pytest.mark.parametrize(
    ("native", "view"),
    [
        # section 15: Windows value a<D800>b -> a, three U+FFFD, b
        ("a\ud800b", "a\ufffd\ufffd\ufffdb"),
        ("\ude00x", "\ufffd\ufffd\ufffdx"),  # lone low
        # a valid pair held as two surrogates is its astral character
        ("\ud83d\ude00", "\U0001f600"),
        ("x\U0001f600", "x\U0001f600"),
        ("\ufeffbom", "\ufeffbom"),  # a BOM is kept
        ("plain", "plain"),
    ],
)
def test_native_string_view_is_section_15(native: str, view: str) -> None:
    assert node_native_string_view(native) == view


def test_native_string_view_agrees_with_the_certified_environment_view() -> None:
    """The same rule as WP-13.3's certified `node_environment_view` value decode (section 15)."""
    from minion_agent.execution import EnvSnapshot, Platform
    from minion_agent.tools.builtin.environment import node_environment_view

    values = ["a\ud800b", "\ude00x", "\ud83d\ude00", "\ufeffbom", "x\ud83dy"]
    snapshot = EnvSnapshot([(f"N{i}", v) for i, v in enumerate(values)], Platform.WINDOWS)
    view = node_environment_view(snapshot)
    assert [view[f"N{i}"] for i in range(len(values))] == [
        node_native_string_view(v) for v in values
    ]


# --- W1 / W4 / W5: the REAL binding in child processes (win32) ---------------------------------

_CHILD = """
import asyncio, json, os, sys
sys.path.insert(0, sys.argv[1])
from minion_agent.execution.filesystem import LocalFileSystem
data = b"".join(u.to_bytes(2, "little") for u in json.loads(sys.argv[2]))
logical = data.decode("utf-16-le", "surrogatepass")
out = asyncio.run(LocalFileSystem("C:\\\\x").absolute_path(logical)).value
cwd = os.getcwd()
print(json.dumps({"cwd": list(memoryview(cwd.encode("utf-16-le", "surrogatepass")).cast("H")),
                  "out": list(memoryview(out.encode("utf-16-le", "surrogatepass")).cast("H"))}))
"""


def _child(input_path: str, entry: tuple[str, str] | None, cwd: Path) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("=")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if entry is not None:
        env[entry[0]] = entry[1]
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, SRC, json.dumps(units(input_path))],
        env=env,
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    result: dict[str, Any] = json.loads(done.stdout)
    return result


def _require_e_drive(path: Path) -> str:
    drive = str(path)[0].upper()
    if drive in ("A", "C"):
        raise AssertionError(f"PRECONDITION REFUSED: tmp_path on {drive}: (must not be A:/C:)")
    return drive


@WIN
@pytest.mark.parametrize("label", sorted(ENTRY))
def test_real_binding_child_with_the_crafted_entry(label: str, tmp_path: Path) -> None:
    """W1: exactly the crafted `=A:` (lone surrogates included) through the real reader and
    provider."""
    row = ENTRY[label]
    entry = text(row["entry"])
    observed = _child("a:leaf", None if entry is None else ("=A:", entry), tmp_path)
    assert observed["out"] == row["node"]


@WIN
@pytest.mark.parametrize("label", sorted(CWD_NAMES))
@pytest.mark.parametrize("entry", [None, ""], ids=["entry-absent", "entry-empty"])
def test_real_binding_child_in_a_crafted_cwd(label: str, entry: str | None, tmp_path: Path) -> None:
    """W4: the process cwd is the selected value; its name contributes Node's recorded view. The
    directory is created inside pytest's contained tmp_path and is only the child's cwd."""
    drive = _require_e_drive(tmp_path)
    row = CWD_NAMES[label]
    directory = tmp_path / (text(row["name"]) or "")
    directory.mkdir()
    observed = _child(
        f"{drive.lower()}:leaf", None if entry is None else (f"={drive}:", entry), directory
    )
    assert text(observed["cwd"]) == str(directory)
    prefix = str(tmp_path)[1:]  # ASCII: Node's view of it is itself
    expected = f"{drive.lower()}{prefix}\\{text(row['node_view'])}\\leaf"
    assert text(observed["out"]) == expected


@WIN
def test_real_binding_child_in_a_crafted_cwd_neighbors(tmp_path: Path) -> None:
    """W4 positives: a matching non-empty entry is selected (the cwd is not read), and another
    drive's input falls back to its root."""
    drive = _require_e_drive(tmp_path)
    directory = tmp_path / (text(CWD_NAMES["lone-high"]["name"]) or "")
    directory.mkdir()
    deep = _child(f"{drive.lower()}:leaf", (f"={drive}:", f"{drive}:\\deep"), directory)
    assert text(deep["out"]) == f"{drive.lower()}:\\deep\\leaf"
    other = _child("a:leaf", None, directory)
    assert text(other["out"]) == "a:\\leaf"


@WIN
@pytest.mark.parametrize("label", sorted(LOGICAL))
def test_real_binding_child_keeps_the_logical_tail_raw(label: str, tmp_path: Path) -> None:
    """W5: the logical input travels as UTF-16 units and is rebuilt inside the child."""
    row = LOGICAL[label]
    observed = _child(text(row["logical"]) or "", ("=E:", text(row["entry"]) or ""), tmp_path)
    assert observed["out"] == row["node"]


# --- W5 through the real HOOK, in-process (win32), and the controls ----------------------------


def _hook(monkeypatch: pytest.MonkeyPatch, raw_entry: str | None, logical: str) -> str:
    from minion_agent.execution import _libuv_win32

    monkeypatch.setattr(_libuv_win32, "inherited_drive_cwd", lambda device: raw_entry)
    return filesystem.resolve_local_path("C:\\x", logical)


@WIN
@pytest.mark.parametrize("label", sorted(LOGICAL))
def test_hook_keeps_the_logical_tail_raw(label: str, monkeypatch: pytest.MonkeyPatch) -> None:
    row = LOGICAL[label]
    observed = _hook(monkeypatch, text(row["entry"]), text(row["logical"]) or "")
    assert units(observed) == row["node"]


def _code_point_selection(device: str, inherited: Any, process_cwd: Any) -> str:
    """K1 (known-bad): positions counted in Python characters."""
    path = inherited(device) or process_cwd()
    if path[:2].lower() != device.lower() and path[2:3] == "\\":
        return f"{device}\\"
    return str(path)


ASTRAL = ["astral-backslash", "x-astral-backslash", "astral-x-backslash"]
LONE = ["lone-high-backslash", "x-lone-high-backslash", "lone-low-x-backslash"]
RAW_TAIL = ["logical-lone-high", "logical-lone-low", "native-and-logical-nonscalar"]


def test_control_k1_code_point_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    def observe() -> dict[str, bool]:
        return {
            label: units(_port(text(row["node_entry"]), "a:leaf")) == row["node"]
            for label, row in ENTRY.items()
        }

    assert all(observe().values())  # green baseline
    monkeypatch.setattr(filesystem, "_select_drive_base", _code_point_selection)
    failed = sorted(label for label, ok in observe().items() if not ok)
    assert failed == sorted(ASTRAL), failed


@WIN
@pytest.mark.parametrize(
    ("name", "wrong"),
    [
        ("K2-raw-entry", lambda value: value),
        ("K3-per-surrogate", filesystem.scalar_value_string),  # one U+FFFD per lone surrogate
    ],
)
def test_control_entry_projection(name: str, wrong: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def observe() -> dict[str, bool]:
        return {
            label: units(_hook(monkeypatch, text(row["entry"]), "a:leaf")) == row["node"]
            for label, row in ENTRY.items()
        }

    assert all(observe().values())
    monkeypatch.setattr(filesystem, "node_native_string_view", wrong)
    failed = sorted(label for label, ok in observe().items() if not ok)
    assert failed == sorted(LONE), (name, failed)


@WIN
def test_control_k5_raw_process_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """K5 (known-bad): the process cwd passed raw. The cwd is a scripted string only."""
    from minion_agent.execution import _libuv_win32

    monkeypatch.setattr(_libuv_win32, "inherited_drive_cwd", lambda device: None)

    def observe() -> dict[str, bool]:
        out = {}
        for label, row in CWD_NAMES.items():
            monkeypatch.setattr(os, "getcwd", lambda r=row: f"E:\\sb\\{text(r['name'])}")
            observed = filesystem.resolve_local_path("C:\\x", "e:leaf")
            out[label] = observed == f"e:\\sb\\{text(row['node_view'])}\\leaf"
        return out

    assert all(observe().values())
    monkeypatch.setattr(filesystem, "_native_process_cwd_view", lambda: os.getcwd())
    failed = sorted(label for label, ok in observe().items() if not ok)
    assert failed == ["lone-high", "lone-high-then-ascii", "lone-low"], failed


@WIN
def test_control_k6_branch_local_logical_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """K6 (known-bad): the correct native views AND the view applied to the logical argument inside
    the other-drive hook only. Killed by the raw-tail rows; scalar and pair delimit it."""

    def observe() -> dict[str, bool]:
        return {
            label: units(_hook(monkeypatch, text(row["entry"]), text(row["logical"]) or ""))
            == row["node"]
            for label, row in LOGICAL.items()
        }

    assert all(observe().values())
    real = filesystem.node_win32_resolve

    def branch_local(paths: list[str], **kw: Any) -> str:
        return real([paths[0], node_native_string_view(paths[1])], **kw)

    monkeypatch.setattr(filesystem, "node_win32_resolve", branch_local)
    failed = sorted(label for label, ok in observe().items() if not ok)
    assert failed == sorted(RAW_TAIL), failed


@WIN
async def test_control_k6b_global_fs_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """K6b (known-bad): section 14.1's one-U+FFFD fs projection applied to the logical argument
    before resolution. Killed by W5's raw-tail rows and by L12-D001's logical rule (a lone-surrogate
    name's absolute_path stays raw)."""
    from minion_agent.execution import _libuv_win32

    row = LOGICAL["logical-lone-high"]
    monkeypatch.setattr(_libuv_win32, "inherited_drive_cwd", lambda device: text(row["entry"]))
    fs = LocalFileSystem("C:\\x")  # the other-drive branch is reached for e:leaf
    good = await fs.absolute_path(text(row["logical"]) or "")
    raw_name = await fs.absolute_path("a\ud800")
    assert units(good.value) == row["node"] and raw_name.value == "C:\\x\\a\ud800"  # type: ignore[union-attr]
    real = LocalFileSystem.absolute_path

    async def projected(self: Any, path: str, signal: Any = None) -> Any:
        return await real(self, filesystem.native_path(path), signal)

    monkeypatch.setattr(LocalFileSystem, "absolute_path", projected)
    bad = await fs.absolute_path(text(row["logical"]) or "")
    bad_name = await fs.absolute_path("a\ud800")
    assert units(bad.value) != row["node"]  # type: ignore[union-attr]
    assert bad_name.value != "C:\\x\\a\ud800"  # type: ignore[union-attr]


def test_fixture_is_the_frozen_characterization() -> None:
    assert len(ENTRY) == 15 and len(CWD_NAMES) == 5 and len(LOGICAL) == 5
    assert (
        sum(r["node_entry"] != r["entry"] for r in ENTRY.values()) == 3
    )  # the lone-surrogate rows
