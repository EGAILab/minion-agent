"""Execute the L12-D012 `path-resolution` canonical cases on this host's platform (spec/execution.md
section 3.2; record 12-l12-d012-path-colon.md sections 9 and 10) through the real `ctx.fs`
`absolute_path` seam, plus the permanent controls the contract requires of every runner."""

from __future__ import annotations

import os
import subprocess as subprocess_module
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import filesystem

from . import path_resolution_runner as runner
from .path_resolution_runner import (
    DOCUMENT,
    HOST_PLATFORM,
    instantiate,
    precondition_refusal,
    run_baseline_case,
    run_host_case,
    text,
)

CASES: list[dict[str, Any]] = DOCUMENT["path_resolution"]["cases"]
MINE = [c for c in CASES if c["platform"] == HOST_PLATFORM]
BASELINE = [c for c in MINE if c["host"] is None]
HOST = [c for c in MINE if c["host"] is not None]
ALL_HOST = [c for c in CASES if c["host"] is not None]
BY_ID = {c["id"]: c for c in CASES}
SYNTHETIC_CWD = "E:\\sandbox\\child"  # the in-process port witnesses' process cwd (E: is not A:/C:)


@pytest.mark.parametrize("case", BASELINE, ids=lambda c: c["id"])
async def test_baseline_case(case: dict[str, Any]) -> None:
    assert await run_baseline_case(case) == text(case["expect"])


@pytest.mark.skipif(sys.platform != "win32", reason="host-state cases are win32 only")
@pytest.mark.parametrize("case", HOST, ids=lambda c: c["id"])
def test_host_case_in_crafted_child(case: dict[str, Any], tmp_path: Path) -> None:
    observed, expected = run_host_case(case, tmp_path)
    assert observed == expected


def test_every_case_of_this_platform_is_selected() -> None:
    assert len(MINE) == (55 if HOST_PLATFORM == "win32" else 25)
    assert len(BASELINE) + len(HOST) == len(MINE)


# --- the binding's port, driven in-process with injected host state (both platforms) ------------


def _injected(case: dict[str, Any]) -> tuple[Any, Any, str]:
    host = case["host"]
    letter = SYNTHETIC_CWD[0] if host["drive"] == "cwd" else host["drive"]
    entry = None if host["entry"] is None else instantiate(host["entry"], SYNTHETIC_CWD)

    def inherited(device: str) -> str | None:
        return entry if device.upper() == f"{letter}:" else None

    path = instantiate(case["input"], SYNTHETIC_CWD)
    return inherited, (lambda: SYNTHETIC_CWD), path


def _port(case: dict[str, Any]) -> str:
    inherited, process_cwd, path = _injected(case)
    return filesystem.node_win32_resolve(
        [text(case["provider_cwd"]), path], inherited=inherited, process_cwd=process_cwd
    )


@pytest.mark.parametrize("case", ALL_HOST, ids=lambda c: c["id"])
def test_port_reproduces_every_host_class(case: dict[str, Any]) -> None:
    assert _port(case) == instantiate(case["expect"], SYNTHETIC_CWD)


# --- permanent controls (record section 9, runner requirement 3) ---------------------------------


def _fall_through(device: str, inherited: Any, process_cwd: Any) -> str:
    """WRONG: the first source that names the device (revision 1's rejected paraphrase)."""
    entry, cwd = inherited(device), process_cwd()
    for base in (entry, cwd):
        if base and base[:2].lower() == device.lower():
            return str(base)
    return f"{device}\\"


def _mismatch_always_root(device: str, inherited: Any, process_cwd: Any) -> str:
    """WRONG: any device mismatch selects the root (ignores the third-character conjunction)."""
    base = inherited(device) or process_cwd()
    return f"{device}\\" if base[:2].lower() != device.lower() else str(base)


def _empty_is_a_value(device: str, inherited: Any, process_cwd: Any) -> str:
    """WRONG: an empty entry is used as the base instead of selecting the process cwd."""
    entry = inherited(device)
    base = entry if entry is not None else process_cwd()
    if base[:2].lower() != device.lower() and base[2:3] == "\\":
        return f"{device}\\"
    return str(base)


@pytest.mark.parametrize(
    ("wrong", "killed_by"),
    [
        (_fall_through, "win32/host/cwd-drive/08"),
        (_mismatch_always_root, "win32/host/cwd-drive/09"),
        (_empty_is_a_value, "win32/host/cwd-drive/02"),
    ],
    ids=["fall-through", "mismatch-always-root", "empty-is-a-value"],
)
def test_control_wrong_selection_is_killed_by_its_named_case(
    wrong: Any, killed_by: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = BY_ID[killed_by]
    expected = instantiate(case["expect"], SYNTHETIC_CWD)
    assert _port(case) == expected  # the real selection passes
    monkeypatch.setattr(filesystem, "_select_drive_base", wrong)
    assert _port(case) != expected, f"{wrong.__name__} survived {killed_by}"


def test_control_pairs_discriminate() -> None:
    """The wrong-drive entry (08) and the absent entry (01) differ on the cwd's own drive, and the
    empty entry (02) equals the absent one: the pairs the controls rely on."""
    observed = {n: _port(BY_ID[f"win32/host/cwd-drive/{n}"]) for n in ("01", "02", "08")}
    assert observed["01"] == observed["02"] != observed["08"]


def test_precondition_helper_truth_table() -> None:
    """The precondition helper alone (supporting evidence; the entry-point enforcement controls
    below are the L12D012-PY-I001 witnesses)."""
    case = BY_ID["win32/baseline/12"]
    assert precondition_refusal(case, inherited=lambda _: None, process_cwd="E:\\w") is None
    for entry in ("A:\\deep", ""):  # present -- including present but EMPTY -- refuses
        refusal = precondition_refusal(
            case, inherited=lambda d, v=entry: v if d == "A:" else None, process_cwd="E:\\w"
        )
        assert refusal == "inherited =A: entry is present"
    assert precondition_refusal(case, inherited=lambda _: None, process_cwd="A:\\w") is not None
    host = BY_ID["win32/host/a-drive/01"]
    assert precondition_refusal(host, inherited=lambda _: None, process_cwd="C:\\w") is not None


# --- entry-point enforcement controls (L12D012-PY-I001) ------------------------------------------
# The ACTUAL runner entry points run under scripted host state (the inherited-entry lookup and the
# process cwd are module-level in the runner; a host case's child cwd is its sandbox argument). No
# case path and no scripted cwd is ever created, accessed or used as a real cwd: a refused run stops
# before the provider or the child, and an admitted host run stops AT the child launch (a spy).


class _Reached(Exception):
    """The admitted host run reached the child launch (the spy stops it there)."""


class _Observed:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.provider: list[str] = []
        self.child: list[list[str]] = []
        real = filesystem.LocalFileSystem.absolute_path

        async def absolute_path(fs: Any, path: str, signal: Any = None) -> Any:
            self.provider.append(path)
            return await real(fs, path, signal)

        def run(args: list[str], **_: Any) -> Any:
            self.child.append(args)
            raise _Reached

        monkeypatch.setattr(filesystem.LocalFileSystem, "absolute_path", absolute_path)
        monkeypatch.setattr(subprocess_module, "run", run)


def _script(monkeypatch: pytest.MonkeyPatch, module: Any, entry: str | None, cwd: str) -> None:
    monkeypatch.setattr(module, "real_inherited", lambda d: entry if d == "A:" else None)
    monkeypatch.setattr(module, "process_cwd", lambda: cwd)


async def _baseline_refused(module: Any, observed: _Observed) -> bool:
    """The refusal witness: the entry point refuses, and the provider is never invoked."""
    try:
        await module.run_baseline_case(BY_ID["win32/baseline/12"])
    except AssertionError as error:
        return str(error).startswith("PRECONDITION REFUSED") and observed.provider == []
    return False


def _host_refused(module: Any, observed: _Observed, case_id: str, sandbox: str) -> bool:
    try:
        module.run_host_case(BY_ID[case_id], Path(sandbox))
    except _Reached:
        return False
    except AssertionError as error:
        return str(error).startswith("PRECONDITION REFUSED") and observed.child == []
    return False


FORBIDDEN_BASELINE = [
    ("", "E:\\w"),
    ("A:\\deep", "E:\\w"),
    (None, "A:\\w"),
    (None, "X:\\w"),
    (None, "C:\\w"),
]
FORBIDDEN_HOST = [
    ("win32/host/a-drive/01", "C:\\never-created"),
    ("win32/host/a-drive/08", "A:\\never-created"),
    ("win32/host/cwd-drive/01", "C:\\never-created"),
]


@pytest.mark.parametrize(("entry", "cwd"), FORBIDDEN_BASELINE)
async def test_baseline_entry_refuses_before_the_provider(
    entry: str | None, cwd: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Present (including present-but-EMPTY) =A:, or a process cwd on A:/X:/C:: refused, and the
    real provider is never reached."""
    observed = _Observed(monkeypatch)
    _script(monkeypatch, runner, entry, cwd)
    assert await _baseline_refused(runner, observed)


async def test_baseline_entry_admits_and_reaches_the_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: admissible state reaches the real `absolute_path` exactly once."""
    observed = _Observed(monkeypatch)
    _script(monkeypatch, runner, None, "E:\\w")
    value = await runner.run_baseline_case(BY_ID["win32/baseline/12"])
    assert observed.provider == ["a::/b"]
    if HOST_PLATFORM == "win32":
        assert value == text(BY_ID["win32/baseline/12"]["expect"])


@pytest.mark.parametrize(("case_id", "sandbox"), FORBIDDEN_HOST)
def test_host_entry_refuses_before_the_child(
    case_id: str, sandbox: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = _Observed(monkeypatch)
    assert _host_refused(runner, observed, case_id, sandbox)


def test_host_entry_admits_and_reaches_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: an admissible child cwd reaches the child launch (stopped by the spy)."""
    observed = _Observed(monkeypatch)
    with pytest.raises(_Reached):
        runner.run_host_case(BY_ID["win32/host/a-drive/01"], Path("E:\\never-created"))
    assert len(observed.child) == 1


def _without_enforcement(marker: str) -> Any:
    """The runner with ONE entry point's refusal branch removed, compiled in memory (nothing is
    written); `marker` is that branch's exact source."""
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert source.count(marker) == 1
    mutated = source.replace(marker, marker.replace("if refusal is not None:", "if False:"))
    module = types.ModuleType("path_resolution_runner_without_enforcement")
    module.__file__ = runner.__file__
    exec(compile(mutated, runner.__file__, "exec"), module.__dict__)
    return module


_BASELINE_BRANCH = (
    '    if refusal is not None:\n        raise AssertionError(f"PRECONDITION REFUSED: {refusal}")'
)
_HOST_BRANCH = (
    "    if refusal is not None:\n"
    '        raise AssertionError(f"PRECONDITION REFUSED (child cwd): {refusal}")'
)


@pytest.mark.parametrize(("entry", "cwd"), FORBIDDEN_BASELINE[:1] + FORBIDDEN_BASELINE[2:3])
async def test_control_baseline_enforcement_omission_is_killed(
    entry: str | None, cwd: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known-bad (the reviewer's mutant): the baseline entry point without its refusal branch runs
    the forbidden case through the provider, so the refusal witness above FAILS on it."""
    mutant = _without_enforcement(_BASELINE_BRANCH)
    observed = _Observed(monkeypatch)
    _script(monkeypatch, mutant, entry, cwd)
    assert not await _baseline_refused(mutant, observed)
    assert observed.provider == ["a::/b"]


@pytest.mark.parametrize(("case_id", "sandbox"), FORBIDDEN_HOST[:1])
def test_control_host_enforcement_omission_is_killed(
    case_id: str, sandbox: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = _without_enforcement(_HOST_BRANCH)
    observed = _Observed(monkeypatch)
    assert not _host_refused(mutant, observed, case_id, sandbox)
    assert len(observed.child) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="the per-drive entry exists only on win32")
def test_host_runner_child_environment_carries_exactly_the_case_entry(tmp_path: Path) -> None:
    from .path_resolution_runner import host_child_env

    env = host_child_env(BY_ID["win32/host/a-drive/08"], str(tmp_path))
    assert env["=A:"] == "X:\\other"
    assert [k for k in env if k.startswith("=")] == ["=A:"]
    assert "=A:" not in os.environ
