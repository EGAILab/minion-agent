"""Execute the L12-D012 `path-resolution` canonical cases on this host's platform (spec/execution.md
section 3.2; record 12-l12-d012-path-colon.md sections 9 and 10) through the real `ctx.fs`
`absolute_path` seam, plus the permanent controls the contract requires of every runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import filesystem

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


def test_control_missing_precondition_refuses() -> None:
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


@pytest.mark.skipif(sys.platform != "win32", reason="the per-drive entry exists only on win32")
def test_host_runner_child_environment_carries_exactly_the_case_entry(tmp_path: Path) -> None:
    from .path_resolution_runner import host_child_env

    env = host_child_env(BY_ID["win32/host/a-drive/08"], str(tmp_path))
    assert env["=A:"] == "X:\\other"
    assert [k for k in env if k.startswith("=")] == ["=A:"]
    assert "=A:" not in os.environ
