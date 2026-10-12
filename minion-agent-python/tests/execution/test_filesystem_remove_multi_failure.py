"""L12-D008 / DIV-008 (minion-agent#215, provenance #127; spec/execution.md section 19.5 rule 4
as amended): recursive `remove` with several failing children.

Every row of `data/div008_pi.json` (pinned Pi's real `NodeExecutionEnv.remove`, measured on
Windows and Linux after its in-flight removals drained) runs through the real
`LocalFileSystem.remove` with the same injected failures, at the calls the binding makes. Each
observation must satisfy, against pinned Pi on this platform:

- EXTENT (required parity, Owner decision #215 issuecomment-6118734550): what remains equals what
  pinned Pi leaves once its in-flight removals drain -- every removable entry is gone, the
  failing entries and their ancestors stay;
- COMPLETE: every injected failure fires, so no sibling was skipped after a failure;
- SELECTION (DIV-008): the reported failure is the FIRST failure the binding encountered, in its
  own (host enumeration) order, with that failure's code; where pinned Pi's first-SETTLED
  failure is a different entry, the binding does not claim Pi's.

The controls replace one rule with a realistic wrong implementation; each must make its intended
rows fail, for the named reason."""

from __future__ import annotations

import errno
import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import LocalFileSystem, Ok
from minion_agent.execution import filesystem as fs

DATA = json.loads((Path(__file__).parent / "data" / "div008_pi.json").read_text("utf-8"))
ROWS = {row["id"]: row for row in DATA["rows"]}
PLATFORM = (
    "win32" if sys.platform == "win32" else "linux" if sys.platform.startswith("linux") else None
)
WIN32 = {"EACCES": 1920, "EBUSY": 32}
# The code pinned Pi reports for each injected failure when it selects it (rows M3/M4, both
# platforms).
CODE_OF = {"EACCES": "permission_denied", "EBUSY": "unknown"}

pytestmark = pytest.mark.skipif(PLATFORM is None, reason="pinned Pi measured on Windows and Linux")


def _error(code: str, path: str) -> OSError:
    if sys.platform == "win32":
        return OSError(0, f"injected {code}", path, WIN32[code])
    return OSError(getattr(errno, code), f"injected {code}", path)


class _Run:
    def __init__(self, root: Path, plans: list[dict[str, Any]]) -> None:
        self.root = root
        self.plans = [dict(plan) for plan in plans]
        self.calls: dict[str, int] = {}
        self.failures: list[tuple[str, str]] = []  # (rel, code), in the order the binding met them

    def wrap(self, op: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def call(p: Any, *args: Any, **kwargs: Any) -> Any:
            if not isinstance(p, int):
                rel = os.path.relpath(os.fspath(p), self.root).replace(os.sep, "/")
                if rel == "tree" or rel.startswith("tree/"):
                    key = f"{op} {rel}"
                    self.calls[key] = self.calls.get(key, 0) + 1
                    for plan in self.plans:
                        if (plan["op"], plan["rel"], plan["nth"]) == (op, rel, self.calls[key]):
                            plan["fired"] = True
                            self.failures.append((rel, plan["code"]))
                            raise _error(plan["code"], os.fspath(p))
            return real(p, *args, **kwargs)

        return call


@contextmanager
def _seams(monkeypatch: pytest.MonkeyPatch, run: _Run) -> Iterator[None]:
    with monkeypatch.context() as patch:
        patch.setattr(os, "lstat", run.wrap("lstat", os.lstat))
        patch.setattr(os, "stat", run.wrap("stat", os.stat))
        patch.setattr(os, "scandir", run.wrap("readdir", os.scandir))
        patch.setattr(fs, "_node_rmdir", run.wrap("rmdir", fs._node_rmdir))
        if sys.platform == "win32":
            patch.setattr(fs, "_libuv_unlink", run.wrap("unlink", fs._libuv_unlink))
        else:
            patch.setattr(os, "unlink", run.wrap("unlink", os.unlink))
        yield


def _remains(root: Path) -> list[str]:
    remains: list[str] = []

    def walk(rel: str) -> None:
        p = root / rel
        is_dir = p.is_dir() and not os.path.islink(p)
        remains.append(rel + "/" if is_dir else rel)
        if is_dir:
            for name in sorted(os.listdir(p)):
                walk(f"{rel}/{name}")

    for name in sorted(os.listdir(root)):
        walk(name)
    return remains


async def _observe(
    row: dict[str, Any], root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    for entry in row["fixture"]:
        if entry.endswith("/"):
            (root / entry[:-1]).mkdir()
        else:
            (root / entry).write_text("x")
    run = _Run(root, row["inject"])
    with _seams(monkeypatch, run):
        result = await LocalFileSystem(str(root)).remove("tree", recursive=True)
    return {
        "result": "ok" if isinstance(result, Ok) else str(result.error.code),
        "path": None
        if isinstance(result, Ok) or not result.error.path
        else os.path.relpath(result.error.path, root).replace(os.sep, "/"),
        "remains": _remains(root),
        "unfired": [f"{q['op']} {q['rel']} #{q['nth']}" for q in run.plans if not q.get("fired")],
        "failures": run.failures,
    }


def _violations(row: dict[str, Any], obs: dict[str, Any]) -> list[str]:
    pi = row["pi"][PLATFORM]
    problems: list[str] = []
    if obs["remains"] != pi["remains_after_drain"]:
        problems.append(f"extent: {obs['remains']} != Pi after drain {pi['remains_after_drain']}")
    if obs["unfired"]:
        problems.append(f"complete: never attempted {obs['unfired']}")
    first_rel, first_code = obs["failures"][0]
    if (obs["path"], obs["result"]) != (first_rel, CODE_OF[first_code]):
        problems.append(
            f"selection: reported {obs['path']} {obs['result']},"
            f" first encountered {first_rel} {first_code}"
        )
    if pi["path"] != first_rel and obs["path"] == pi["path"]:
        problems.append(
            f"selection: claims pinned Pi's first-settled {pi['path']},"
            f" not its own first {first_rel}"
        )
    return problems


@pytest.mark.parametrize("row_id", sorted(ROWS))
async def test_multi_failure_matches_div008(
    row_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    obs = await _observe(ROWS[row_id], tmp_path, monkeypatch)
    assert _violations(ROWS[row_id], obs) == []


async def test_drained_extent_differs_from_pi_settlement_state_where_pi_returns_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S1: pinned Pi returned while `tree/b` was still in flight (present at settlement, gone after
    # the drain). The binding returns after every child was attempted, so it matches the drained
    # state, not the settlement state.
    row = ROWS["S1-single-failure-before-sibling"]
    pi = row["pi"][PLATFORM]
    assert pi["remains_at_settle"] != pi["remains_after_drain"]
    obs = await _observe(row, tmp_path, monkeypatch)
    assert obs["remains"] == pi["remains_after_drain"]


def _stop_at_first_failure(path: str, original: OSError | None = None) -> None:
    """Control: the pre-L12-D008 walk (section 19.5 rule 4 as certified by L12-D007): the first
    child failure stops the walk, so later siblings are never attempted."""
    children = fs._rmdir_first(path, original)
    if children is None:
        return
    pending: list[tuple[str, list[str]]] = [(path, children[::-1])]
    while pending:
        directory, remaining = pending[-1]
        if not remaining:
            pending.pop()
            fs._rmdir_last(directory)
            continue
        descend = fs._rimraf_entry(remaining.pop())
        if descend is not None:
            grandchildren = fs._rmdir_first(*descend)
            if grandchildren is not None:
                pending.append((descend[0], grandchildren[::-1]))


def _report_last_failure(real: Callable[..., None]) -> Callable[..., None]:
    """Control: continues siblings correctly but reports the LAST failure encountered (as pinned
    Pi would when the later failure settles first): a false first-settled claim for a sequential
    binding."""

    def rimraf(path: str, original: OSError | None = None) -> None:
        seen: list[OSError] = []
        real_entry = fs._rimraf_entry

        def entry(p: str) -> Any:
            try:
                return real_entry(p)
            except OSError as exc:
                seen.append(exc)
                raise

        fs._rimraf_entry = entry  # type: ignore[assignment]
        try:
            real(path, original)
        except OSError:
            raise seen[-1] from None
        finally:
            fs._rimraf_entry = real_entry  # type: ignore[assignment]

    return rimraf


# Each control, the rows it must fail, and the violation each row must show (a control that fails
# for another reason is not credited). Row choice is order-independent: every named row's expected
# violation holds whichever child the host enumerates first.
CONTROLS: dict[str, tuple[Callable[[], Callable[..., None]], dict[str, str]]] = {
    "stop-at-first-failure": (
        lambda: _stop_at_first_failure,
        {
            # The second failing child is never attempted.
            "M5-sibling-between-settle-az": "complete:",
            "M1-two-files-settle-ab": "complete:",
        },
    ),
    "report-last-failure": (
        lambda: _report_last_failure(fs._rimraf),
        {
            "M1-two-files-settle-ab": "selection: reported",
            "M5-sibling-between-settle-az": "selection: reported",
        },
    ),
}


@pytest.mark.parametrize("control", sorted(CONTROLS))
async def test_controls_fail_their_intended_rows(
    control: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make, intended = CONTROLS[control]
    replacement = make()
    for row_id, reason in sorted(intended.items()):
        root = tmp_path / row_id
        root.mkdir()
        with monkeypatch.context() as patch:
            patch.setattr(fs, "_rimraf", replacement)
            obs = await _observe(ROWS[row_id], root, monkeypatch)
        violations = _violations(ROWS[row_id], obs)
        assert any(v.startswith(reason) for v in violations), (
            f"control {control} on {row_id}: {violations}"
        )


async def test_stop_at_first_failure_breaks_extent_somewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The extent witness for the stop-at-first control. With ONE failing child, the old walk
    # leaves the removable sibling whenever the host enumerates the failing child first. S1 and S2
    # fail opposite children, so on any enumeration order one of them visits the failing child
    # first; the control must break extent there.
    for row_id in (
        "S1-single-failure-before-sibling",
        "S2-single-failure-after-sibling",
        "S3-single-nested-failure",
    ):
        root = tmp_path / row_id
        root.mkdir()
        with monkeypatch.context() as patch:
            patch.setattr(fs, "_rimraf", _stop_at_first_failure)
            obs = await _observe(ROWS[row_id], root, monkeypatch)
        if obs["remains"] != ROWS[row_id]["pi"][PLATFORM]["remains_after_drain"]:
            return
    pytest.fail(
        "the stop-at-first-failure control left the drained extent intact on every single-failure"
        " row"
    )
