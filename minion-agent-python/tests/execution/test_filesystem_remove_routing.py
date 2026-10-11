"""`CE-L12D007-02` (agreed revision 3, minion-agent#199; spec/execution.md section 19.5): `remove`'s
target validation, entry classification, failure routing and Windows `EPERM` recovery.

Every row of the three agreed matrices (`data/remove_routing_pi.json`, generated from pinned Pi's
real `NodeExecutionEnv.remove` on Windows and Linux) runs through the real `LocalFileSystem.remove`
with the same injected failures, at the calls the binding makes: `os.lstat` (validation, then
classification), `os.stat` (recovery inspection), `os.scandir`, the provider's rmdir / unlink, and
the own-attribute correction (`chmod` in the matrix). Every other call is real. The observation --
result, path, every call per path, what remains, what is left read-only, and which planned
injections never fired -- must equal pinned Pi's on this platform.

The controls N1-N14 replace one rule with a realistic wrong implementation; each must make its
intended rows fail."""

from __future__ import annotations

import errno
import json
import os
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import LocalFileSystem, Ok
from minion_agent.execution import filesystem as fs

DATA = json.loads((Path(__file__).parent / "data" / "remove_routing_pi.json").read_text("utf-8"))
ROWS = {row["id"]: row for row in DATA["rows"]}
PLATFORM = (
    "win32" if sys.platform == "win32" else "linux" if sys.platform.startswith("linux") else None
)
WIN32 = {"EIO": 1117, "EACCES": 1920, "EBUSY": 32, "ENOENT": 2, "EPERM": 5}

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
        patch.setattr(fs, "_correct_own_attribute", run.wrap("chmod", fs._correct_own_attribute))
        if sys.platform == "win32":
            patch.setattr(fs, "_libuv_unlink", run.wrap("unlink", fs._libuv_unlink))
        else:
            patch.setattr(os, "unlink", run.wrap("unlink", os.unlink))
        yield


def _build(root: Path, fixture: list[Any]) -> list[Path]:
    readonly: list[Path] = []
    for entry in fixture:
        if isinstance(entry, str):
            if entry.endswith("/"):
                (root / entry[:-1]).mkdir()
            else:
                (root / entry).write_text("x")
        elif "link" in entry:
            os.symlink(root / entry["to"], root / entry["link"])
        else:
            readonly.append(root / entry["ro"])
    for path in readonly:
        os.chmod(path, stat.S_IREAD)
    return readonly


def _state(root: Path) -> tuple[list[str], list[str]]:
    remains: list[str] = []
    readonly: list[str] = []

    def walk(rel: str) -> None:
        p = root / rel
        is_link = os.path.islink(p)
        is_dir = p.is_dir() and not is_link
        remains.append(rel + "@" if is_link else rel + "/" if is_dir else rel)
        if not is_link and not os.lstat(p).st_mode & stat.S_IWRITE:
            readonly.append(rel)
        if is_dir:
            for name in sorted(os.listdir(p)):
                walk(f"{rel}/{name}")

    for name in sorted(os.listdir(root)):
        walk(name)
    return remains, readonly


async def _observe(
    row: dict[str, Any], root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    readonly_fixture = _build(root, row["fixture"])
    remove = row["remove"]
    run = _Run(root, row["inject"])
    with _seams(monkeypatch, run):
        result = await LocalFileSystem(str(root)).remove(
            remove["path"], recursive=remove["recursive"], force=remove.get("force", False)
        )
    path = None
    if not isinstance(result, Ok) and result.error.path:
        path = os.path.relpath(result.error.path, root).replace(os.sep, "/")
    remains, readonly = _state(root)
    for p in readonly_fixture:
        if os.path.lexists(p):
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
    observed = {
        "result": "ok" if isinstance(result, Ok) else str(result.error.code),
        "path": path,
        "calls": dict(sorted(run.calls.items())),
        "remains": remains,
        "readonly": readonly,
        "unfired": [f"{q['op']} {q['rel']} #{q['nth']}" for q in run.plans if not q.get("fired")],
    }
    return observed


def _expected(row: dict[str, Any]) -> dict[str, Any]:
    pi = dict(row["pi"][PLATFORM])
    pi.setdefault("readonly", [])
    pi.setdefault("unfired", [])
    return pi


@pytest.mark.parametrize("row_id", list(ROWS))
async def test_remove_routing_matches_pinned_pi(
    row_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = ROWS[row_id]
    assert await _observe(row, tmp_path, monkeypatch) == _expected(row)


# --- Negative controls N1-N14 (episode record; each must fail its intended rows) ----------------


def _n1_broad_catch(path: str) -> Any:
    """The `I003` defect: the `lstat` handler also encloses the directory's removal."""
    try:
        st = os.lstat(path)
        if fs._is_tree(st):
            fs._rimraf(path)
            return None
    except OSError as exc:
        if fs._vanished(exc):
            return None
        if fs._is_win_eperm(exc):
            return fs._fix_win_eperm(path, exc)
    return fs._unlink_routed(path)


def _n2_retry_rmdir(real: Callable[..., Any]) -> Callable[..., Any]:
    """A directory whose first `rmdir` failed is retried once (above the native seam, so the retried
    `rmdir` is a second real call)."""

    def rmdir_first(path: str, original: OSError | None) -> Any:
        try:
            return real(path, original)
        except OSError:
            fs._node_rmdir(path)
            return None

    return rmdir_first


def _n3_swallow(real: Callable[[str], None]) -> Callable[[str], None]:
    def entry(path: str) -> None:
        with suppress(OSError):
            real(path)

    return entry


def _n4_ancestor(real: Callable[..., None]) -> Callable[..., None]:
    def rimraf(path: str, original: OSError | None = None) -> None:
        try:
            real(path, original)
        except OSError as exc:
            exc.filename = path
            raise

    return rimraf


def _n5_list_first(real: Callable[..., Any]) -> Callable[..., Any]:
    def rmdir_first(path: str, original: OSError | None) -> Any:
        with os.scandir(path):
            pass
        return real(path, original)

    return rmdir_first


def _n6_no_lstat_recovery(path: str) -> None:
    try:
        st: os.stat_result | None = os.lstat(path)
    except OSError as exc:
        if fs._vanished(exc):
            return
        st = None
    if st is not None and fs._is_tree(st):
        return (path, None)
    return fs._unlink_routed(path)


def _recovery(*, own_error: bool = False, conditional: bool = False, keep_original: bool = False):  # type: ignore[no-untyped-def]
    """A wrong `fixWinEPERM`: N7 reports the correction / inspection error itself; N8 retries only
    when the read-only attribute was set (L12-D005's former rule); N9 replaces the retry's error."""

    def fix(path: str, original: OSError) -> None:
        try:
            if conditional and not os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_READONLY:
                raise original
            fs._correct_own_attribute(path)
            st = os.stat(path)
        except OSError as exc:
            if exc is original:
                raise
            if fs._vanished(exc):
                return
            if own_error:
                raise
            raise original from None
        if stat.S_ISDIR(st.st_mode):
            return (path, original)
        try:
            fs._node_unlink(path)
        except OSError as exc:
            if not fs._vanished(exc):
                if keep_original:
                    raise original from None
                raise

    return fix


def _n10_follow_link(path: str) -> None:
    os.chmod(os.path.realpath(path), stat.S_IREAD | stat.S_IWRITE)  # corrects the link's TARGET


def _remove(validate: Callable[[str, bool, bool], bool]) -> Callable[[str, bool, bool], None]:
    def remove_sync(path: str, recursive: bool, force: bool) -> None:
        if validate(path, recursive, force):
            descend = fs._rimraf_entry(path)
            if descend is not None:
                fs._rimraf(*descend)

    return remove_sync


def _n11_top_bypasses(path: str, recursive: bool, force: bool) -> bool:
    st = os.lstat(path)
    if fs._is_tree(st):
        if not recursive:
            raise fs._RmDirectoryRefusal(path)
        return True
    fs._node_unlink(path)
    return False


def _n12_validation_into_routing(path: str, recursive: bool, force: bool) -> bool:
    try:
        st = os.lstat(path)
    except OSError as exc:
        if force and fs._vanished(exc):
            return True
        return True  # the validation error routed into rimraf's own handling
    if fs._is_tree(st) and not recursive:
        raise fs._RmDirectoryRefusal(path)
    return True


def _n13_reuse_validation(path: str, recursive: bool, force: bool) -> bool:
    try:
        st = os.lstat(path)
    except OSError as exc:
        if not (force and fs._vanished(exc)):
            raise
        return False
    if fs._is_tree(st):
        if not recursive:
            raise fs._RmDirectoryRefusal(path)
        fs._rimraf(path)
    else:
        descend = fs._unlink_routed(path)
        if descend is not None:
            fs._rimraf(*descend)
    return False


def _n14_force_returns(path: str, recursive: bool, force: bool) -> bool:
    try:
        st = os.lstat(path)
    except OSError as exc:
        if force and fs._vanished(exc):
            return False
        raise
    if fs._is_tree(st) and not recursive:
        raise fs._RmDirectoryRefusal(path)
    return True


def _mutate(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if name == "N1":
        monkeypatch.setattr(fs, "_rimraf_entry", _n1_broad_catch)
    elif name == "N2":
        monkeypatch.setattr(fs, "_rmdir_first", _n2_retry_rmdir(fs._rmdir_first))
    elif name == "N3":
        monkeypatch.setattr(fs, "_rimraf_entry", _n3_swallow(fs._rimraf_entry))
    elif name == "N4":
        monkeypatch.setattr(fs, "_rimraf", _n4_ancestor(fs._rimraf))
    elif name == "N5":
        monkeypatch.setattr(fs, "_rmdir_first", _n5_list_first(fs._rmdir_first))
    elif name == "N6":
        monkeypatch.setattr(fs, "_rimraf_entry", _n6_no_lstat_recovery)
    elif name == "N7":
        monkeypatch.setattr(fs, "_fix_win_eperm", _recovery(own_error=True))
    elif name == "N8":
        monkeypatch.setattr(fs, "_fix_win_eperm", _recovery(conditional=True))
    elif name == "N9":
        monkeypatch.setattr(fs, "_fix_win_eperm", _recovery(keep_original=True))
    elif name == "N10":
        monkeypatch.setattr(fs, "_correct_own_attribute", _n10_follow_link)
    elif name == "N11":
        monkeypatch.setattr(fs, "_remove_sync", _remove(_n11_top_bypasses))
    elif name == "N12":
        monkeypatch.setattr(fs, "_remove_sync", _remove(_n12_validation_into_routing))
    elif name == "N13":
        monkeypatch.setattr(fs, "_remove_sync", _remove(_n13_reuse_validation))
    else:
        monkeypatch.setattr(fs, "_remove_sync", _remove(_n14_force_returns))


# Each control's intended rows, per platform (the episode record). A Windows-only recovery stage has
# no Linux row: pinned Pi itself makes no such call there.
CONTROLS: dict[str, dict[str, list[str]]] = {
    "N1": {"both": ["E1-child-rmdir-fails", "E2-inner-unlink-fails", "E3-child-readdir-fails",
                    "E4-child-final-rmdir-fails", "E5-child-rmdir-busy",
                    "E10-grandchild-unlink-fails", "E12-grandchild-rmdir-fails"]},
    "N2": {"both": ["E1-child-rmdir-fails"]},
    "N3": {"both": ["E2-inner-unlink-fails", "E10-grandchild-unlink-fails"]},
    "N4": {"both": ["E2-inner-unlink-fails", "E10-grandchild-unlink-fails"]},
    "N5": {"both": ["E5-child-rmdir-busy"]},
    "N6": {"win32": ["L3-chmod-fails", "L5-stat-fails"]},
    "N7": {"win32": ["L3-chmod-fails", "L5-stat-fails", "U2-chmod-fails",
                     "T2-single-file-chmod-fails"]},
    "N8": {"win32": ["U1-file-recovered", "U4-retry-fails-eio", "T1-single-file-recovered"]},
    "N9": {"win32": ["L7-retry-unlink-fails", "U4-retry-fails-eio"]},
    "N10": {"win32": ["L9-link-to-readonly-target", "U7-link-to-readonly-target"]},
    "N11": {"both": ["T1-single-file-recovered", "T2-single-file-chmod-fails"]},
    "N12": {"both": ["V1-validation-eperm-file", "V3-validation-eperm-dir"]},
    "N13": {"both": ["V2-classification-eperm-file", "V4-classification-eperm-dir",
                     "V10-classification-eio-file"]},
    "N14": {"both": ["V9-validation-enoent-force"]},
}  # fmt: skip
CONTROL_CASES = [
    (name, row_id)
    for name, rows in CONTROLS.items()
    for row_id in rows.get("both", []) + rows.get(PLATFORM or "", [])
]


@pytest.mark.parametrize(("control", "row_id"), CONTROL_CASES)
async def test_control_fails_its_intended_row(
    control: str, row_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mutate(control, monkeypatch)
    row = ROWS[row_id]
    assert await _observe(row, tmp_path, monkeypatch) != _expected(row)


def test_every_control_has_an_intended_row_on_this_platform_or_is_windows_only() -> None:
    covered = {name for name, _ in CONTROL_CASES}
    windows_only = {name for name, rows in CONTROLS.items() if set(rows) == {"win32"}}
    assert covered | windows_only == set(CONTROLS)
    assert covered == set(CONTROLS) or PLATFORM != "win32"


async def test_an_unlink_finding_the_entry_gone_counts_as_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 19.5 rule 2: rimraf's `unlink` callback treats ENOENT as success -- no recovery, no
    other call -- even though, here, the file is in fact still there."""
    row = {"fixture": ["tree"], "remove": {"path": "tree", "recursive": False},
           "inject": [{"op": "unlink", "rel": "tree", "nth": 1, "code": "ENOENT"}]}  # fmt: skip
    observed = await _observe(row, tmp_path, monkeypatch)
    assert (observed["result"], observed["remains"], observed["unfired"]) == ("ok", ["tree"], [])
    assert observed["calls"] == {"lstat tree": 2, "unlink tree": 1}
