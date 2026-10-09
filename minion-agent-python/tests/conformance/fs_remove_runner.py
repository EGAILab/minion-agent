"""L12-D005 (`minion-agent#188`) runner for `conformance/agent/fs-remove-readonly/*.json`.

Thin by construction: the fixture is prepared natively (files, directories, the platform
read-only attribute, symlinks, Windows ACL denials), then the binding's REAL
`LocalFileSystem.remove` is called once. The runner reports the Result, the entries left under
the case cwd and each external path's state. It never removes anything itself, and it restores
attributes and ACLs afterwards so the temporary tree can be cleaned up.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minion_agent.execution import LocalFileSystem, is_ok

WIN = sys.platform == "win32"


def applies(document: dict[str, Any]) -> bool:
    return sys.platform in document["platforms"]


def needs_unprivileged(document: dict[str, Any]) -> bool:
    """A POSIX permission case cannot be observed as root, which bypasses permission checks."""
    return (
        not WIN
        and os.geteuid() == 0
        and any("readonly" in step for step in document["fs_remove"]["fixture"])
        and document["fs_remove"]["expect"] != {"ok": True}
    )


def _native(cwd: Path, rel: str) -> Path:
    return cwd.joinpath(*rel.split("/"))


def _run(*args: str) -> None:
    subprocess.run(list(args), check=True, capture_output=True)


def _quiet(*args: str) -> None:
    subprocess.run(list(args), capture_output=True)


def _prepare(cwd: Path, step: dict[str, Any], cleanups: list[Callable[[], None]]) -> None:
    if "file" in step:
        target = _native(cwd, step["file"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(step.get("text", "x"), encoding="utf-8")
    elif "dir" in step:
        _native(cwd, step["dir"]).mkdir(parents=True, exist_ok=True)
    elif "readonly" in step:
        target = _native(cwd, step["readonly"])
        if WIN:
            _run("attrib", "+R", str(target))
            cleanups.append(lambda t=target: _quiet("attrib", "-R", str(t)))
        else:
            directory = target.is_dir()
            target.chmod(0o555 if directory else 0o444)
            cleanups.append(lambda t=target, d=directory: t.chmod(0o755 if d else 0o644))
    elif "readonly_link" in step:
        target = _native(cwd, step["readonly_link"])
        _run("attrib", "+R", "/L", str(target))
        cleanups.append(lambda t=target: _quiet("attrib", "-R", "/L", str(t)))
    elif "symlink" in step:
        link = _native(cwd, step["symlink"])
        link.symlink_to(_native(cwd, step["to"]), target_is_directory=step["kind"] == "dir")
    elif "deny_delete" in step:
        target = _native(cwd, step["deny_delete"])
        user = getpass.getuser()
        _run("icacls", str(target), "/deny", f"{user}:(D)")
        _run("icacls", str(target.parent), "/deny", f"{user}:(DC)")
        cleanups.append(lambda t=target, u=user: _quiet("icacls", str(t), "/remove:d", u))
        cleanups.append(lambda t=target, u=user: _quiet("icacls", str(t.parent), "/remove:d", u))
    elif "deny_write_attributes" in step:
        target = _native(cwd, step["deny_write_attributes"])
        user = getpass.getuser()
        _run("icacls", str(target), "/deny", f"{user}:(WA)")
        cleanups.append(lambda t=target, u=user: _quiet("icacls", str(t), "/remove:d", u))
    else:
        raise ValueError(f"unknown fixture step {step!r}")


def _left(directory: Path, base: Path) -> list[str]:
    out: list[str] = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return out
    for name in names:
        entry = directory / name
        rel = entry.relative_to(base).as_posix()
        if entry.is_symlink():
            out.append(rel + "@")
        elif entry.is_dir():
            out.append(rel + "/")
            out.extend(_left(entry, base))
        else:
            out.append(rel)
    return out


def _readonly(target: Path) -> bool:
    if WIN:
        return bool(os.stat(target).st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)
    return not os.stat(target).st_mode & 0o222


def _external(cwd: Path, rel: str) -> dict[str, Any]:
    target = _native(cwd, rel)
    if not target.exists():
        return {"path": rel, "exists": False}
    state: dict[str, Any] = {"path": rel, "exists": True, "readonly": _readonly(target)}
    if target.is_file():
        state["text"] = target.read_text(encoding="utf-8")
    return state


async def run_case(document: dict[str, Any], cwd: Path) -> dict[str, Any]:
    case = document["fs_remove"]
    cleanups: list[Callable[[], None]] = []
    try:
        for step in case["fixture"]:
            _prepare(cwd, step, cleanups)
        result = await LocalFileSystem(str(cwd)).remove(
            case["remove"]["path"], recursive=case["remove"]["recursive"]
        )
        if is_ok(result):
            expect: dict[str, Any] = {"ok": True}
        else:
            error = result.error
            path = (
                list(Path(os.path.relpath(error.path, cwd)).parts)
                if error.path is not None
                else None
            )
            expect = {"error": error.code.value, "path": path}
        observed: dict[str, Any] = {"expect": expect, "expect_left": _left(cwd, cwd)}
        if "expect_external" in case:
            observed["expect_external"] = [
                _external(cwd, e["path"]) for e in case["expect_external"]
            ]
        return observed
    finally:
        for cleanup in reversed(cleanups):
            # The remove under test may have deleted the entry a cleanup restores.
            with contextlib.suppress(FileNotFoundError):
                cleanup()
