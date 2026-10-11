"""Runner for `fs_path_domain` scenarios (`conformance/schema/fs-path-domain-scenario.schema.json`,
L12-D001 /
`EXEC-002`, `EXEC-003`: the filesystem path JavaScript-string domain through the Layer-12 `ctx.fs`
seam).

Thin by design: each case runs in a fresh working directory that is already its own realpath; each
step calls ONE
real `LocalFileSystem` operation (or `resolve()` for `target_key`) with the step's path, decoded
from UTF-16 code
units, and only OBSERVES the result -- path results as components relative to the working directory,
strings as
code units, errors as `{code, path}`. A `file_url_tail` path is the working directory's own file URL
plus the tail
(fixture construction). The runner never projects, normalizes or resolves on the provider's behalf.
"""

from __future__ import annotations

import errno
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Any

from minion_agent.execution import LocalFileSystem
from minion_agent.execution.filesystem import native_path, resolve_local_path
from minion_agent.execution.result import Ok
from minion_agent.runtime.signal import RunAbortController, RunSignal


def string(code_units: list[int]) -> str:
    return struct.pack(f"<{len(code_units)}H", *code_units).decode("utf-16-le", "surrogatepass")


def units(value: str) -> list[int]:
    data = value.encode("utf-16-le", "surrogatepass")
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def observe_path(cwd: str, value: str) -> dict[str, Any]:
    rel = os.path.relpath(value, cwd)
    if rel == ".":
        return {"components": []}
    if rel.startswith("..") or os.path.isabs(rel):
        return {"outside": True}
    return {"components": [units(part) for part in rel.split(os.sep)]}


def observe_error(cwd: str, error: Any) -> dict[str, Any]:
    path = error.path
    return {
        "error": str(error.code),
        "path": observe_path(cwd, path) if isinstance(path, str) else None,
    }


PLATFORM = "win32" if sys.platform == "win32" else "linux"


def applies(case: dict[str, Any]) -> bool:
    """An error-origin case may be declared for some platforms only (`platform_note` says why)."""
    return PLATFORM in case.get("platforms", ["linux", "win32"])


def _aborted_signal() -> RunSignal:
    controller = RunAbortController()
    controller.abort()
    return controller.signal


def path_argument(cwd: str, path: dict[str, Any]) -> str:
    if "utf16" in path:
        return string(path["utf16"])
    return f"{Path(cwd).as_uri()}/{string(path['file_url_tail'])}"


def _inside(base: str, path: str) -> bool:
    base, path = os.path.normcase(base), os.path.normcase(path)
    try:
        return os.path.commonpath([base, path]) == base and path != base
    except ValueError:  # different drives
        return False


# CE-L12D007-01 (AGREED FOR IMPLEMENTATION, revision 3): the containment proof is chosen by the
# OPERATION CLASS. REFERENT (follows the final link) proves the whole effective native path; ENTRY
# (a verified no-follow operation on the entry itself) proves the containing directory and never
# dereferences the final component. The R2 proof succeeds only at a completed traversal, a
# proven-missing component (R3) or a repeated, fully checked state; an exhausted budget, an
# outward hop, an unknown inspection outcome or a readlink failure REFUSE.
_BUDGET = 64
_MISSING_WIN32 = frozenset({2, 3, 123, 161, 267})


def _project(text: str) -> str:
    """Section 14.1's native projection, identical on Linux and Windows."""
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _inspect(path: str) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        if getattr(exc, "winerror", None) in _MISSING_WIN32:
            return None
        if exc.errno == errno.ENAMETOOLONG and _overlong_component(path):
            return None
        raise AssertionError(f"containment: cannot inspect {path!r} ({exc}); refused") from exc


def _query_limit(name: str, directory: str) -> int | None:
    try:
        value = os.pathconf(directory, name)
    except (OSError, ValueError, AttributeError):  # AttributeError: no pathconf (Windows)
        return None
    return value if isinstance(value, int) and value > 0 else None


def _overlong_component(path: str) -> bool:
    """R3a (CE-L12D007-01 revision 5): an ENAMETOOLONG proves the component missing only when its
    containing directory is proven (reached only after every earlier component was inspected and is
    not a link), its native UTF-8 bytes exceed the directory's queried NAME_MAX, and the whole
    path with its terminator fits a queried PATH_MAX (L + 1 <= PATH_MAX). No default limits."""
    directory, name = os.path.dirname(path), os.path.basename(path)
    name_max = _query_limit("PC_NAME_MAX", directory)
    if name_max is None or len(_project(name).encode("utf-8")) <= name_max:
        return False
    path_max = _query_limit("PC_PATH_MAX", directory)
    return path_max is not None and len(_project(path).encode("utf-8")) + 1 <= path_max


def _is_link(st: os.stat_result) -> bool:
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & 0x400) and getattr(
        st, "st_reparse_tag", 0
    ) in (0xA000000C, 0xA0000003)


def _prove(target: str, boundary: str, what: str) -> None:
    pending = os.path.abspath(_project(target))
    seen: set[str] = set()
    for _ in range(_BUDGET):
        if not (
            _inside(boundary, pending) or os.path.normcase(pending) == os.path.normcase(boundary)
        ):
            raise AssertionError(f"containment: {what} reaches {pending!r}, outside the case")
        if os.path.normcase(pending) in seen:
            return  # a repeated, fully checked state: a contained cycle
        seen.add(os.path.normcase(pending))
        drive, rest = os.path.splitdrive(pending)
        parts = [part for part in rest.replace("/", os.sep).split(os.sep) if part]
        current, redirected = drive, None
        for i, part in enumerate(parts):
            current = current + os.sep + part
            st = _inspect(current)
            if st is None:
                break  # proven missing
            if _is_link(st):
                try:
                    text = os.readlink(current)
                except OSError as exc:
                    raise AssertionError(f"containment: cannot read link {current!r}") from exc
                text = text[4:] if text.startswith("\\\\?\\") else text
                # ".." in link text resolves against the RESOLVED predecessor (POSIX 4.13); a
                # lexical collapse could erase an unchecked link ("b/../leaf"): refused.
                if ".." in text.replace("\\", "/").split("/"):
                    raise AssertionError(f"containment: link {current!r} text has '..'; refused")
                base = text if os.path.isabs(text) else os.path.dirname(current) + os.sep + text
                redirected = os.path.abspath(os.sep.join([base, *parts[i + 1 :]]))
                break
        if redirected is None:
            return  # completed traversal, or proven missing
        if not _inside(boundary, redirected):
            raise AssertionError(f"containment: {what} reaches {redirected!r} through a link")
        pending = redirected
    raise AssertionError(f"containment: {what}: budget exhausted without a repeated state")


def _raw_form(p: str) -> None:
    """The Owner's prohibited raw forms, refused BEFORE any normalization."""
    if (
        not p
        or (len(p) >= 2 and p[0].isalpha() and p[1] == ":")
        or p[0] in "\\/"
        or ".." in p.replace("\\", "/").split("/")
    ):
        raise AssertionError(f"fixture target {p!r} is a prohibited raw form")


def _fixture_target(cwd: str, p: str, start: str | None = None) -> str:
    """REFERENT proof for a fixture step's target (or a symlink's text, resolved from `start`, the
    link's own directory). String concatenation, never `os.path.join` (an `X:` component would
    be a drive)."""
    _raw_form(p)
    target = os.path.abspath((start or cwd) + os.sep + p)
    if not _inside(cwd, target):
        raise AssertionError(f"fixture target {p!r} escapes the case directory")
    _prove(target, cwd, repr(p))
    return target


def _fixture_entry(cwd: str, p: str) -> str:
    """ENTRY proof for a no-follow operation on the entry itself (link creation): the containing
    directory is proven, the final component is not dereferenced."""
    _raw_form(p)
    target = os.path.abspath(cwd + os.sep + p)
    if not _inside(cwd, target):
        raise AssertionError(f"fixture entry {p!r} escapes the case directory")
    parent = os.path.dirname(target)
    if os.path.normcase(parent) != os.path.normcase(cwd):
        _prove(parent, cwd, f"parent of {p!r}")
    return target


# R5: provider operations under test. Their ORIGINAL argument goes to the provider unchanged; the
# native path the provider will touch (Pi's logical resolution, then the section-14.1 projection) is
# proven by the operation's class. Read-only operations mutate nothing and need no proof.
_PROVIDER_REFERENT = frozenset({"write_file", "append_file", "create_dir"})
_PROVIDER_ENTRY = frozenset({"remove", "rename_file"})


def _provider_target(cwd: str, argument: str, op: str) -> None:
    native = native_path(resolve_local_path(cwd, argument))
    if "\0" in native:
        # Section 18 (certified): every native call rejects a NUL-containing argument before
        # touching the filesystem; the one earlier effect is the parent creation of write/append
        # when the NUL is only in the final component. So the provider can mutate at most the
        # NUL-free directory prefix: that directory is proven; the NUL-containing remainder is never
        # inspected (it cannot be).
        prefix = native[: native.index("\0")]
        directory = (
            prefix[: max(prefix.rfind("\\"), prefix.rfind("/"))] if os.sep in prefix else cwd
        )
        if os.path.normcase(directory) != os.path.normcase(cwd):
            if not _inside(cwd, directory):
                raise AssertionError(f"containment: provider {op} prefix {directory!r} is outside")
            _prove(directory, cwd, f"provider {op} NUL-free prefix")
        return
    if not _inside(cwd, native):
        raise AssertionError(f"containment: provider {op} target {native!r} is outside the case")
    if op in _PROVIDER_REFERENT:
        _prove(native, cwd, f"provider {op}")
    else:
        parent = os.path.dirname(native)
        if os.path.normcase(parent) != os.path.normcase(cwd):
            _prove(parent, cwd, f"provider {op} parent")


def _restore_access(cwd: str, rel: str, kind: str, restore: Any) -> None:
    """R6: undo a deny only on an EXISTING, NON-LINK entry of the recorded kind, re-proven now;
    otherwise skip (the case directory is then left as is). A restore that RUNS and fails is never
    swallowed: its error propagates and fails the case (closure review 1)."""
    try:
        target = _fixture_target(cwd, rel)
    except AssertionError:
        return
    st = _inspect(target)
    if st is None or _is_link(st) or (kind == "directory") != stat.S_ISDIR(st.st_mode):
        return
    restore(target)


def _deny_access(cwd: str, rel: str, target: str) -> Any:
    """L12D007-C002 `deny_access`: Windows denies Everyone every read right except read-control
    (`icacls /deny *S-1-1-0:(RD,REA,RA,S)`: a generic (R) deny also denies READ_CONTROL, and under
    Python 3.13's protected 0o700 temp-directory ACL the owner could then never undo it); POSIX
    removes every mode bit (meaningful for a non-root user). Undone at case end by
    `_restore_access`."""
    kind = "directory" if os.path.isdir(target) else "file"
    if sys.platform == "win32":  # pragma: no cover -- per-platform fixture
        import subprocess

        subprocess.run(
            ["icacls", target, "/deny", "*S-1-1-0:(RD,REA,RA,S)"], check=True, capture_output=True
        )
        return lambda: _restore_access(
            cwd,
            rel,
            kind,
            lambda t: subprocess.run(
                ["icacls", t, "/remove:d", "*S-1-1-0"], check=True, capture_output=True
            ),
        )
    mode = os.stat(target).st_mode & 0o777  # pragma: no cover -- per-platform fixture
    os.chmod(target, 0)  # pragma: no cover

    def restore() -> None:  # pragma: no cover
        _restore_access(cwd, rel, kind, lambda t: os.chmod(t, mode))

    return restore  # pragma: no cover


def _hold(op: str, target: str) -> Any:  # pragma: no cover -- win32-only fixture
    """`hold_exclusive`: a FileShare.None handle; `lock_range`: a shared handle with bytes 0-63
    locked. Both are held until the case ends, so every provider call meets the sharing / lock
    violation."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    k32.CreateFileW.restype = wintypes.HANDLE
    share = 0 if op == "hold_exclusive" else 3  # none / read+write (no delete), as the schema says
    # RW, OPEN_EXISTING; FILE_FLAG_BACKUP_SEMANTICS so a DIRECTORY can be held too (L12D007-C002).
    handle = k32.CreateFileW(target, 0xC0000000, share, None, 3, 0x80 | 0x02000000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), f"{op} fixture open failed", target)
    if op == "lock_range" and not k32.LockFile(handle, 0, 0, 64, 0):
        raise OSError(ctypes.get_last_error(), "lock_range fixture lock failed", target)
    return lambda: k32.CloseHandle(handle)


async def run_case(case: dict[str, Any], root: Path) -> list[Any]:
    releases: list[Any] = []
    try:
        return await _run_steps(case, root, releases)
    finally:
        for release in releases:  # pragma: no cover -- win32-only fixtures
            release()


async def _run_steps(case: dict[str, Any], root: Path, releases: list[Any]) -> list[Any]:
    cwd = os.path.realpath(root)
    fs = LocalFileSystem(cwd)
    observed: list[Any] = []
    for step in case["steps"]:
        p = path_argument(cwd, step["path"])
        op = step["op"]
        if op == "make_symlink":
            link = _fixture_entry(cwd, p)  # ENTRY: creating a link writes only its entry
            _fixture_target(cwd, string(step["to"]["utf16"]), os.path.dirname(link))
            os.symlink(string(step["to"]["utf16"]), link)
            observed.append({"ok": None})
            continue
        if op in ("hold_exclusive", "lock_range"):  # pragma: no cover -- win32-only fixtures
            releases.append(_hold(op, _fixture_target(cwd, p)))
            observed.append({"ok": None})
            continue
        if op == "deny_access":
            releases.append(_deny_access(cwd, p, _fixture_target(cwd, p)))
            observed.append({"ok": None})
            continue
        if op in _PROVIDER_REFERENT or op in _PROVIDER_ENTRY:  # R5: prove what the provider touches
            _provider_target(cwd, p, op)
            if op == "rename_file":
                _provider_target(cwd, path_argument(cwd, step["to"]), op)
        # L12-D006: `aborted` hands the operation an already-aborted signal (fixture construction).
        signal = _aborted_signal() if step.get("aborted") else None
        if op == "write_file":
            r: Any = await fs.write_file(p, string(step["content"]), signal=signal)
            o: Any = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "read_text_file":
            r = await fs.read_text_file(p, signal=signal)
            o = {"ok": units(r.value)} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "list_dir":
            r = await fs.list_dir(p, signal=signal)
            o = (
                {"names": sorted(units(i.name) for i in r.value)}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "canonical_path":
            r = await fs.canonical_path(p)
            o = observe_path(cwd, r.value) if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "absolute_path":
            r = await fs.absolute_path(p)
            assert isinstance(r, Ok)
            o = (
                {"last": units(os.path.basename(r.value))}
                if step.get("observe") == "last_component"
                else observe_path(cwd, r.value)
            )
        elif op == "target_key":
            r = await fs.resolve(p)
            o = (
                observe_path(cwd, r.value.target_key)
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "exists":
            r = await fs.exists(p)
            o = {"ok": r.value} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "file_info":
            r = await fs.file_info(p)
            o = (
                {"kind": str(r.value.kind), "name": units(r.value.name)}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "append_file":
            r = await fs.append_file(p, string(step["content"]))
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "read_text_lines":
            r = await fs.read_text_lines(p, max_lines=step.get("max_lines"), signal=signal)
            o = (
                {"ok": [units(line) for line in r.value]}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "read_binary_file":
            r = await fs.read_binary_file(p, signal=signal)
            o = {"ok": list(r.value)} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "create_dir":
            r = await fs.create_dir(p, recursive=step["recursive"])
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "remove":
            r = await fs.remove(
                p, recursive=step.get("recursive", False), force=step.get("force", False)
            )
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "rename_file":
            r = await fs.rename_file(p, path_argument(cwd, step["to"]), signal=signal)
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "list_dir_raw":
            r = await fs.list_dir_raw(p)
            o = (
                {"names": sorted(units(name) for name in r.value)}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "probe_dir_entry":
            r = await fs.probe_dir_entry(p)
            o = (
                {"kind": str(r.value.kind), "name": units(r.value.name)}
                if isinstance(r, Ok)
                else observe_error(cwd, r.error)
            )
        elif op == "check_readable":
            r = await fs.check_readable(p)
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        elif op == "check_read_write":
            r = await fs.check_read_write(p)
            o = {"ok": None} if isinstance(r, Ok) else observe_error(cwd, r.error)
        else:  # pragma: no cover -- the schema closes the op set
            raise AssertionError(f"unknown op {op}")
        observed.append(o)
    return observed


def expected(step: dict[str, Any]) -> Any:
    """`expect`, or this platform's answer where pinned Node itself differs by platform."""
    return step["expect"] if "expect" in step else step["expect_by_platform"][PLATFORM]


def check(case: dict[str, Any], observed: list[Any]) -> None:
    for index, (step, got) in enumerate(zip(case["steps"], observed, strict=True)):
        want = expected(step)
        assert got == want, (case["id"], index, step.get("op", step.get("tool")), got, want)


async def run_tool_case(case: dict[str, Any], root: Path) -> list[Any]:
    """One `fs_path_tools` case: each step is ONE real built-in tool call
    (`write`/`read`/`ls`/`edit` over the real
    `LocalFileSystem`), observed as `{is_error, text}` with the text as code units."""
    from minion_agent.llm import TextBlock, ToolCallBlock
    from minion_agent.runtime import Context
    from minion_agent.tools.builtin import (
        create_edit_tool,
        create_ls_tool,
        create_read_tool,
        create_write_tool,
    )
    from minion_agent.tools.events import declare_tools_events
    from minion_agent.tools.execute import execute_call
    from minion_agent.tools.registry import ToolRegistry

    fs = LocalFileSystem(os.path.realpath(root))
    registry = ToolRegistry()
    for create in (create_write_tool, create_read_tool, create_ls_tool, create_edit_tool):
        registry.register(create(fs))
    observed = []
    for index, step in enumerate(case["steps"]):
        ctx = Context()
        declare_tools_events(ctx.events)
        call = ToolCallBlock(
            id=f"c{index}", name=step["tool"], arguments=_decode(step["arguments"])
        )
        result = await execute_call(call, registry=registry, ctx=ctx)
        first = result.content[0] if result.content else None
        body = first.text if isinstance(first, TextBlock) else ""
        observed.append({"is_error": result.is_error, "text": {"utf16": units(body)}})
    return observed


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if "utf16" in value:
            return string(value["utf16"])
        return {key: _decode(item) for key, item in value.items()}
    return value
