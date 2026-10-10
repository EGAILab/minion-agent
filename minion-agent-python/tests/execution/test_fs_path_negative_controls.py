"""L12-D001 negative controls (Owner decision FSP-Q001 section 19): each realistic WRONG
implementation of the
filesystem path JavaScript-string domain, installed at the seam it would live in, must make the
canonical corpus
(`conformance/agent/fs-path-domain/`, ctx.fs and tool-level cases) FAIL, while the unmodified code
passes.

Every mutant is a runtime monkeypatch of production code; the runner and the scenarios are
unchanged."""

from __future__ import annotations

import dataclasses
import functools
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import filesystem as fs_module
from minion_agent.execution.filesystem import FsTarget, LocalFileSystem, resolve_local_path
from minion_agent.execution.result import Err, Ok
from minion_agent.tools import builtin
from minion_agent.tools.builtin.paths import BuiltinToolError

from ..conformance import fs_path_runner as runner
from ..conformance.test_fs_path_conformance import CASES, TOOL_CASES, _l12_d006_pending

LONE = range(0xD800, 0xE000)


def _has_lone(text: str) -> bool:
    return any(ord(c) in LONE for c in text)


async def _failures(tmp_path: Path) -> list[str]:
    failed = []
    for index, case in enumerate([*CASES, *TOOL_CASES]):
        if not runner.applies(case):
            continue
        if _l12_d006_pending(case["id"]):
            continue  # L12-D006 contract stage: strict xfail in the corpus (removed with the fix)
        root = tmp_path / str(index)
        root.mkdir()
        try:
            run = runner.run_tool_case if case in TOOL_CASES else runner.run_case
            runner.check(case, await run(case, root))
        except Exception:  # an assertion mismatch or an escaping exception both fail the case
            failed.append(case["id"])
    return failed


async def test_the_unmodified_code_passes_every_case(tmp_path: Path) -> None:
    assert await _failures(tmp_path) == []


def _wrap_tools(monkeypatch: pytest.MonkeyPatch, around: Callable[[str, Any], Any]) -> None:
    """Replace each built-in tool factory with one whose `execute` is `around(tool_name, original)`;
    the original's
    signature is kept (`functools.wraps`), so the pipeline's arity dispatch is unchanged."""
    for factory_name in (
        "create_write_tool",
        "create_read_tool",
        "create_ls_tool",
        "create_edit_tool",
    ):
        original_factory = getattr(builtin, factory_name)

        def factory(fs: Any, *args: Any, _original: Any = original_factory, **kwargs: Any) -> Any:
            definition = _original(fs, *args, **kwargs)
            return dataclasses.replace(
                definition, execute=around(definition.name, definition.execute)
            )

        monkeypatch.setattr(builtin, factory_name, factory)


def _project_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Early tool-level U+FFFD conversion: the tool projects its path argument on entry."""

    def around(name: str, execute: Any) -> Any:
        @functools.wraps(execute)
        async def wrapped(tool_call_id: str, arguments: dict[str, Any], *rest: Any) -> Any:
            arguments = {
                **arguments,
                "path": fs_module.scalar_value_string(arguments.get("path") or ""),
            }
            return await execute(tool_call_id, arguments, *rest)

        return wrapped

    _wrap_tools(monkeypatch, around)


def _project_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool-authored text rewritten to U+FFFD while the filesystem operation is correct."""

    def around(name: str, execute: Any) -> Any:
        @functools.wraps(execute)
        async def wrapped(*args: Any) -> Any:
            result = await execute(*args)
            blocks = tuple(
                dataclasses.replace(b, text=fs_module.scalar_value_string(b.text))
                if hasattr(b, "text")
                else b
                for b in result.content
            )
            return dataclasses.replace(result, content=blocks)

        return wrapped

    _wrap_tools(monkeypatch, around)


def _refuse_before_fs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rust's former behavior: a non-scalar path is refused ("path is required") before ctx.fs."""

    def around(name: str, execute: Any) -> Any:
        @functools.wraps(execute)
        async def wrapped(tool_call_id: str, arguments: dict[str, Any], *rest: Any) -> Any:
            if _has_lone(arguments.get("path") or ""):
                raise BuiltinToolError("path is required")
            return await execute(tool_call_id, arguments, *rest)

        return wrapped

    _wrap_tools(monkeypatch, around)


def _ls_dot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rust's former ls behavior: a non-scalar path silently becomes "."."""

    def around(name: str, execute: Any) -> Any:
        @functools.wraps(execute)
        async def wrapped(tool_call_id: str, arguments: dict[str, Any], *rest: Any) -> Any:
            if name == "ls" and _has_lone(arguments.get("path") or ""):
                arguments = {**arguments, "path": "."}
            return await execute(tool_call_id, arguments, *rest)

        return wrapped

    _wrap_tools(monkeypatch, around)


def _raise_like_posix(path: str) -> str:
    """Python's former POSIX behavior: the host codec refuses an unpaired surrogate (escaping the
    Result)."""
    path.encode("utf-8")
    return path


def _project_all_surrogates(path: str) -> str:
    """A valid pair wrongly replaced too (every surrogate code unit -> U+FFFD)."""
    data = path.encode("utf-16-le", "surrogatepass")
    out = []
    for i in range(0, len(data), 2):
        unit = data[i] | data[i + 1] << 8
        out.append("�" if unit in LONE else chr(unit))
    return "".join(out)


def _project_last_component_only(path: str) -> str:
    """A raw surrogate leaked to the native path: only the final component is projected."""
    head, tail = os.path.split(path)
    return os.path.join(head, fs_module.scalar_value_string(tail))


def _always_project_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """always-project-before-target_key: the missing-path fallback key is projected too."""
    original = LocalFileSystem.resolve

    async def resolve(self: LocalFileSystem, path: str, signal: Any = None) -> Any:
        result = await original(self, path, signal)
        if isinstance(result, Ok):
            return Ok(
                FsTarget(target_key=fs_module.native_path(result.value.target_key), _provider=self)
            )
        return result

    monkeypatch.setattr(LocalFileSystem, "resolve", resolve)


def _never_project_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """never-project-before-target_key: canonicalization reports the LOGICAL path (no
    projection)."""
    original = LocalFileSystem.canonical_path

    async def canonical_path(self: LocalFileSystem, path: str, signal: Any = None) -> Any:
        result = await original(self, path, signal)
        if isinstance(result, Err):
            return result
        return Ok(resolve_local_path(self.cwd, path))

    monkeypatch.setattr(LocalFileSystem, "canonical_path", canonical_path)


def _report_requested_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """`L12-D001-R001`: a write/append failure names the projected full target (the requested
    child) instead of the path of the native call that failed."""
    for name in ("write_file", "append_file"):
        original = getattr(LocalFileSystem, name)

        async def wrapped(
            self: LocalFileSystem, path: str, *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            result = await _original(self, path, *args, **kwargs)
            if isinstance(result, Err):
                target = fs_module.native_path(resolve_local_path(self.cwd, path))
                return Err(dataclasses.replace(result.error, path=target))
            return result

        monkeypatch.setattr(LocalFileSystem, name, wrapped)


MUTANTS: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "early-tool-level-fffd-conversion": _project_arguments,
    "tool-message-changed-to-fffd": _project_messages,
    "rejection-before-ctx-fs": _refuse_before_fs,
    "ls-dot-substitution": _ls_dot,
    "raw-os-passthrough": lambda m: m.setattr(fs_module, "native_path", lambda p: p),
    "posix-unicode-encode-error": lambda m: m.setattr(fs_module, "native_path", _raise_like_posix),
    "valid-pair-replaced": lambda m: m.setattr(fs_module, "native_path", _project_all_surrogates),
    "raw-surrogate-leaked-in-directory-component": lambda m: m.setattr(
        fs_module, "native_path", _project_last_component_only
    ),
    "always-project-before-target-key": _always_project_key,
    "never-project-before-target-key": _never_project_key,
    # L12-D001-R001 (Codex contract review 1): which path an OS-originated failure names.
    "error-path-is-the-requested-target": _report_requested_target,
    "os-makedirs-walk": lambda m: m.setattr(
        fs_module, "_node_mkdirp", lambda p: os.makedirs(p, exist_ok=True)
    ),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
async def test_a_wrong_implementation_fails_the_corpus(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    MUTANTS[name](monkeypatch)
    assert await _failures(tmp_path), f"{name} survived the L12-D001 corpus"
