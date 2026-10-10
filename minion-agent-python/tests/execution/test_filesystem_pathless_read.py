"""L12D007-I001: a read failure whose error names no path reports pinned Pi's fallback, the resolved
LOGICAL path (`toFileError(error, resolved)`); a failure that names its path keeps the native origin
(section 14.2). The logical spelling `lone-<U+D800>` projects natively to `lone-<U+FFFD>`, so the
two answers differ on every platform."""

from __future__ import annotations

import errno
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import Err, FsErrorCode, LocalFileSystem
from minion_agent.execution import filesystem as fs_module
from minion_agent.execution.filesystem import native_path, resolve_local_path

LOGICAL = "lone-\ud800"
NATIVE_NAME = "lone-�"
READS = {
    "read_text_file": "_read_text_sync",
    "read_binary_file": "_read_binary_sync",
    "read_text_lines": "_read_text_lines_sync",
}


def _failing_worker(error: OSError) -> Callable[..., Any]:
    def worker(*args: Any) -> Any:
        raise error

    return worker


async def _observed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> dict[str, str | None]:
    """Each read's reported error path when its worker raises `error`."""
    observed: dict[str, str | None] = {}
    fs = LocalFileSystem(str(tmp_path))
    for method, seam in READS.items():
        with monkeypatch.context() as patch:
            patch.setattr(fs_module, seam, _failing_worker(error))
            result = await getattr(fs, method)(LOGICAL)
        assert isinstance(result, Err)
        observed[method] = result.error.path
    return observed


async def test_a_pathless_read_failure_reports_the_logical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = resolve_local_path(str(tmp_path), LOGICAL)
    assert native_path(resolved) != resolved
    observed = await _observed_paths(tmp_path, monkeypatch, OSError(errno.EIO, "I/O error"))
    assert observed == dict.fromkeys(READS, resolved)


async def test_a_read_failure_naming_its_path_keeps_the_native_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open-error control: an error that names the path reports its native spelling."""
    native = native_path(resolve_local_path(str(tmp_path), LOGICAL))
    error = PermissionError(errno.EACCES, "Permission denied", native)
    assert await _observed_paths(tmp_path, monkeypatch, error) == dict.fromkeys(READS, native)


@contextmanager
def _range_locked(target: str) -> Iterator[None]:
    from tests.conformance.fs_path_runner import _hold

    release = _hold("lock_range", target)
    try:
        yield
    finally:
        release()


@pytest.mark.skipif(sys.platform != "win32", reason="mandatory byte-range locks are Windows-only")
async def test_a_byte_range_locked_read_reports_the_logical_path(tmp_path: Path) -> None:
    """Real host, no injected error: under a byte-range lock `ReadFile` fails with Win32 33, which
    libuv reports without a path. Pinned Pi (`.tmp/codex-scratch/l12d007-impl-pathless-native.mjs`,
    review issuecomment-6102670593): `unknown`, path ending in the logical `lone-<U+D800>`."""
    from tests.conformance.fs_path_runner import _fixture_target

    target = _fixture_target(str(tmp_path), NATIVE_NAME)
    Path(target).write_bytes(b"data")
    fs = LocalFileSystem(str(tmp_path))
    resolved = resolve_local_path(str(tmp_path), LOGICAL)
    with _range_locked(target):
        results = {method: await getattr(fs, method)(LOGICAL) for method in READS}
    for method, result in results.items():
        assert isinstance(result, Err), method
        assert (result.error.code, result.error.path) == (FsErrorCode.UNKNOWN, resolved), method


async def test_control_projecting_the_fallback_fails_the_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutant: the path-less fallback projected natively (the reviewed defect). The witness's
    expectation no longer holds for any of the three reads."""
    monkeypatch.setattr(fs_module, "_failing_path", lambda exc, native, resolved: native)
    resolved = resolve_local_path(str(tmp_path), LOGICAL)
    observed = await _observed_paths(tmp_path, monkeypatch, OSError(errno.EIO, "I/O error"))
    assert all(path != resolved for path in observed.values())
