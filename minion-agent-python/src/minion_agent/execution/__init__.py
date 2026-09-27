"""Execution capability seams (Layer 12, WP-12.1): `ctx.fs`, `ctx.shell`, `ctx.subprocess`, the
`FsTarget` bridge, and execution-world compatibility. See `minion-agent-docs/spec/execution.md`
and `pi-parity-manifest.yaml` rows `EXEC-001` through `EXEC-006` for the full normative contract.
"""

from __future__ import annotations

from .errors import (
    FsError,
    FsErrorCode,
    ShellError,
    ShellErrorCode,
    SubprocessError,
    SubprocessErrorCode,
)
from .filesystem import (
    FileInfo,
    FileKind,
    FileSystem,
    FsTarget,
    LocalFileSystem,
    file_url_to_path,
    resolve_local_path,
)
from .plugin import fs_plugin, shell_plugin, subprocess_plugin
from .result import Err, Ok, Result, is_err, is_ok
from .shell import LocalShell, Shell, ShellResult
from .subprocess import (
    ExitStatus,
    LocalSubprocess,
    Process,
    ReadableStream,
    SpawnOptions,
    StdioMode,
    Subprocess,
    WritableStream,
)
from .world import (
    ExecutionWorldError,
    ExecutionWorldIdentity,
    IncompatiblePair,
    compatible,
    validate,
)

__all__ = [
    "Err",
    "ExecutionWorldError",
    "ExecutionWorldIdentity",
    "ExitStatus",
    "FileInfo",
    "FileKind",
    "FileSystem",
    "FsError",
    "FsErrorCode",
    "FsTarget",
    "IncompatiblePair",
    "LocalFileSystem",
    "LocalShell",
    "LocalSubprocess",
    "Ok",
    "Process",
    "ReadableStream",
    "Result",
    "Shell",
    "ShellError",
    "ShellErrorCode",
    "ShellResult",
    "SpawnOptions",
    "StdioMode",
    "Subprocess",
    "SubprocessError",
    "SubprocessErrorCode",
    "WritableStream",
    "compatible",
    "file_url_to_path",
    "fs_plugin",
    "is_err",
    "is_ok",
    "resolve_local_path",
    "shell_plugin",
    "subprocess_plugin",
    "validate",
]
