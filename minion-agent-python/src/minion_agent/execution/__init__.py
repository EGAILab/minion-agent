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
    resolve_local_path,
)
from .result import Err, Ok, Result, is_err, is_ok
from .shell import LocalShell, Shell, ShellResult
from .subprocess import (
    ExitStatus,
    LocalSubprocess,
    Process,
    ReadableStream,
    SpawnOptions,
    StdioMode,
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
    "SubprocessError",
    "SubprocessErrorCode",
    "WritableStream",
    "compatible",
    "is_err",
    "is_ok",
    "resolve_local_path",
    "validate",
]
