"""Node's `path` functions for the execution world's platform -- `path.win32` on WINDOWS,
`path.posix` on POSIX -- as WP-13.4 uses them (spec/tools.md WP-13.4 "Path functions"). Pure string
functions; nothing touches the filesystem.

Where Python's `ntpath`/`posixpath` differ from Node, Node wins: `relative` of equal paths is `""`
(not `"."`), a cross-drive `win32.relative` returns the resolved target (Python raises), the win32
comparison is case-insensitive (as Node lowercases both sides), and `win32.isAbsolute` accepts a
rooted path without a drive (`\\x`), which Python 3.13's `ntpath.isabs` no longer does.
"""

from __future__ import annotations

import ntpath
import posixpath
from dataclasses import dataclass

from ...execution import Platform


@dataclass(frozen=True, slots=True)
class NodePath:
    platform: Platform

    @property
    def windows(self) -> bool:
        return self.platform is Platform.WINDOWS

    @property
    def sep(self) -> str:
        return "\\" if self.windows else "/"

    def is_absolute(self, path: str) -> bool:
        if not self.windows:
            return path.startswith("/")
        if path[:1] in ("/", "\\"):
            return True
        return len(path) > 2 and path[0].isalpha() and path[1] == ":" and path[2] in ("/", "\\")

    def dirname(self, path: str) -> str:
        return ntpath.dirname(path) if self.windows else posixpath.dirname(path)

    def join(self, *parts: str) -> str:
        return ntpath.join(*parts) if self.windows else posixpath.join(*parts)

    def basename(self, path: str) -> str:
        if self.windows:
            return ntpath.basename(path.rstrip("\\/") or path)
        return posixpath.basename(path.rstrip("/") or path)

    def relative(self, start: str, target: str) -> str:
        if self.windows:
            start_n, target_n = ntpath.normpath(start), ntpath.normpath(target)
            if start_n.lower() == target_n.lower():
                return ""
            try:
                result = ntpath.relpath(target_n, start_n)
            except ValueError:  # different drives: Node returns the resolved target
                return target_n
            return "" if result == "." else result
        start_n, target_n = posixpath.normpath(start), posixpath.normpath(target)
        if start_n == target_n:
            return ""
        result = posixpath.relpath(target_n, start_n)
        return "" if result == "." else result
