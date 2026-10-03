"""`WP-12.E4` (`EXEC-010`, spec/execution.md section 15): the execution world's platform family and
its environment snapshot. `MINION_EXTENSION` -- additive capabilities of the `ctx.subprocess`
provider (section 6 is otherwise unchanged).

An `EnvSnapshot` is read-only and isolated (section 15.1, `WP12E4-CON-R001`): it exposes no
mutator, a later change to the provider's baseline does not reach it, and a consumer edits its own
`copy()`. It is lossless with respect to the baseline (section 15.3): on POSIX each name and value
is the `str` Python's `os.fsencode` maps back to the native bytes (surrogateescape); on WINDOWS it
is the `str` of the native UTF-16 code units, an unpaired surrogate kept as a surrogate code point.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from collections.abc import Iterable, Iterator, Mapping
from enum import StrEnum
from importlib import resources
from typing import Any


class Platform(StrEnum):
    """The execution world's platform family -- closed for this extension (Owner F2)."""

    WINDOWS = "windows"
    POSIX = "posix"


def host_platform() -> Platform:
    """The family a LOCAL provider declares: its host's. A remote or fake provider declares its
    own world's family instead (section 15.2) -- never this."""
    return Platform.WINDOWS if os.name == "nt" else Platform.POSIX


def utf16_units(text: str) -> tuple[int, ...]:
    """The UTF-16 code units a `str` stands for: an astral code point as its pair, an unpaired
    surrogate as itself. Windows counts and compares in these units (`WP12E4-I001`)."""
    data = text.encode("utf-16-le", "surrogatepass")
    return tuple(int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2))


@functools.cache
def _pinned_upcase() -> dict[int, int]:
    """Windows' environment-name uppercase table, captured from `ntdll!RtlUpcaseUnicodeChar` over
    every UTF-16 unit (`windows_upcase.json`, its build recorded there)."""
    document = json.loads(resources.files(__package__).joinpath("windows_upcase.json").read_text())
    return {int(unit, 16): int(upper, 16) for unit, upper in document["map"].items()}


def _live_upcase(unit: int) -> int:
    import ctypes

    ntdll = ctypes.windll.ntdll
    ntdll.RtlUpcaseUnicodeChar.restype = ctypes.c_uint16
    ntdll.RtlUpcaseUnicodeChar.argtypes = [ctypes.c_uint16]
    return int(ntdll.RtlUpcaseUnicodeChar(unit))


def _windows_key(name: str) -> tuple[int, ...]:
    """A name under the native Windows comparison (section 15.3, `WP12E4-CON-R002`,
    `WP12E4-I003`): each UTF-16 unit through the OS uppercase table, so `Q<U+00DF>`/`Qss` and
    `Q<U+0131>`/`QI` stay distinct while `Q<U+00E9>`/`Q<U+00C9>` are one name. On a Windows host
    this is the live OS table; elsewhere (a declared WINDOWS world on another host) it is the pinned
    table, so the comparison never depends on the host."""
    if sys.platform == "win32":
        return tuple(_live_upcase(unit) for unit in utf16_units(name))
    table = _pinned_upcase()  # pragma: no cover -- non-Windows hosts
    return tuple(table.get(unit, unit) for unit in utf16_units(name))  # pragma: no cover


def _windows_names_equal(a: str, b: str) -> bool:
    return _windows_key(a) == _windows_key(b)


class EnvSnapshot(Mapping[str, str]):
    """A read-only snapshot of one provider's environment baseline (section 15.3).

    Entries are exactly the baseline's, with their spelling; no deduplication is imposed. Lookup
    follows the platform: exact on POSIX, the native Windows comparison on WINDOWS (so a lookup of
    `ProgramFiles` finds `PROGRAMFILES`). Entry order is not normative."""

    __slots__ = ("_entries", "_platform")

    def __init__(self, entries: Iterable[tuple[str, str]], platform: Platform) -> None:
        self._entries: tuple[tuple[str, str], ...] = tuple(entries)
        self._platform = platform

    @property
    def platform(self) -> Platform:
        return self._platform

    def entries(self) -> tuple[tuple[str, str], ...]:
        """Every (name, value) pair, as the baseline holds it."""
        return self._entries

    def __getitem__(self, name: str) -> str:
        if self._platform is Platform.POSIX:
            for key, value in self._entries:
                if key == name:
                    return value
        else:
            for key, value in self._entries:
                if key == name or _windows_names_equal(key, name):
                    return value
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, "_platform"):
            raise AttributeError("EnvSnapshot is read-only")
        object.__setattr__(self, name, value)

    def copy(self) -> dict[str, str]:
        """A consumer's own mutable copy (section 15.1): editing it never touches this snapshot
        or the provider."""
        return dict(self._entries)

    def __repr__(self) -> str:
        return f"EnvSnapshot({len(self._entries)} entries, {self._platform.value})"


def native_windows_environment() -> list[tuple[str, str]]:
    """The LIVE native Windows process environment (Owner `WP12E4-C002` = B): original spelling,
    entries absent from `os.environ` (for example set through `os.putenv`) included, current
    values, read now. `=`-prefixed per-drive working-directory records (`=C:=C:\\`) are not
    variables; `os.environ` and Node's `process.env` both omit them, and so does this."""
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetEnvironmentStringsW.restype = ctypes.c_void_p
    kernel32.FreeEnvironmentStringsW.argtypes = [ctypes.c_void_p]
    block = kernel32.GetEnvironmentStringsW()
    entries: list[tuple[str, str]] = []
    try:
        # `WP12E4-I001`: walk the block in UTF-16 UNITS. Each entry is NUL-terminated and the block
        # ends with an empty one; decoding pairs to one code point must never shorten the step.
        units = ctypes.cast(block, ctypes.POINTER(ctypes.c_uint16))
        offset = 0
        while True:
            end = offset
            while units[end] != 0:
                end += 1
            if end == offset:
                break
            raw = bytes(ctypes.string_at(block + offset * 2, (end - offset) * 2))
            text = raw.decode("utf-16-le", "surrogatepass")
            offset = end + 1
            if text.startswith("="):
                continue
            name, _, value = text.partition("=")
            entries.append((name, value))
    finally:
        kernel32.FreeEnvironmentStringsW(block)
    return entries


def local_baseline() -> list[tuple[str, str]]:
    """The local provider's inherited baseline, read now (section 15.4): the live native
    environment on Windows (C002), `os.environ` on POSIX (unchanged)."""
    if sys.platform == "win32":
        return native_windows_environment()
    return list(os.environ.items())  # pragma: no cover -- POSIX hosts only


__all__ = [
    "EnvSnapshot",
    "Platform",
    "host_platform",
    "local_baseline",
    "native_windows_environment",
    "utf16_units",
]
