"""`ls`'s pinned collation (`R006-C`, spec/tools.md `TOOL-028` "Collation"; `TOOL-040`).

PyICU 2.16.2 over ONE pinned ICU 78.3 build: locale `en-001`, TERTIARY strength, NUMERIC off,
CASE_FIRST off, NORMALIZATION_MODE on; keys are the same ICU's root-locale full lowercase. An
implementation MUST fail rather than fall back to any other ICU (a host's own ICU, or a host
language's own lowercase/sort), so loading verifies both the binding and the ICU actually loaded
at runtime -- `icu.ICU_VERSION` is only the version PyICU was compiled against.

Version strings do not identify a BUILD: another ICU that also reports 78.3 would pass them
(`L13-WP131-FR003`). So loading also verifies the build itself: every ICU library actually mapped
into this process (common, i18n and data) must be byte-identical (SHA-256) to the ONE verified
build that scripts/pinned-icu/build.sh produced from the SHA-512-checked release tarball, as
recorded in that build's identity file (`MINION_AGENT_ICU_IDENTITY`).

Windows: the pinned ICU's DLL directory is given by `MINION_AGENT_ICU_BIN`
(scripts/pinned-icu/build.sh prints it); the DLL names carry the major version (`icuuc78.dll`),
and the runtime check below pins the minor version too.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

PINNED_PYICU = "2.16.2"
PINNED_ICU = "78.3"
ICU_BIN_ENV = "MINION_AGENT_ICU_BIN"
ICU_IDENTITY_ENV = "MINION_AGENT_ICU_IDENTITY"
PINNED_SOURCE_SHA512 = (
    "04a49455e1489030c520a4bfd2664fa2171e7938d08f2acdbbcb1fda976639fd"
    "8b1f0704f2eec89ba59a7b6d118ceaab6ec5a096e40d9085a0895d91ce225245"
)
"""SHA-512 of `icu4c-78.3-sources.tgz` (the release's `SHASUM512.txt`), which the identity file must
name: the verified build is the one built from exactly this source."""

ICU_LIBRARIES = ("icuuc", "icui18n", "icudata")
"""The ICU libraries whose loaded bytes are checked: common, i18n (collation) and data."""

_WINDOWS_DLLS = {"icuuc": "icuuc78.dll", "icui18n": "icuin78.dll", "icudata": "icudt78.dll"}
_LINUX_SONAMES = {"icuuc": "libicuuc.so", "icui18n": "libicui18n.so", "icudata": "libicudata.so"}


class PinnedIcuError(RuntimeError):
    """The pinned ICU tuple is not what is loaded. `ls` fails rather than sorting any other way."""


def _runtime_icu_version() -> str:
    """`u_getVersion()` from the ICU library actually loaded into this process."""
    name = "icuuc78.dll" if sys.platform == "win32" else "libicuuc.so.78"
    lib = ctypes.CDLL(name)
    version = (ctypes.c_uint8 * 4)()
    lib.u_getVersion_78(version)
    parts = list(version)
    while len(parts) > 2 and parts[-1] == 0:
        parts.pop()
    return ".".join(map(str, parts))


def _loaded_windows_libraries() -> dict[str, str]:
    """The files of the ICU DLLs actually loaded into this process (`GetModuleHandleW` never loads
    anything; a module that is not loaded is simply absent)."""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    loaded: dict[str, str] = {}
    for name, dll in _WINDOWS_DLLS.items():
        handle = kernel32.GetModuleHandleW(dll)
        if handle:
            buffer = ctypes.create_unicode_buffer(32768)
            if kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
                loaded[name] = buffer.value
    return loaded


def _loaded_linux_libraries(maps: str) -> dict[str, str]:
    """The files of the ICU shared objects mapped into this process, from `/proc/self/maps`."""
    loaded: dict[str, str] = {}
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5]
        base = os.path.basename(path)
        for name, soname in _LINUX_SONAMES.items():
            if base.startswith(soname):
                loaded.setdefault(name, path)
    return loaded


def _loaded_icu_libraries(maps_path: str = "/proc/self/maps") -> dict[str, str]:
    if sys.platform == "win32":
        return _loaded_windows_libraries()
    with open(maps_path, encoding="utf-8", errors="replace") as maps:
        return _loaded_linux_libraries(maps.read())


def _parse_identity(text: str) -> dict[str, str]:
    """`<name> <hex>` lines (`#` comments ignored), as scripts/pinned-icu/build.sh writes them."""
    entries: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        entries[name] = value.strip().lower()
    return entries


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_build_identity(loaded: dict[str, str], identity_path: str | None) -> None:
    """Fail closed unless every ICU library actually loaded is the ONE verified build
    (`R006-C`; `L13-WP131-FR003`): the identity file must come from the pinned source, and each of
    common/i18n/data must be loaded and byte-identical to that build. A same-version build from
    anywhere else -- a distribution's ICU, another compile of the same release -- is rejected."""
    if not identity_path:
        raise PinnedIcuError(
            f"{ICU_IDENTITY_ENV} is not set: the verified ICU {PINNED_ICU} build's identity is "
            "unknown (scripts/pinned-icu/build.sh --env prints it)"
        )
    try:
        with open(identity_path, encoding="utf-8") as file:
            identity = _parse_identity(file.read())
    except OSError as exc:
        raise PinnedIcuError(f"the ICU build identity file is unreadable: {exc}") from exc
    if identity.get("source-sha512") != PINNED_SOURCE_SHA512:
        raise PinnedIcuError(
            f"the ICU build identity {identity_path} does not name the pinned ICU {PINNED_ICU} "
            "source tarball"
        )
    for name in ICU_LIBRARIES:
        path = loaded.get(name)
        if path is None:
            raise PinnedIcuError(f"ICU library {name} is not loaded from the verified build")
        expected = identity.get(name)
        if expected is None:
            raise PinnedIcuError(f"the ICU build identity {identity_path} does not list {name}")
        actual = _sha256(path)
        if actual != expected:
            raise PinnedIcuError(
                f"loaded ICU library {name} ({path}) is not the verified ICU {PINNED_ICU} build: "
                f"sha256 {actual}, expected {expected}"
            )


class Collation:
    """The pinned comparator: `compare(key(a), key(b))`, `key` = ICU root-locale lowercase."""

    def __init__(self, icu: Any) -> None:
        attribute, value = icu.UCollAttribute, icu.UCollAttributeValue
        collator = icu.Collator.createInstance(icu.Locale("en-001"))
        collator.setAttribute(attribute.STRENGTH, value.TERTIARY)
        collator.setAttribute(attribute.NUMERIC_COLLATION, value.OFF)
        collator.setAttribute(attribute.CASE_FIRST, value.OFF)
        collator.setAttribute(attribute.NORMALIZATION_MODE, value.ON)
        self._collator = collator
        self._root = icu.Locale.getRoot()
        self._unicode_string: Callable[[str], Any] = icu.UnicodeString

    def key(self, name: str) -> str:
        return str(self._unicode_string(name).toLower(self._root))

    def compare(self, a: str, b: str) -> int:
        return int(self._collator.compare(a, b))

    def sort(self, names: Sequence[str]) -> list[str]:
        """Stable sort by the pinned comparator on the lowercase keys; ties keep input order."""
        keyed = [(self.key(name), name) for name in names]

        def by_key(a: tuple[str, str], b: tuple[str, str]) -> int:
            return self.compare(a[0], b[0])

        keyed.sort(key=functools.cmp_to_key(by_key))
        return [name for _, name in keyed]


@functools.cache
def pinned_collation() -> Collation:
    """Load and verify the pinned tuple once per process. Raises `PinnedIcuError` otherwise."""
    icu_bin = os.environ.get(ICU_BIN_ENV)
    if sys.platform == "win32" and icu_bin:
        os.add_dll_directory(icu_bin)
    try:
        import icu
    except ImportError as exc:
        raise PinnedIcuError(
            f"PyICU {PINNED_PYICU} over ICU {PINNED_ICU} is not loadable: {exc}"
        ) from exc
    loaded = {
        "PyICU": icu.VERSION,
        "ICU (compiled)": icu.ICU_VERSION,
        "ICU (runtime)": _runtime_icu_version(),
    }
    expected = {"PyICU": PINNED_PYICU, "ICU (compiled)": PINNED_ICU, "ICU (runtime)": PINNED_ICU}
    if loaded != expected:
        raise PinnedIcuError(
            f"pinned collation tuple mismatch: expected {expected}, loaded {loaded}"
        )
    verify_build_identity(_loaded_icu_libraries(), os.environ.get(ICU_IDENTITY_ENV))
    return Collation(icu)
