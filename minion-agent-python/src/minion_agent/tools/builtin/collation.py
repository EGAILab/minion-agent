"""`ls`'s pinned collation (`R006-C`, spec/tools.md `TOOL-028` "Collation"; `TOOL-040`).

PyICU 2.16.2 over ONE pinned ICU 78.3 build: locale `en-001`, TERTIARY strength, NUMERIC off,
CASE_FIRST off, NORMALIZATION_MODE on; keys are the same ICU's root-locale full lowercase. An
implementation MUST fail rather than fall back to any other ICU (a host's own ICU, or a host
language's own lowercase/sort), so loading verifies both the binding and the ICU actually loaded
at runtime -- `icu.ICU_VERSION` is only the version PyICU was compiled against.

Version strings do not identify a BUILD: another ICU that also reports 78.3 would pass them
(`L13-WP131-FR003`). So loading also verifies the build itself (`CE-L13-WP131-03`): EVERY ICU
library instance loaded into this process -- all of them, of any version, not the first match per
name -- must be one the ONE verified build produced and byte-identical (SHA-256) to it, as recorded
in the identity file that scripts/pinned-icu/build.sh writes in the same run that verified the
release tarball and compiled it (`MINION_AGENT_ICU_IDENTITY`). The check runs when collation is
first loaded; an ICU loaded later cannot rebind what PyICU has already bound (disclosed boundary).

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

REQUIRED_ROLES = ("icuuc", "icui18n", "icudata")
"""Every process that sorts must have loaded the verified build's common, i18n (collation) and data
libraries; any OTHER ICU library it has loaded must be the verified build's too (R-F2)."""


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


def _base_name(path: str) -> str:
    """A module's file name, lowercased, whichever separator its path uses."""
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_icu_library(path: str) -> bool:
    """Whether a loaded module is an ICU library, of ANY version or origin (R-F1): `icu*.dll` on
    Windows, `libicu*.so*` elsewhere. PyICU's own extension (`_icu_*.pyd` / `.so`) is not."""
    base = _base_name(path)
    if base.endswith(".dll"):
        return base.startswith("icu")
    return base.startswith("libicu") and ".so" in base


def _enum_process_modules(psapi: Any, process: Any, handles: Any, needed: Any) -> bool:
    return bool(psapi.EnumProcessModules(process, handles, ctypes.sizeof(handles), needed))


def _loaded_windows_modules(capacity: int = 1024) -> list[str]:
    """Every module loaded into this process (`EnumProcessModules`): a second module with the same
    base name loaded from another directory is a separate entry, unlike `GetModuleHandleW`."""
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    process = kernel32.GetCurrentProcess()
    count = capacity
    while True:
        handles = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not _enum_process_modules(psapi, process, handles, ctypes.byref(needed)):
            raise PinnedIcuError("cannot enumerate the process's loaded modules")
        if needed.value <= ctypes.sizeof(handles):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 64
    modules: list[str] = []
    buffer = ctypes.create_unicode_buffer(32768)
    for index in range(needed.value // ctypes.sizeof(wintypes.HMODULE)):
        if kernel32.GetModuleFileNameW(handles[index], buffer, len(buffer)):
            modules.append(buffer.value)
    return modules


def _mapped_linux_files(maps: str) -> list[str]:
    """Every file mapped into this process, from `/proc/self/maps` (each distinct path once)."""
    files: list[str] = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and fields[5] not in files:
            files.append(fields[5])
    return files


def loaded_icu_instances(maps_path: str = "/proc/self/maps") -> list[str]:
    """Every distinct ICU library instance loaded into this process (R-F1) -- no first-match
    lookups, so a foreign same-name copy loaded next to the verified one is listed too."""
    if sys.platform == "win32":
        modules = _loaded_windows_modules()
    else:
        with open(maps_path, encoding="utf-8", errors="replace") as maps:
            modules = _mapped_linux_files(maps.read())
    return [module for module in modules if is_icu_library(module)]


def _parse_identity(text: str) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """scripts/pinned-icu/build.sh's identity file: `source-sha512 <hex>`, `platform <name>`, and
    one `library <role> <file name> <sha256>` line per ICU runtime library the build produced."""
    facts: dict[str, str] = {}
    libraries: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "library" and len(fields) == 4:
            libraries[fields[2].lower()] = (fields[1], fields[3].lower())
        elif len(fields) == 2:
            facts[fields[0]] = fields[1].lower()
    return facts, libraries


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_build_identity(instances: Sequence[str], identity_path: str | None) -> None:
    """Fail closed unless EVERY ICU library instance loaded into this process belongs to the ONE
    verified build (`R006-C`: "fail rather than fall back if it would link or load any other ICU";
    `L13-WP131-FR003`, `CE-L13-WP131-03` R-F2). In order, for each instance: its file name must be
    one the build produced (a library the build did not produce -- another major version, a
    distribution or system ICU -- is rejected even if its bytes equal a listed one), and its bytes
    must equal that library's recorded SHA-256 (byte-identical copies of the same library are the
    same build's bytes and pass). The build's common, i18n and data libraries must be loaded."""
    if not identity_path:
        raise PinnedIcuError(
            f"{ICU_IDENTITY_ENV} is not set: the verified ICU {PINNED_ICU} build's identity is "
            "unknown (scripts/pinned-icu/build.sh --env prints it)"
        )
    try:
        with open(identity_path, encoding="utf-8") as file:
            facts, libraries = _parse_identity(file.read())
    except OSError as exc:
        raise PinnedIcuError(f"the ICU build identity file is unreadable: {exc}") from exc
    if facts.get("source-sha512") != PINNED_SOURCE_SHA512:
        raise PinnedIcuError(
            f"the ICU build identity {identity_path} does not name the pinned ICU {PINNED_ICU} "
            "source tarball"
        )
    roles_loaded: set[str] = set()
    for path in instances:
        listed = libraries.get(_base_name(path))
        if listed is None:
            raise PinnedIcuError(
                f"loaded ICU library {path} is not part of the verified ICU {PINNED_ICU} build"
            )
        role, expected = listed
        actual = _sha256(path)
        if actual != expected:
            raise PinnedIcuError(
                f"loaded ICU library {role} ({path}) is not the verified ICU {PINNED_ICU} build: "
                f"sha256 {actual}, expected {expected}"
            )
        roles_loaded.add(role)
    for role in REQUIRED_ROLES:
        if role not in roles_loaded:
            raise PinnedIcuError(f"ICU library {role} is not loaded from the verified build")


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
    try:
        runtime = _runtime_icu_version()
    except (OSError, AttributeError) as exc:
        # No ICU 78 is loaded (e.g. a stand-in binding) or it lacks u_getVersion_78: fail closed
        # with the typed error rather than a raw loader error (CE-L13-WP131-03 W-F7, Linux).
        raise PinnedIcuError(f"the ICU {PINNED_ICU} runtime is not loadable: {exc}") from exc
    loaded = {
        "PyICU": icu.VERSION,
        "ICU (compiled)": icu.ICU_VERSION,
        "ICU (runtime)": runtime,
    }
    expected = {"PyICU": PINNED_PYICU, "ICU (compiled)": PINNED_ICU, "ICU (runtime)": PINNED_ICU}
    if loaded != expected:
        raise PinnedIcuError(
            f"pinned collation tuple mismatch: expected {expected}, loaded {loaded}"
        )
    verify_build_identity(loaded_icu_instances(), os.environ.get(ICU_IDENTITY_ENV))
    return Collation(icu)
