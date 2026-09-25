"""`ls`'s pinned collation (`R006-C`, spec/tools.md `TOOL-028` "Collation"; `TOOL-040`).

PyICU 2.16.2 over ONE pinned ICU 78.3 build: locale `en-001`, TERTIARY strength, NUMERIC off,
CASE_FIRST off, NORMALIZATION_MODE on; keys are the same ICU's root-locale full lowercase. An
implementation MUST fail rather than fall back to any other ICU (a host's own ICU, or a host
language's own lowercase/sort), so loading verifies both the binding and the ICU actually loaded
at runtime -- `icu.ICU_VERSION` is only the version PyICU was compiled against.

Windows: the pinned ICU's DLL directory is given by `MINION_AGENT_ICU_BIN`
(scripts/pinned-icu/build.sh prints it); the DLL names carry the major version (`icuuc78.dll`),
and the runtime check below pins the minor version too.
"""

from __future__ import annotations

import ctypes
import functools
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

PINNED_PYICU = "2.16.2"
PINNED_ICU = "78.3"
ICU_BIN_ENV = "MINION_AGENT_ICU_BIN"


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
    return Collation(icu)
