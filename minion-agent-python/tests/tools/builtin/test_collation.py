"""`ls` collation: the pinned tuple loads and verifies itself, and fails closed otherwise."""

import builtins
import sys
from typing import Any

import pytest

from minion_agent.tools.builtin import collation
from minion_agent.tools.builtin.collation import PinnedIcuError, pinned_collation


def test_pinned_tuple_is_what_is_loaded() -> None:
    assert pinned_collation() is pinned_collation()  # loaded (and verified) once per process
    assert collation._runtime_icu_version() == "78.3"


def test_keys_are_icu_root_lowercase() -> None:
    """ICU's root-locale FULL lowercase (U+0130 -> 'i' + U+0307; final sigma by context)."""
    pinned = pinned_collation()
    assert pinned.key("\u0130") == "i\u0307"
    assert pinned.key("\u03a3\u0391\u03a3") == "\u03c3\u03b1\u03c2"
    assert pinned.key("\u1e9e") == "\u00df"


def test_sort_is_stable_on_equal_keys() -> None:
    pinned = pinned_collation()
    assert pinned.sort(["b", "APPLE", "apple", "Apple", "a"]) == [
        "a",
        "APPLE",
        "apple",
        "Apple",
        "b",
    ]


def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned_collation.cache_clear()
    monkeypatch.setattr(collation, "pinned_collation", pinned_collation)


def test_version_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh(monkeypatch)
    monkeypatch.setattr(collation, "_runtime_icu_version", lambda: "78.1")
    try:
        with pytest.raises(PinnedIcuError, match=r"ICU \(runtime\)': '78.1'"):
            pinned_collation()
    finally:
        pinned_collation.cache_clear()


def test_missing_pyicu_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh(monkeypatch)
    real_import = builtins.__import__

    def refuse_icu(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "icu":
            raise ImportError("no module named icu")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "icu", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse_icu)
    try:
        with pytest.raises(PinnedIcuError, match="is not loadable"):
            pinned_collation()
    finally:
        pinned_collation.cache_clear()
