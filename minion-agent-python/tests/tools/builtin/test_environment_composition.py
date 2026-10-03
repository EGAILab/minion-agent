"""`WP-12.E4` section 15.5: Pi's spawn environment rebuilt from a snapshot -- Node's view (WHATWG
decode with grouping, invalid names dropped, BOM kept) and Windows duplicate arbitration by
ECMAScript `toUpperCase` with the UTF-16-first name winning. Expectations are pinned Node v22.15.1's
executed outputs (minion-agent-docs assurance/layers/data/12-wp12e4/out/)."""

from __future__ import annotations

import os
from typing import Any

import pytest

from minion_agent.execution import EnvSnapshot, Platform
from minion_agent.tools.builtin import environment as composition
from minion_agent.tools.builtin.environment import (
    compose_spawn_environment,
    node_environment_view,
    windows_spawn_environment,
)

FFFD = "\ufffd"


def _posix(*pairs: tuple[bytes, bytes]) -> EnvSnapshot:
    """A POSIX baseline as Python holds it: native bytes behind surrogateescape `str`."""
    return EnvSnapshot(
        [
            (n.decode("utf-8", "surrogateescape"), v.decode("utf-8", "surrogateescape"))
            for n, v in pairs
        ],
        Platform.POSIX,
    )


# node-envbytes-linux.json (characterization section 8)
POSIX_ROWS = [
    (b"a\xffb", "a" + FFFD + "b"),
    (b"a\xe1\x80", "a" + FFFD),
    (b"a\xe1\x80b", "a" + FFFD + "b"),
    (b"a\xf0\x90\x80", "a" + FFFD),
    (b"a\xed\xa0\x80b", "a" + FFFD * 3 + "b"),
    (b"a\xc0\xafb", "a" + FFFD * 2 + "b"),
    (b"\xe2\x82\xac\xe1\x80\xe2\x82\xac", "\u20ac" + FFFD + "\u20ac"),
    (b"\xef\xbb\xbfa", "\ufeffa"),
]


@pytest.mark.parametrize(("raw", "expected"), POSIX_ROWS)
def test_posix_values_decode_as_node_does(raw: bytes, expected: str) -> None:
    assert node_environment_view(_posix((b"V", raw))) == {"V": expected}


def test_posix_invalid_names_are_dropped_and_valid_non_ascii_kept() -> None:
    view = node_environment_view(_posix((b"N_\xff", b"x"), (b"N_\xc3\xa9", b"y")))
    assert view == {"N_\u00e9": "y"}


def test_windows_lone_surrogates_follow_generalized_utf8() -> None:
    """node-envunits-win32.json: a lone unit -> three U+FFFD; a valid pair kept; a
    lone-surrogate NAME dropped."""
    snapshot = EnvSnapshot(
        [("W_LONE", "a\ud800b"), ("W_PAIR", "a\U0001f600b"), ("W_N\udc80", "x")], Platform.WINDOWS
    )
    assert node_environment_view(snapshot) == {
        "W_LONE": "a" + FFFD * 3 + "b",
        "W_PAIR": "a\U0001f600b",
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_windows_arbitration_by_touppercase_utf16_first(reverse: bool) -> None:
    """node-unicode-names-win32.json and node-dedup*.json, both insertion orders."""
    for pairs, expected in [
        ([("Q\u00df", "sharp"), ("Qss", "ss")], {"Qss": "ss"}),
        ([("Q\u0131", "dotless"), ("QI", "ascii")], {"QI": "ascii"}),
        ([("XK", "upper"), ("xk", "lower")], {"XK": "upper"}),
        ([("xK", "m"), ("xk", "l")], {"xK": "m"}),
        ([("Xk", "n"), ("xK", "m")], {"Xk": "n"}),
    ]:
        ordered = list(reversed(pairs)) if reverse else pairs
        assert windows_spawn_environment(dict(ordered)) == expected


def test_compose_removes_exact_spellings_injects_and_arbitrates_on_windows() -> None:
    """Q1/C003: an injected upper-case name wins over an inherited case variant; with nothing
    injected, the variant survives (exact-spelling removal)."""
    snapshot = EnvSnapshot(
        [("Minion_Session_Id", "stale"), ("MINION_MODEL", "old"), ("ProgramFiles", "C:/PF")],
        Platform.WINDOWS,
    )
    names = ["MINION_SESSION_ID", "MINION_MODEL"]
    live = compose_spawn_environment(snapshot, remove=names, inject={"MINION_SESSION_ID": "live"})
    assert live == {"MINION_SESSION_ID": "live", "ProgramFiles": "C:/PF"}
    none = compose_spawn_environment(snapshot, remove=names)
    assert none == {"Minion_Session_Id": "stale", "ProgramFiles": "C:/PF"}


def test_compose_on_posix_has_no_arbitration() -> None:
    snapshot = _posix((b"Path", b"a"), (b"PATH", b"b"))
    assert compose_spawn_environment(snapshot, inject={"X": "y"}) == {
        "Path": "a",
        "PATH": "b",
        "X": "y",
    }


def test_a_fake_windows_world_is_composed_by_windows_rules_whatever_the_host() -> None:
    """Owner F1 section 13: the provider's world decides, not the test runner's host."""
    world = EnvSnapshot(
        [("PROGRAMFILES", "D:/World"), ("Path", "W"), ("PATH", "w2")], Platform.WINDOWS
    )
    assert world["ProgramFiles"] == "D:/World"
    assert compose_spawn_environment(world) == {"PATH": "w2", "PROGRAMFILES": "D:/World"}


# --- negative controls (spec/execution.md section 15.6) ----------------------------------------


def _fails(witness: Any) -> None:
    with pytest.raises(AssertionError):
        witness()


def test_control_per_surrogate_replacement_fails() -> None:
    """Replacing each surrogateescape unit alone gives two U+FFFD for `E1 80`; Node gives one."""

    def per_unit(snapshot: EnvSnapshot) -> dict[str, str]:
        return {
            n: "".join(FFFD if "\udc80" <= c <= "\udcff" else c for c in v)
            for n, v in snapshot.entries()
        }

    _fails(lambda: _assert_view(per_unit, _posix((b"V", b"a\xe1\x80")), {"V": "a" + FFFD}))


def test_control_bom_stripping_fails(monkeypatch: Any) -> None:
    original = node_environment_view

    def strip(snapshot: EnvSnapshot) -> dict[str, str]:
        return {n: v.removeprefix("\ufeff") for n, v in original(snapshot).items()}

    _fails(lambda: _assert_view(strip, _posix((b"V", b"\xef\xbb\xbfa")), {"V": "\ufeffa"}))


def test_control_keeping_an_invalid_name_fails() -> None:
    def keep(snapshot: EnvSnapshot) -> dict[str, str]:
        return {
            n.encode("utf-8", "surrogateescape").decode("utf-8", "replace"): v
            for n, v in snapshot.entries()
        }

    _fails(lambda: _assert_view(keep, _posix((b"N_\xff", b"x")), {}))


@pytest.mark.parametrize(
    "wrong_key", [str.lower, lambda n: n.casefold(), lambda n: n.upper() if n.isascii() else n]
)
def test_control_a_wrong_equivalence_fails_the_non_ascii_pairs(
    monkeypatch: Any, wrong_key: Any
) -> None:
    class Wrong:
        upper_unicode16 = staticmethod(wrong_key)

    monkeypatch.setattr(composition, "pinned_collation", lambda: Wrong())
    _fails(lambda: test_windows_arbitration_by_touppercase_utf16_first(False))


def test_control_last_inserted_arbitration_fails(monkeypatch: Any) -> None:
    def last_wins(env: Any) -> dict[str, str]:
        chosen: dict[str, str] = {}
        for name in env:
            chosen[name.upper()] = name
        return {name: env[name] for name in chosen.values()}

    monkeypatch.setattr(composition, "windows_spawn_environment", last_wins)
    _fails(
        lambda: _assert_arbitration(
            composition.windows_spawn_environment, [("XK", "u"), ("xk", "l")], {"XK": "u"}
        )
    )


def test_control_reading_the_host_environment_fails() -> None:
    def host(snapshot: EnvSnapshot) -> dict[str, str]:
        return dict(os.environ)

    _fails(
        lambda: _assert_view(
            host, EnvSnapshot([("ONLY_IN_WORLD", "v")], Platform.WINDOWS), {"ONLY_IN_WORLD": "v"}
        )
    )


def _assert_view(view: Any, snapshot: EnvSnapshot, expected: dict[str, str]) -> None:
    assert view(snapshot) == expected


def _assert_arbitration(
    arbitrate: Any, pairs: list[tuple[str, str]], expected: dict[str, str]
) -> None:
    assert arbitrate(dict(pairs)) == expected
