"""W3 fixture for the L12-D011 control evidence (CE-L12D011-01 revision 1, record section 7.4).

NOT part of the suite (outside `testpaths`): the driver runs it explicitly, with its own pytest
command and the evidence reporter, on each gate platform. Every node FAILS on purpose; the
driver's real parser and classifier must give each its declared row (`FIXTURE` in
l12_d011_negative_controls.py). It pairs genuine observations with same-rendering counterfeits
(C001: failure origin; C002: exception identity), and the genuine native case is the executable
confirmation of the native NUL origin on the platform where it runs. No file writes; no process
starts (the NUL fails the native call before process creation).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from minion_agent.execution import subprocess as seam

CWD = str(Path(__file__).resolve().parent)


def test_genuine_assertion() -> None:
    assert "a" == "b", "FIXTURE MARK: genuine"


def test_marker_not_at_start() -> None:
    assert "a" == "b", "wrapper: FIXTURE MARK"


def test_wrong_statement() -> None:
    assert "a" == "a", "FIXTURE MARK: declared"
    assert "a" == "b", "FIXTURE MARK: another statement"


def test_fake_assertion_class() -> None:
    fake = type("AssertionError", (Exception,), {"__module__": "builtins"})
    raise fake("FIXTURE MARK: fake class")


def test_assert_named_class() -> None:
    fake = type("assert ['A', 'Z']", (Exception,), {"__module__": "builtins"})
    raise fake("setup helper failed")


def test_assertion_subclass() -> None:
    class Sub(AssertionError):
        pass

    raise Sub("FIXTURE MARK: subclass")


def test_pytest_fail() -> None:
    pytest.fail("FIXTURE MARK: via pytest.fail")


def test_marker_name_error() -> None:
    raise NameError("FIXTURE MARK: counterfeit setup failure")


def test_rewritten_elided() -> None:
    common = "The property 'options.cwd' must be a string, Uint8Array, or URL without null bytes."
    got, want = common + " Received 'd'", common + " Received 'e'"
    assert got == want


def _helper() -> None:
    raise ValueError("embedded null character")


def test_helper_value_error() -> None:
    _helper()


def test_genuine_native_through_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seam, "spawn_rejection", lambda *args, **kwargs: None)  # as the mutants do
    asyncio.run(seam.LocalSubprocess(CWD).spawn([sys.executable], seam.SpawnOptions(cwd="d\0")))


def test_native_without_seam() -> None:
    asyncio.run(asyncio.create_subprocess_exec(sys.executable, "a\0"))
