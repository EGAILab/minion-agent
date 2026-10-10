"""L12D007-C001: the canonical runner's fixture steps never reach outside their case directory.

Permanent rejecting controls for `fs_path_runner._fixture_target`. Links are created INSIDE the
test's own directory (some pointing outside) and only the guard is asked about them; nothing is ever
written, renamed or deleted through a link, and the outside targets are never created."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.conformance.fs_path_runner import _fixture_target

OUTSIDE = (
    "C:\\l12d007-runner-never-created"
    if sys.platform == "win32"
    else "/l12d007-runner-never-created"
)


@pytest.mark.parametrize(
    "raw", ["", "..", "a/../b", "a\\..\\b", "C:x", "E:", "E:\\", "/abs", "\\abs", "\\\\srv\\share"]
)
def test_prohibited_raw_forms_are_refused_before_normalization(tmp_path: Path, raw: str) -> None:
    with pytest.raises(AssertionError, match="prohibited raw form"):
        _fixture_target(str(tmp_path), raw)


@pytest.mark.parametrize("name", ["f", "d/c", "./f:stream:bad", "x<y", "n" * 300])
def test_ordinary_fixture_names_are_accepted(tmp_path: Path, name: str) -> None:
    assert _fixture_target(str(tmp_path), name).startswith(str(tmp_path))


def test_a_dangling_link_pointing_outside_is_refused(tmp_path: Path) -> None:
    os.symlink(OUTSIDE, tmp_path / "out")
    with pytest.raises(AssertionError, match="through a link"):
        _fixture_target(str(tmp_path), "out")


def test_a_two_hop_escape_is_refused(tmp_path: Path) -> None:
    os.symlink("mid", tmp_path / "hop")
    os.symlink(OUTSIDE, tmp_path / "mid")
    with pytest.raises(AssertionError, match="through a link"):
        _fixture_target(str(tmp_path), "hop")


def test_a_directory_link_to_the_filesystem_root_is_refused(tmp_path: Path) -> None:
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction("C:\\", str(tmp_path / "j"))
    else:
        os.symlink("/", tmp_path / "j", target_is_directory=True)
    with pytest.raises(AssertionError, match="through a link"):
        _fixture_target(str(tmp_path), "j/x")


def test_a_loop_that_stays_inside_is_accepted(tmp_path: Path) -> None:
    os.symlink("b", tmp_path / "a")
    os.symlink("a", tmp_path / "b")
    assert _fixture_target(str(tmp_path), "a") == str(tmp_path / "a")


def test_a_symlink_text_is_checked_from_the_link_directory(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    assert _fixture_target(str(tmp_path), "x", str(tmp_path / "d")) == str(tmp_path / "d" / "x")
    with pytest.raises(AssertionError):
        _fixture_target(str(tmp_path), "..", str(tmp_path / "d"))
