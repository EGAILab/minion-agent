"""`L13-WP131-FR003`: `ls` collation loads only the ONE verified ICU 78.3 build (`R006-C`).

Version strings cannot tell that build from another ICU that also reports 78.3, so loading also
checks that every ICU library actually mapped into the process is byte-identical to the build
recorded in its identity file. The two subprocess tests are the discriminating negative controls:
a same-version foreign build, and a stand-in binding that reports the pinned versions."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from minion_agent.tools.builtin import collation
from minion_agent.tools.builtin.collation import (
    ICU_BIN_ENV,
    ICU_IDENTITY_ENV,
    PINNED_SOURCE_SHA512,
    PinnedIcuError,
    verify_build_identity,
)


def _identity(path: Path, libraries: dict[str, Path], source: str = PINNED_SOURCE_SHA512) -> str:
    lines = ["# test identity", f"source-sha512 {source}", ""]
    lines += [
        f"{name} {hashlib.sha256(file.read_bytes()).hexdigest()}"
        for name, file in libraries.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def build(tmp_path: Path) -> dict[str, Path]:
    libraries = {}
    for name in collation.ICU_LIBRARIES:
        file = tmp_path / f"{name}.bin"
        file.write_bytes(name.encode() * 100)
        libraries[name] = file
    return libraries


def _loaded(build: dict[str, Path]) -> dict[str, str]:
    return {name: str(file) for name, file in build.items()}


def test_the_verified_build_is_accepted(tmp_path: Path, build: dict[str, Path]) -> None:
    verify_build_identity(_loaded(build), _identity(tmp_path / "id.txt", build))


def test_a_same_version_foreign_build_is_rejected(tmp_path: Path, build: dict[str, Path]) -> None:
    """The discriminating case: same file names, same version, different bytes."""
    identity = _identity(tmp_path / "id.txt", build)
    build["icui18n"].write_bytes(b"another compile of ICU 78.3")
    with pytest.raises(PinnedIcuError, match=r"icui18n .* is not the verified ICU 78\.3 build"):
        verify_build_identity(_loaded(build), identity)


def test_an_unset_identity_fails_closed(build: dict[str, Path]) -> None:
    with pytest.raises(PinnedIcuError, match=f"{ICU_IDENTITY_ENV} is not set"):
        verify_build_identity(_loaded(build), None)


def test_an_unreadable_identity_fails_closed(tmp_path: Path, build: dict[str, Path]) -> None:
    with pytest.raises(PinnedIcuError, match="identity file is unreadable"):
        verify_build_identity(_loaded(build), str(tmp_path / "missing.txt"))


def test_an_identity_from_another_source_fails_closed(
    tmp_path: Path, build: dict[str, Path]
) -> None:
    identity = _identity(tmp_path / "id.txt", build, source="0" * 128)
    with pytest.raises(PinnedIcuError, match=r"does not name the pinned ICU 78\.3 source tarball"):
        verify_build_identity(_loaded(build), identity)


def test_a_library_not_loaded_fails_closed(tmp_path: Path, build: dict[str, Path]) -> None:
    identity = _identity(tmp_path / "id.txt", build)
    loaded = _loaded(build)
    del loaded["icudata"]
    with pytest.raises(PinnedIcuError, match="icudata is not loaded from the verified build"):
        verify_build_identity(loaded, identity)


def test_an_identity_missing_a_library_fails_closed(tmp_path: Path, build: dict[str, Path]) -> None:
    identity = _identity(tmp_path / "id.txt", {k: v for k, v in build.items() if k != "icuuc"})
    with pytest.raises(PinnedIcuError, match="does not list icuuc"):
        verify_build_identity(_loaded(build), identity)


def test_linux_loaded_libraries_come_from_the_process_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r--p 00000000 08:01 1 /opt/icu/lib/libicuuc.so.78.3\n"
        "7f01-7f02 r-xp 00001000 08:01 1 /opt/icu/lib/libicuuc.so.78.3\n"
        "7f02-7f03 r--p 00000000 08:01 2 /opt/icu/lib/libicui18n.so.78.3\n"
        "7f03-7f04 r--p 00000000 08:01 3 /opt/icu/lib/libicudata.so.78.3\n"
        "7f04-7f05 r--p 00000000 08:01 4 /usr/lib/libc.so.6\n"
        "7f05-7f06 rw-p 00000000 00:00 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "platform", "linux")
    assert collation._loaded_icu_libraries(str(maps)) == {
        "icuuc": "/opt/icu/lib/libicuuc.so.78.3",
        "icui18n": "/opt/icu/lib/libicui18n.so.78.3",
        "icudata": "/opt/icu/lib/libicudata.so.78.3",
    }


# ---------------------------------------------------------------------------
# Real negative controls: a fresh process, the real binding, the real loader
# ---------------------------------------------------------------------------

_PINNED_BIN = os.environ.get(ICU_BIN_ENV)
_PINNED_IDENTITY = os.environ.get(ICU_IDENTITY_ENV)
real_windows_build = pytest.mark.skipif(
    sys.platform != "win32" or not _PINNED_BIN or not _PINNED_IDENTITY,
    reason="needs the pinned Windows ICU build (scripts/pinned-icu/build.sh --env)",
)

_LOAD = (
    "from minion_agent.tools.builtin.collation import PinnedIcuError, pinned_collation\n"
    "try:\n"
    "    print('ACCEPTED', pinned_collation().sort(['b', 'a']))\n"
    "except PinnedIcuError as exc:\n"
    "    print('REJECTED', exc)\n"
)


def _load_in_fresh_process(env: dict[str, str], pythonpath: str | None = None) -> str:
    environment = dict(os.environ, **env)
    if pythonpath is not None:
        environment["PYTHONPATH"] = pythonpath + os.pathsep + environment.get("PYTHONPATH", "")
    run = subprocess.run(
        [sys.executable, "-c", _LOAD], capture_output=True, text=True, env=environment, timeout=120
    )
    return run.stdout.strip() or run.stderr.strip()


@real_windows_build
def test_real_verified_build_is_accepted_in_a_fresh_process() -> None:
    assert _load_in_fresh_process({}) == "ACCEPTED ['a', 'b']"


@real_windows_build
def test_real_same_version_foreign_build_is_rejected(tmp_path: Path) -> None:
    """A copy of the pinned ICU DLLs whose i18n library differs by bytes appended after its
    image: it still loads and still reports ICU 78.3, so only the build-identity check can reject
    it -- the case the version-only gate accepted."""
    assert _PINNED_BIN is not None
    foreign = tmp_path / "bin64"
    shutil.copytree(_PINNED_BIN, foreign)
    with open(foreign / "icuin78.dll", "ab") as dll:
        dll.write(b"\0same version, different build")
    outcome = _load_in_fresh_process({ICU_BIN_ENV: str(foreign)})
    assert outcome.startswith("REJECTED loaded ICU library icui18n"), outcome
    assert "is not the verified ICU 78.3 build" in outcome


@real_windows_build
def test_real_stand_in_binding_reporting_pinned_versions_is_rejected(tmp_path: Path) -> None:
    """Codex's stand-in (`L13-WP131-FR003`): a module named `icu` that reports PyICU 2.16.2 over
    ICU 78.3 but is not the verified binding -- the i18n library is never loaded."""
    fake = tmp_path / "icu"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        'VERSION = "2.16.2"\nICU_VERSION = "78.3"\n', encoding="utf-8"
    )
    outcome = _load_in_fresh_process({}, pythonpath=str(tmp_path))
    assert outcome.startswith("REJECTED ICU library icui18n is not loaded"), outcome
