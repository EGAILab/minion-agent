"""`L13-WP131-FR003` / `CE-L13-WP131-03`: `ls` collation loads only the ONE verified ICU build.

R006-C: "an implementation MUST fail rather than fall back if it would link or load any other ICU".
Version strings cannot tell the verified build from another ICU that also reports 78.3, so at load
EVERY ICU library instance in the process must be a file the verified build produced, byte for
byte (R-F1, R-F2), and the identity naming those files is written only by the build run that
produced them (R-F4). Witnesses W-F1..W-F7 of the agreed checkpoint; the real ones run in a fresh
process with the real binding and loader."""

from __future__ import annotations

import hashlib
import os
import re
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
    is_icu_library,
    loaded_icu_instances,
    verify_build_identity,
)

BUILD_SH = Path(__file__).resolve().parents[3] / "scripts" / "pinned-icu" / "build.sh"


def _bash() -> str | None:
    """The bash that runs build.sh: Git's on Windows (System32's `bash.exe` is WSL, which cannot
    see Windows paths), otherwise the PATH's."""
    if sys.platform != "win32":
        return shutil.which("bash")
    found = shutil.which("bash")
    if found is not None and "system32" not in found.lower():
        return found
    git = shutil.which("git")
    for parent in Path(git).resolve().parents if git else ():
        if (parent / "bin" / "bash.exe").exists():
            return str(parent / "bin" / "bash.exe")
    return None


# ---------------------------------------------------------------------------
# Unit witnesses over a synthetic build
# ---------------------------------------------------------------------------


@pytest.fixture
def build(tmp_path: Path) -> dict[str, Path]:
    """A synthetic 'verified build': three libraries in one directory, keyed by role."""
    directory = tmp_path / "bin"
    directory.mkdir()
    files = {"icuuc": "icuuc78.dll", "icui18n": "icuin78.dll", "icudata": "icudt78.dll"}
    built = {}
    for role, name in files.items():
        file = directory / name
        file.write_bytes(role.encode() * 100)
        built[role] = file
    return built


def _identity(path: Path, build: dict[str, Path], source: str = PINNED_SOURCE_SHA512) -> str:
    lines = ["# test identity", f"source-sha512 {source}", "platform windows", ""]
    lines += [
        f"library {role} {file.name} {hashlib.sha256(file.read_bytes()).hexdigest()}"
        for role, file in build.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _copy(file: Path, directory: Path, name: str | None = None, extra: bytes = b"") -> str:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (name or file.name)
    target.write_bytes(file.read_bytes() + extra)
    return str(target)


def test_the_verified_build_alone_is_accepted(tmp_path: Path, build: dict[str, Path]) -> None:
    verify_build_identity([str(f) for f in build.values()], _identity(tmp_path / "id", build))


def test_w_f1_a_second_byte_different_instance_is_rejected(
    tmp_path: Path, build: dict[str, Path]
) -> None:
    """`W-F1` (FR003-a): the verified i18n library AND a same-name, same-version foreign copy."""
    identity = _identity(tmp_path / "id", build)
    foreign = _copy(build["icui18n"], tmp_path / "foreign", extra=b"another 78.3 build")
    instances = [str(f) for f in build.values()] + [foreign]
    with pytest.raises(PinnedIcuError, match=re.escape(f"icui18n ({foreign}) is not the verified")):
        verify_build_identity(instances, identity)


def test_w_f2_a_byte_identical_copy_is_the_same_build(
    tmp_path: Path, build: dict[str, Path]
) -> None:
    """`W-F2`: two instances of the same library with identical bytes are one build's bytes."""
    identity = _identity(tmp_path / "id", build)
    twin = _copy(build["icui18n"], tmp_path / "twin")
    verify_build_identity([str(f) for f in build.values()] + [twin], identity)


def test_w_f3_an_unlisted_icu_library_is_rejected_even_with_listed_bytes(
    tmp_path: Path, build: dict[str, Path]
) -> None:
    """`W-F3`: an ICU library the build did not produce -- here another major version's name
    carrying the verified common library's exact bytes. The name check comes first."""
    identity = _identity(tmp_path / "id", build)
    other = _copy(build["icuuc"], tmp_path / "other", name="icuuc77.dll")
    with pytest.raises(PinnedIcuError, match=r"is not part of the verified ICU 78\.3 build"):
        verify_build_identity([str(f) for f in build.values()] + [other], identity)


def test_w_f4_a_required_library_not_loaded_is_rejected(
    tmp_path: Path, build: dict[str, Path]
) -> None:
    identity = _identity(tmp_path / "id", build)
    with pytest.raises(PinnedIcuError, match="icudata is not loaded from the verified build"):
        verify_build_identity([str(build["icuuc"]), str(build["icui18n"])], identity)


def test_an_unset_identity_fails_closed(build: dict[str, Path]) -> None:
    with pytest.raises(PinnedIcuError, match=f"{ICU_IDENTITY_ENV} is not set"):
        verify_build_identity([str(f) for f in build.values()], None)


def test_an_unreadable_identity_fails_closed(tmp_path: Path, build: dict[str, Path]) -> None:
    with pytest.raises(PinnedIcuError, match="identity file is unreadable"):
        verify_build_identity([str(f) for f in build.values()], str(tmp_path / "missing"))


def test_an_identity_from_another_source_fails_closed(
    tmp_path: Path, build: dict[str, Path]
) -> None:
    identity = _identity(tmp_path / "id", build, source="0" * 128)
    with pytest.raises(PinnedIcuError, match=r"does not name the pinned ICU 78\.3 source tarball"):
        verify_build_identity([str(f) for f in build.values()], identity)


@pytest.mark.parametrize(
    ("path", "icu"),
    [
        (r"C:\icu\bin64\icuuc78.dll", True),
        (r"C:\Windows\System32\ICU.DLL", True),
        (r"C:\venv\Lib\site-packages\icu\_icu_.cp313-win_amd64.pyd", False),
        ("/opt/icu/lib/libicui18n.so.78.3", True),
        ("/usr/lib/x86_64-linux-gnu/libicuuc.so.72.1", True),
        ("/venv/site-packages/icu/_icu_.cpython-312-x86_64-linux-gnu.so", False),
        ("/usr/lib/libc.so.6", False),
    ],
)
def test_every_icu_library_of_any_version_counts(path: str, icu: bool) -> None:
    """R-F1: any ICU library, of any version or origin, is an instance to check."""
    assert is_icu_library(path) is icu


def test_linux_inventory_lists_every_mapping_not_the_first_per_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`W-F1`'s inventory half on Linux (Codex's witness): two same-soname mappings are both
    instances; non-ICU and anonymous mappings are not."""
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r--p 00000000 08:01 1 /verified/libicuuc.so.78.3\n"
        "7f01-7f02 r-xp 00001000 08:01 1 /verified/libicuuc.so.78.3\n"
        "7f02-7f03 r--p 00000000 08:01 2 /verified/libicui18n.so.78.3\n"
        "7f03-7f04 r--p 00000000 08:01 3 /verified/libicudata.so.78.3\n"
        "7f04-7f05 r--p 00000000 08:01 5 /foreign/libicui18n.so.78.3\n"
        "7f05-7f06 r--p 00000000 08:01 4 /usr/lib/libc.so.6\n"
        "7f06-7f07 rw-p 00000000 00:00 0 [heap]\n"
        "7f07-7f08 rw-p 00000000 00:00 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "platform", "linux")
    assert loaded_icu_instances(str(maps)) == [
        "/verified/libicuuc.so.78.3",
        "/verified/libicui18n.so.78.3",
        "/verified/libicudata.so.78.3",
        "/foreign/libicui18n.so.78.3",
    ]


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows module enumeration")


@windows_only
def test_windows_inventory_grows_past_a_small_buffer() -> None:
    """Every loaded module is enumerated even when the first buffer is too small."""
    modules = collation._loaded_windows_modules(capacity=1)
    assert len(modules) > 1
    assert any(
        os.path.basename(m).lower() == "python3.dll" or m.lower().endswith(".exe") for m in modules
    )


@windows_only
@pytest.mark.parametrize(("result", "why"), [("zero", "failed"), ("full", "truncated")])
def test_windows_module_path_lookup_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch, result: str, why: str
) -> None:
    """FR003 targeted review (R-F1): a loaded module whose path cannot be read -- lookup failure
    (0) or truncation (the buffer's size) -- makes the inventory incomplete, so loading fails
    closed instead of silently leaving that module unverified."""
    real = collation._module_file_name
    calls = []

    def lookup(kernel32: object, handle: object, buffer: object) -> int:
        calls.append(handle)
        if len(calls) == 2:  # one enumerated module among the others
            return 0 if result == "zero" else len(buffer)  # type: ignore[arg-type]
        return real(kernel32, handle, buffer)

    monkeypatch.setattr(collation, "_module_file_name", lookup)
    with pytest.raises(PinnedIcuError, match=f"GetModuleFileNameW {why}"):
        collation._loaded_windows_modules()


def test_an_unreadable_listed_instance_fails_closed(tmp_path: Path, build: dict[str, Path]) -> None:
    """An instance that is listed but cannot be read (e.g. a deleted mapping) is not verified."""
    identity = _identity(tmp_path / "id", build)
    gone = str(tmp_path / "gone" / "icuin78.dll")
    with pytest.raises(PinnedIcuError, match="cannot be read to verify"):
        verify_build_identity([str(f) for f in build.values()] + [gone], identity)


@windows_only
def test_windows_inventory_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collation, "_enum_process_modules", lambda *args: False)
    with pytest.raises(PinnedIcuError, match="cannot enumerate the process's loaded modules"):
        collation._loaded_windows_modules()


# ---------------------------------------------------------------------------
# W-F5 -- the trust root is the build run itself (FR003-b)
# ---------------------------------------------------------------------------


def test_w_f5_build_sh_has_no_re_attestation_mode(tmp_path: Path) -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("no bash to run scripts/pinned-icu/build.sh")
    run = subprocess.run(
        [bash, str(BUILD_SH), str(tmp_path), "--identity"], capture_output=True, text=True
    )
    assert run.returncode == 2
    assert "unsupported mode: --identity" in run.stderr
    assert not (tmp_path / "pinned-icu-identity.txt").exists()


def test_w_f5_identity_is_written_only_after_a_clean_verified_build() -> None:
    """The ONE place that writes the identity is the last step of a build run, after the tarball
    check, a clean slate (old binaries, install tree and identity removed), the fresh extraction,
    the compile and the install -- so it can only hash what that run produced."""
    lines = BUILD_SH.read_text(encoding="utf-8").splitlines()
    calls = [i for i, line in enumerate(lines) if line.strip() == "write_identity"]
    assert len(calls) == 1
    call = calls[0]
    writes = [i for i, line in enumerate(lines) if "> pinned-icu-identity.txt" in line]
    assert len(writes) == 1  # inside write_identity's definition only

    def before(pattern: str) -> bool:
        return any(re.search(pattern, line) for line in lines[:call])

    assert before(r"sha512sum -c -$")
    assert before(r"rm -rf icu install pinned-icu-identity\.txt && tar xzf")
    assert before(r"MSBuild") and before(r"make install")
    modes = [line for line in lines if re.match(r"\s*build\|--env\)", line)]
    assert modes == ["  build|--env) ;;"]


# ---------------------------------------------------------------------------
# Real negative controls: a fresh process, the real binding, the real loader (Windows)
# ---------------------------------------------------------------------------

_PINNED_BIN = os.environ.get(ICU_BIN_ENV)
_PINNED_IDENTITY = os.environ.get(ICU_IDENTITY_ENV)
real_windows_build = pytest.mark.skipif(
    sys.platform != "win32" or not _PINNED_BIN or not _PINNED_IDENTITY,
    reason="needs the pinned Windows ICU build (scripts/pinned-icu/build.sh --env)",
)

_LOAD = (
    "import ctypes, os, sys\n"
    "from minion_agent.tools.builtin import collation\n"
    "if os.environ.get('MINION_AGENT_ICU_BIN'):\n"
    "    os.add_dll_directory(os.environ['MINION_AGENT_ICU_BIN'])\n"
    "try:\n"
    "    import icu\n"  # the real binding loads its ICU first (a stand-in may shadow it)
    "except ImportError:\n"
    "    pass\n"
    "for path in sys.argv[1:]:\n"
    "    ctypes.WinDLL(path)\n"  # then extra modules, loaded by full path, before the check
    "try:\n"
    "    print('ACCEPTED', collation.pinned_collation().sort(['b', 'a']))\n"
    "except collation.PinnedIcuError as exc:\n"
    "    print('REJECTED', exc)\n"
)


def _load_in_fresh_process(
    *extra_modules: str, env: dict[str, str] | None = None, pythonpath: str | None = None
) -> str:
    environment = dict(os.environ, **(env or {}))
    if pythonpath is not None:
        environment["PYTHONPATH"] = pythonpath + os.pathsep + environment.get("PYTHONPATH", "")
    run = subprocess.run(
        [sys.executable, "-c", _LOAD, *extra_modules],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    return run.stdout.strip() or run.stderr.strip()


def _pinned(name: str) -> Path:
    assert _PINNED_BIN is not None
    return Path(_PINNED_BIN) / name


@real_windows_build
def test_real_verified_build_is_accepted() -> None:
    assert _load_in_fresh_process() == "ACCEPTED ['a', 'b']"


@real_windows_build
def test_real_w_f1_foreign_same_name_instance_next_to_the_verified_one_is_rejected(
    tmp_path: Path,
) -> None:
    """`W-F1` (FR003-a, measured on this host): a byte-different `icuin78.dll` loaded by full
    path coexists with the verified one; the complete inventory sees and rejects it."""
    foreign = _copy(_pinned("icuin78.dll"), tmp_path / "foreign", extra=b"\0another 78.3 build")
    outcome = _load_in_fresh_process(foreign)
    assert outcome.startswith("REJECTED loaded ICU library icui18n"), outcome
    assert foreign in outcome


@real_windows_build
def test_real_w_f2_byte_identical_copy_is_accepted(tmp_path: Path) -> None:
    twin = _copy(_pinned("icuin78.dll"), tmp_path / "twin")
    assert _load_in_fresh_process(twin) == "ACCEPTED ['a', 'b']"


@real_windows_build
def test_real_w_f3_unlisted_icu_library_is_rejected(tmp_path: Path) -> None:
    """`W-F3`: another ICU library -- the verified common library's bytes under another major
    version's name -- loaded into the process."""
    other = _copy(_pinned("icuuc78.dll"), tmp_path / "other", name="icuuc77.dll")
    outcome = _load_in_fresh_process(other)
    expected = f"REJECTED loaded ICU library {other} is not part of the verified ICU 78.3 build"
    assert outcome == expected


@real_windows_build
def test_real_w_f6_same_version_foreign_build_is_rejected(tmp_path: Path) -> None:
    """`W-F6`: the whole ICU directory is a same-version foreign build (a copy whose i18n
    library differs by bytes appended after its image): it loads and reports 78.3."""
    assert _PINNED_BIN is not None
    foreign = tmp_path / "bin64"
    shutil.copytree(_PINNED_BIN, foreign)
    with open(foreign / "icuin78.dll", "ab") as dll:
        dll.write(b"\0same version, different build")
    outcome = _load_in_fresh_process(env={ICU_BIN_ENV: str(foreign)})
    assert outcome.startswith("REJECTED loaded ICU library icui18n"), outcome
    assert "is not the verified ICU 78.3 build" in outcome


@real_windows_build
def test_real_w_f7_stand_in_binding_reporting_pinned_versions_is_rejected(tmp_path: Path) -> None:
    """`W-F7` (Codex's stand-in): a module named `icu` that reports PyICU 2.16.2 over ICU 78.3
    but is not the verified binding -- the i18n library is never loaded."""
    fake = tmp_path / "icu"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        'VERSION = "2.16.2"\nICU_VERSION = "78.3"\n', encoding="utf-8"
    )
    outcome = _load_in_fresh_process(pythonpath=str(tmp_path))
    assert outcome == "REJECTED ICU library icui18n is not loaded from the verified build", outcome


def test_required_roles_are_common_i18n_and_data() -> None:
    assert collation.REQUIRED_ROLES == ("icuuc", "icui18n", "icudata")
