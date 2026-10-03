"""`WP-12.E4` (`EXEC-010`, spec/execution.md section 15.6): platform, the read-only isolated
`base_env()` snapshot, baseline equivalence with `inherit_env=True`, and the Python Windows
amendment `WP12E4-C002` = B (witnesses A-D and the Owner's negative controls)."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from typing import Any

import pytest

from minion_agent.execution import EnvSnapshot, LocalSubprocess, Ok, Platform, SpawnOptions
from minion_agent.execution import environment as environment_module
from minion_agent.execution import subprocess as subprocess_module

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the native Windows baseline")

# A child that reports its NATIVE environment block (not `os.environ`, which upper-cases names).
_CHILD = r"""
import ctypes, json
k = ctypes.windll.kernel32
k.GetEnvironmentStringsW.restype = ctypes.c_void_p
p = k.GetEnvironmentStringsW(); out, i = [], 0
while True:
    s = ctypes.wstring_at(p + i * 2)
    if not s: break
    out.append(s); i += len(s) + 1
print(json.dumps(sorted(e.partition("=")[::2] for e in out if not e.startswith("="))))
"""


async def _child_env(provider: LocalSubprocess, options: SpawnOptions) -> list[tuple[str, str]]:
    spawned = await provider.spawn([sys.executable, "-c", _CHILD], options)
    assert isinstance(spawned, Ok)
    process = spawned.value
    chunks = []
    assert process.stdout is not None
    while True:
        read = await process.stdout.read_chunk()
        assert isinstance(read, Ok)
        if read.value is None:
            break
        chunks.append(read.value)
    await process.wait()
    return [tuple(pair) for pair in json.loads(b"".join(chunks))]


def test_a_local_provider_declares_its_hosts_platform() -> None:
    expected = Platform.WINDOWS if os.name == "nt" else Platform.POSIX
    assert LocalSubprocess().platform is expected


def test_the_snapshot_is_read_only_and_a_copy_is_the_consumers_own() -> None:
    snapshot = EnvSnapshot([("K", "old")], Platform.POSIX)
    with pytest.raises(TypeError):
        snapshot["K"] = "new"  # type: ignore[index]
    with pytest.raises(AttributeError):
        snapshot._entries = ()  # type: ignore[misc]
    mine = snapshot.copy()
    mine["K"] = "new"
    assert snapshot["K"] == "old" and mine == {"K": "new"}


def test_posix_lookup_is_exact_and_windows_lookup_is_native() -> None:
    posix = EnvSnapshot([("Path", "p")], Platform.POSIX)
    assert posix.get("PATH") is None and posix["Path"] == "p"
    windows = EnvSnapshot(
        [("PROGRAMFILES", "C:/PF"), ("Qss", "ss"), ("Q\u00df", "sharp")], Platform.WINDOWS
    )
    assert windows["ProgramFiles"] == "C:/PF"
    assert (
        windows["QSS"] == "ss" and windows["Q\u00df"] == "sharp"
    )  # distinct native names stay distinct
    assert len(windows) == 3 and list(windows) == ["PROGRAMFILES", "Qss", "Q\u00df"]
    assert "MISSING" not in windows and repr(windows) == "EnvSnapshot(3 entries, windows)"


@windows_only
def test_c002_a_b_an_original_case_native_only_variable_is_in_the_snapshot(
    monkeypatch: Any,
) -> None:
    os.putenv("WpE4_Mixed_Case", "native-only")  # native, NOT reflected in os.environ
    try:
        assert "WPE4_MIXED_CASE" not in os.environ
        assert ("WpE4_Mixed_Case", "native-only") in LocalSubprocess().base_env().entries()
    finally:
        os.unsetenv("WpE4_Mixed_Case")


@windows_only
async def test_c002_the_inherit_baseline_and_the_snapshot_agree_a_b_c_d() -> None:
    provider = LocalSubprocess()
    os.putenv("WpE4_Native", "b")  # B: native-only, original case (A)
    os.environ["WPE4_VIA_ENVIRON"] = "c"  # C: an os.environ update
    os.environ["WPE4_REMOVED"] = "gone"
    del os.environ["WPE4_REMOVED"]  # D: removed natively
    try:
        snapshot = sorted(provider.base_env().entries())
        inherited = await _child_env(provider, SpawnOptions(inherit_env=True))
        rebuilt = await _child_env(
            provider, SpawnOptions(inherit_env=False, env=provider.base_env().copy())
        )
        for observed in (snapshot, inherited):
            assert ("WpE4_Native", "b") in observed
            assert ("WPE4_VIA_ENVIRON", "c") in observed
            assert all(name != "WPE4_REMOVED" for name, _ in observed)
        # base_env() + inherit_env=False reconstructs the inherit_env=True child exactly.
        assert rebuilt == inherited
    finally:
        os.unsetenv("WpE4_Native")
        os.environ.pop("WPE4_VIA_ENVIRON", None)


@windows_only
def test_a_snapshot_is_isolated_from_a_later_baseline_change() -> None:
    provider = LocalSubprocess()
    before = provider.base_env()
    os.putenv("WpE4_Later", "x")
    try:
        assert "WpE4_Later" not in before and provider.base_env().get("WpE4_Later") == "x"
    finally:
        os.unsetenv("WpE4_Later")


@windows_only
def test_per_drive_records_are_not_variables() -> None:
    kernel32 = ctypes.windll.kernel32
    assert kernel32.SetEnvironmentVariableW("=Z:", "Z:\\")
    try:
        assert all(not name.startswith("=") for name in LocalSubprocess().base_env())
    finally:
        kernel32.SetEnvironmentVariableW("=Z:", None)


# --- negative controls (Owner C002 section 13) -------------------------------------------------


@windows_only
def test_control_dict_os_environ_as_the_baseline_fails_witness_a_b(monkeypatch: Any) -> None:
    monkeypatch.setattr(environment_module, "local_baseline", lambda: list(os.environ.items()))
    monkeypatch.setattr(subprocess_module, "local_baseline", lambda: list(os.environ.items()))
    with pytest.raises(AssertionError):
        test_c002_a_b_an_original_case_native_only_variable_is_in_the_snapshot(monkeypatch)


@windows_only
async def test_control_an_inherit_baseline_from_another_source_fails_equivalence(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(subprocess_module, "local_baseline", lambda: list(os.environ.items()))
    with pytest.raises(AssertionError):
        await test_c002_the_inherit_baseline_and_the_snapshot_agree_a_b_c_d()


@windows_only
def test_control_a_construction_time_cache_fails_isolation(monkeypatch: Any) -> None:
    cached = LocalSubprocess().base_env()
    monkeypatch.setattr(LocalSubprocess, "base_env", lambda self: cached)
    with pytest.raises(AssertionError):
        test_a_snapshot_is_isolated_from_a_later_baseline_change()


def test_a_fake_windows_world_on_a_posix_host_folds_ascii_names(monkeypatch: Any) -> None:
    """With no OS to ask (a POSIX host), Windows lookup folds ASCII only -- which agrees with the
    native comparison on every name the consumers look up, and keeps non-ASCII names distinct."""
    monkeypatch.setattr(environment_module.sys, "platform", "linux")
    world = EnvSnapshot([("PROGRAMFILES", "D:/PF"), ("Q\u00df", "sharp")], Platform.WINDOWS)
    assert world["ProgramFiles"] == "D:/PF"
    assert world.get("QSS") is None and world["Q\u00df"] == "sharp"
