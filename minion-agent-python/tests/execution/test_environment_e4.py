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
p = k.GetEnvironmentStringsW()
u = ctypes.cast(p, ctypes.POINTER(ctypes.c_uint16))
out, i = [], 0
while True:  # walk in UTF-16 UNITS (WP12E4-I001): a pair must not shorten the step
    j = i
    while u[j]:
        j += 1
    if j == i:
        break
    out.append(ctypes.string_at(p + i * 2, (j - i) * 2).decode("utf-16-le", "surrogatepass"))
    i = j + 1
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


# --- CE-WP12E4-01: UTF-16 units (I001), host-independent native comparison (I003) ------------


@windows_only
async def test_i001_an_entry_after_an_astral_entry_survives_traversal() -> None:
    provider = LocalSubprocess()
    os.putenv("WpE4_AstralProbe", "\U0001f600")
    os.putenv("ZzWpE4_AfterAstral", "must-survive")
    try:
        snapshot = provider.base_env()
        assert snapshot.get("ZzWpE4_AfterAstral") == "must-survive"
        assert snapshot.get("WpE4_AstralProbe") == "\U0001f600"
        inherited = dict(await _child_env(provider, SpawnOptions(inherit_env=True)))
        assert inherited.get("ZzWpE4_AfterAstral") == "must-survive"
    finally:
        os.unsetenv("WpE4_AstralProbe")
        os.unsetenv("ZzWpE4_AfterAstral")


def _assert_native_windows_lookup() -> None:
    """`Q<U+00E9>` finds `Q<U+00C9>`; U+00DF, U+0131 and astral differences stay distinct."""
    world = EnvSnapshot(
        [("PROGRAMFILES", "D:/PF"), ("Q\u00e9", "acute"), ("Q\U0001f600a", "astral")],
        Platform.WINDOWS,
    )
    assert world["ProgramFiles"] == "D:/PF"
    assert world["Q\u00c9"] == "acute"
    assert world.get("Q\U0001f600b") is None and world["Q\U0001f600a"] == "astral"
    assert world.get("Qss") is None


def _no_windows_api(unit: int) -> int:
    raise AssertionError("a non-Windows host must not reach the live Windows uppercase API")


def test_i003_a_windows_world_on_a_non_windows_host_uses_the_pinned_table(monkeypatch: Any) -> None:
    """`CE-WP12E4-01-C001`: runs on EVERY host. The host is non-Windows and the live Windows API
    is unavailable, so the comparison must come from the committed table alone."""
    monkeypatch.setattr(environment_module.sys, "platform", "linux")
    monkeypatch.setattr(environment_module, "_live_upcase", _no_windows_api)
    _assert_native_windows_lookup()


@windows_only
def test_i003_a_windows_world_on_a_windows_host_uses_the_live_table() -> None:
    _assert_native_windows_lookup()


@windows_only
def test_i003_the_pinned_table_is_this_hosts_live_table() -> None:
    """Every UTF-16 unit: the committed table equals `ntdll!RtlUpcaseUnicodeChar` here, and the
    OS's own environment lookup agrees with it on a non-ASCII name."""
    table = environment_module._pinned_upcase()
    assert all(environment_module._live_upcase(u) == table.get(u, u) for u in range(0x10000))
    os.putenv("WpE4_Q\u00e9", "acute")
    try:
        buffer = ctypes.create_unicode_buffer(64)
        assert ctypes.windll.kernel32.GetEnvironmentVariableW("WPE4_Q\u00c9", buffer, 64) > 0
        assert buffer.value == "acute"
    finally:
        os.unsetenv("WpE4_Q\u00e9")


def test_control_an_ascii_only_fallback_fails_the_non_ascii_lookup(monkeypatch: Any) -> None:
    def ascii_only(name: str) -> tuple[int, ...]:
        return tuple(ord(c) - 32 if "a" <= c <= "z" else ord(c) for c in name)

    monkeypatch.setattr(environment_module, "_windows_key", ascii_only)
    with pytest.raises((AssertionError, KeyError)):
        test_i003_a_windows_world_on_a_non_windows_host_uses_the_pinned_table(monkeypatch)


# --- the Owner's remaining C002 section 13 controls, each explicit ------------------------------


@windows_only
def test_control_upper_casing_every_key_fails_witness_a(monkeypatch: Any) -> None:
    real = environment_module.native_windows_environment
    upper = lambda: [(n.upper(), v) for n, v in real()]  # noqa: E731
    monkeypatch.setattr(environment_module, "local_baseline", upper)
    monkeypatch.setattr(subprocess_module, "local_baseline", upper)
    with pytest.raises(AssertionError):
        test_c002_a_b_an_original_case_native_only_variable_is_in_the_snapshot(monkeypatch)


@windows_only
def test_control_omitting_native_only_variables_fails_witness_b(monkeypatch: Any) -> None:
    real = environment_module.native_windows_environment
    only_environ = lambda: [(n, v) for n, v in real() if n.upper() in os.environ]  # noqa: E731
    monkeypatch.setattr(environment_module, "local_baseline", only_environ)
    monkeypatch.setattr(subprocess_module, "local_baseline", only_environ)
    with pytest.raises(AssertionError):
        test_c002_a_b_an_original_case_native_only_variable_is_in_the_snapshot(monkeypatch)


@windows_only
async def test_control_a_snapshot_from_another_source_fails_equivalence(monkeypatch: Any) -> None:
    """The reverse of the inherit-source control: spawn native, `base_env()` from `os.environ`."""
    monkeypatch.setattr(environment_module, "local_baseline", lambda: list(os.environ.items()))
    monkeypatch.setattr(
        LocalSubprocess,
        "base_env",
        lambda self: EnvSnapshot(list(os.environ.items()), self.platform),
    )
    with pytest.raises(AssertionError):
        await test_c002_the_inherit_baseline_and_the_snapshot_agree_a_b_c_d()


@windows_only
async def test_inherit_env_false_is_exactly_the_callers_environment() -> None:
    env = {"SystemRoot": os.environ["SYSTEMROOT"], "WpE4_Only": "1"}
    child = await _child_env(LocalSubprocess(), SpawnOptions(inherit_env=False, env=env))
    assert dict(child) == env


@windows_only
async def test_control_changing_inherit_env_false_fails(monkeypatch: Any) -> None:
    real = subprocess_module._effective_env
    monkeypatch.setattr(subprocess_module, "_effective_env", lambda env, inherit: real(env, True))
    with pytest.raises(AssertionError):
        await test_inherit_env_false_is_exactly_the_callers_environment()


def test_i004_a_local_providers_platform_cannot_be_retagged() -> None:
    """`WP12E4-I004`: the declaration is constant for the provider's lifetime; assigning it is
    refused, and later snapshots keep the original family and its lookup."""
    provider = LocalSubprocess()
    declared = provider.platform
    with pytest.raises(AttributeError):
        provider.platform = (  # type: ignore[misc]
            Platform.POSIX if declared is Platform.WINDOWS else Platform.WINDOWS
        )
    assert provider.platform is declared
    assert provider.base_env().platform is declared
