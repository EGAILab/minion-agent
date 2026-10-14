"""L12-D011 binding witnesses (spec/execution.md section 21.7) beside the canonical spawn-nul cases:
the pure diagnostic/order functions on BOTH platforms, no native spawn after a rejection, an
unrelated ValueError unchanged, the tools' narrow distinction, stdin transport, and the section
15.5 consumer arbitration feeding bash."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import Err, LocalFileSystem
from minion_agent.execution import subprocess as subprocess_module
from minion_agent.execution.environment import EnvSnapshot, Platform
from minion_agent.execution.errors import SubprocessError, SubprocessErrorCode
from minion_agent.execution.spawn_nul import (
    argv_rejection,
    diagnostic,
    env_check_order,
    from_units,
    inspect_string,
    spawn_rejection,
    units,
)
from minion_agent.execution.subprocess import LocalSubprocess, SpawnOptions
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin import _search, bash_shell
from minion_agent.tools.builtin.bash import create_bash_tool
from minion_agent.tools.builtin.bash_shell import ShellConfig
from minion_agent.tools.builtin.paths import BuiltinToolError
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

N = "\x00"
FULLWIDTH = chr(0xFF5E)  # UTF-16 unit 0xFF5E sorts AFTER an astral's 0xD800 lead, unlike in UTF-8
ASTRAL = chr(0x10000)


def ins(value: str) -> str:
    return from_units(inspect_string(value))


# ---- the pure functions --------------------------------------------------------------------------
def test_units_round_trip_keeps_pairs_and_lone_surrogates() -> None:
    assert units("\U0001f600") == [0xD83D, 0xDE00]
    assert from_units([0xD83D, 0xDE00]) == "\U0001f600"
    assert from_units([0xD800]) == "\ud800"


def test_inspect_quotes_escapes_and_trailer() -> None:
    assert ins("a") == "'a'"
    assert ins("it's") == '"it\'s"'
    assert ins('it\'s "q"') == '`it\'s "q"`'
    assert ins('it\'s "q" `b`') == "'it\\'s \"q\" `b`'"
    assert ins('it\'s "q" ${x}') == "'it\\'s \"q\" ${x}'"
    assert ins("\\" + N + "\x0b\x7f\x9f\xa0") == "'\\\\\\x00\\x0B\\x7F\\x9F\xa0'"
    assert ins("𐀀\udc00") == "'\U00010000\\udc00'"
    assert ins("y" * 10001) == "'" + "y" * 10000 + "'... 1 more character"
    assert ins("y" * 10002).endswith("... 2 more characters")


def test_inspect_line_split_boundaries() -> None:
    assert ins("x" * 75 + "\n") == "'" + "x" * 75 + "\\n'"  # 76 units: not split
    assert ins("x" * 76 + "\n") == "'" + "x" * 76 + "\\n'"  # 77 units, one piece, no empty tail
    assert ins("a\n" + "x" * 80) == "'a\\n' +\n  '" + "x" * 80 + "'"
    assert ins("x" * 77) == "'" + "x" * 77 + "'"


def test_diagnostic_kind_reason_and_cut() -> None:
    assert (
        diagnostic("file", "a" + N)
        == "The argument 'file' must be a string without null bytes. Received 'a\\x00'"
    )
    assert diagnostic("options.env['K']", "v").startswith("The property ")
    long = diagnostic("args[0]", "a" * 200 + N)
    assert long.endswith("a" * 127 + "...")
    exact = diagnostic("args[0]", "a" * 122 + N)  # inspected is exactly 128 units: not cut
    assert exact.endswith("\\x00'")


def test_env_check_order_both_platforms() -> None:
    baseline, overlay = ["A", "Z", "K"], ["Z", "A", "2", "10", "01", "4294967295", "Z"]
    assert env_check_order(baseline, overlay, Platform.POSIX) == [
        "2",
        "10",
        "Z",
        "A",
        "01",
        "4294967295",
        "K",
    ]
    assert env_check_order(baseline, overlay, Platform.WINDOWS) == sorted(
        {"A", "Z", "K", "2", "10", "01", "4294967295"}, key=units
    )
    assert env_check_order([FULLWIDTH, ASTRAL], [], Platform.POSIX) == [ASTRAL, FULLWIDTH]


LONG = "1" * 4301  # beyond the host's default integer-string conversion limit (4300 digits)


def test_index_domain_neighborhood_and_long_names() -> None:
    """L12D011-I001: only canonical ASCII decimals 0..4294967294 are indices; a longer numeric name
    is an ordinary name and is never converted."""
    overlay = ["4294967294", "4294967295", "12345678901", "1234567890", "007", "١٢", "0", LONG]
    baseline = ["1" * 5000, "Z"]
    try:
        order = env_check_order(baseline, overlay, Platform.POSIX)
    except ValueError as error:  # pragma: no cover - the I001 defect
        raise AssertionError(f"LONG NAME CONVERTED: {error}") from error
    assert order == [
        "0", "1234567890", "4294967294",
        "4294967295", "12345678901", "007", "١٢", LONG,
        "1" * 5000, "Z",
    ], f"INDEX DOMAIN: {[n[:12] for n in order]}"  # fmt: skip
    assert env_check_order(["1" * 5000, LONG], [], Platform.WINDOWS) == ["1" * 4301, "1" * 5000]


async def test_long_numeric_env_name_is_an_ordinary_name_at_the_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L12D011-I001 at the real provider, POSIX branch: a non-NUL spawn with a long numeric name
    reaches the native boundary unchanged; with a NUL entry K, K is reported, zero native calls."""
    calls: list[Any] = []

    async def boundary(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise OSError(2, "native boundary reached")

    monkeypatch.setattr(subprocess_module.asyncio, "create_subprocess_exec", boundary)
    provider = LocalSubprocess(str(tmp_path))
    monkeypatch.setattr(provider, "_platform", Platform.POSIX)
    try:
        plain = await provider.spawn(["x", "ok"], SpawnOptions(env={LONG: "ok"}, inherit_env=False))
        bad = await provider.spawn(
            ["x", "ok"], SpawnOptions(env={LONG: "ok", "K": "v" + N}, inherit_env=False)
        )
    except ValueError as error:  # pragma: no cover - the I001 defect
        raise AssertionError(f"LONG NAME CONVERTED: {error}") from error
    assert isinstance(plain, Err) and not plain.error.nul_input and len(calls) == 1
    assert isinstance(bad, Err) and bad.error.nul_input and "options.env['K']" in bad.error.message
    assert len(calls) == 1, "a rejected spawn reached the native boundary"


def test_spawn_rejection_precedence_and_superseded_value() -> None:
    env = {"K": "ok"}  # the effective value; a superseded baseline value is never checked
    assert spawn_rejection(["x"], "/d", env, ["K"], ["K"], Platform.POSIX) is None
    cwd = spawn_rejection(["x"], "/d" + N, {"A": "a" + N}, [], ["A"], Platform.POSIX)
    assert cwd is not None and "'options.cwd' must be a string, Uint8Array, or URL" in cwd.message
    name = spawn_rejection(["x"], "/d", {"A" + N: "v" + N}, [], ["A" + N], Platform.POSIX)
    assert name is not None and name.message.endswith("Received 'A\\x00'") and name.nul_input
    assert argv_rejection(["ok", "a", "b" + N, "c" + N]) is not None
    assert argv_rejection(["ok", "a"]) is None


# ---- the provider: no native spawn after a rejection; unrelated errors unchanged ----------------
@pytest.mark.parametrize(
    ("argv", "options"),
    [
        (["x" + N], SpawnOptions()),
        (["x", "a" + N], SpawnOptions()),
        (["x"], SpawnOptions(cwd="d" + N)),
        (["x"], SpawnOptions(env={"K": "v" + N}, inherit_env=False)),
    ],
)
async def test_no_native_spawn_after_rejection(
    argv: list[str], options: SpawnOptions, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Any] = []

    async def spy(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must never run
        calls.append(args)
        raise AssertionError("native spawn reached")

    monkeypatch.setattr(subprocess_module.asyncio, "create_subprocess_exec", spy)
    result = await LocalSubprocess(str(tmp_path)).spawn(argv, options)
    assert isinstance(result, Err) and result.error.nul_input and result.error.code == "spawn_error"
    assert calls == []


async def test_unrelated_value_error_still_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def unrelated(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("unrelated")

    monkeypatch.setattr(subprocess_module.asyncio, "create_subprocess_exec", unrelated)
    with pytest.raises(ValueError, match="unrelated"):
        await LocalSubprocess(str(tmp_path)).spawn(["x"], SpawnOptions(inherit_env=False))


# ---- the tools' narrow distinction --------------------------------------------------------------
class _FakeSubprocess:
    def __init__(
        self, inner: LocalSubprocess, error: SubprocessError, snapshot: EnvSnapshot | None = None
    ) -> None:
        self._inner, self._error, self._snapshot = inner, error, snapshot
        self.calls: list[tuple[list[str], SpawnOptions | None]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def base_env(self) -> EnvSnapshot:
        return self._snapshot if self._snapshot is not None else self._inner.base_env()

    async def spawn(self, argv: list[str], options: SpawnOptions | None = None) -> Any:
        self.calls.append((list(argv), options))
        return Err(self._error)


async def test_engine_keeps_its_prefix_for_a_non_nul_failure(tmp_path: Path) -> None:
    fake = _FakeSubprocess(
        LocalSubprocess(str(tmp_path)), SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "The fake")
    )
    with pytest.raises(BuiltinToolError) as caught:
        await _search.spawn_engine(fake, ["fd", "x"], "Failed to run fd")  # type: ignore[arg-type]
    assert str(caught.value) == "Failed to run fd: The fake", f"PREFIX LOST: {caught.value}"


async def test_engine_surfaces_a_seam_nul_rejection_unchanged(tmp_path: Path) -> None:
    error = SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "D", nul_input=True)
    fake = _FakeSubprocess(LocalSubprocess(str(tmp_path)), error)
    with pytest.raises(BuiltinToolError, match=r"^D$"):
        await _search.spawn_engine(fake, ["fd", "x"], "Failed to run fd")  # type: ignore[arg-type]
    rejected = _FakeSubprocess(LocalSubprocess(str(tmp_path)), error)
    with pytest.raises(BuiltinToolError, match=r"^The argument 'args\[0\]'"):
        await _search.spawn_engine(rejected, ["fd", "a" + N], "Failed to run fd")  # type: ignore[arg-type]
    assert rejected.calls == []  # the unprojected argv check precedes the spawn


async def test_engine_diagnostic_shows_the_unprojected_value(tmp_path: Path) -> None:
    fake = _FakeSubprocess(
        LocalSubprocess(str(tmp_path)), SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "x")
    )
    with pytest.raises(BuiltinToolError) as caught:
        await _search.spawn_engine(fake, ["fd", "\ud800" + N], "Failed to run fd")  # type: ignore[arg-type]
    assert str(caught.value).endswith("Received '\\ud800\\x00'"), (
        f"PROJECTED DIAGNOSTIC: {caught.value!a}"
    )
    assert fake.calls == []


async def _run_bash(fake: _FakeSubprocess, tmp_path: Path, command: str) -> str:
    """The real bash tool through the Layer-06 pipeline; returns the error result's text."""
    registry = ToolRegistry()
    registry.register(create_bash_tool(LocalFileSystem(str(tmp_path)), fake))  # type: ignore[arg-type]
    ctx = Context()
    declare_tools_events(ctx.events)
    result = await execute_call(
        ToolCallBlock(id="c", name="bash", arguments={"command": command}),
        registry=registry,
        ctx=ctx,
        signal=RunAbortController().signal,
    )
    first = result.content[0]
    assert result.is_error and isinstance(first, TextBlock)
    return first.text


async def test_bash_seam_rejection_unchanged_and_other_failures_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def argv_shell(*args: Any) -> ShellConfig:
        return ShellConfig("sh", ("-c",), "argv")

    monkeypatch.setattr("minion_agent.tools.builtin.bash.select_shell", argv_shell)
    seam = SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "D", nul_input=True)
    assert (
        await _run_bash(_FakeSubprocess(LocalSubprocess(str(tmp_path)), seam), tmp_path, "true")
        == "D"
    )
    other = SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "The fake")
    assert await _run_bash(
        _FakeSubprocess(LocalSubprocess(str(tmp_path)), other), tmp_path, "true"
    ) == ("Failed to start the shell sh")


async def test_bash_stdin_transport_command_is_not_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stdin_shell(*args: Any) -> ShellConfig:
        return ShellConfig("wsl-bash", ("-s",), "stdin")

    monkeypatch.setattr("minion_agent.tools.builtin.bash.select_shell", stdin_shell)
    fake = _FakeSubprocess(
        LocalSubprocess(str(tmp_path)), SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "x")
    )
    assert await _run_bash(fake, tmp_path, "echo a" + N) == "Failed to start the shell wsl-bash"
    assert fake.calls and fake.calls[0][0] == [
        "wsl-bash",
        "-s",
    ]  # spawned: the NUL is stdin, not argv


async def test_bash_consumer_arbitration_drops_the_colliding_entry_before_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def argv_shell(*args: Any) -> ShellConfig:
        return ShellConfig("sh", ("-c",), "argv")

    monkeypatch.setattr("minion_agent.tools.builtin.bash.select_shell", argv_shell)
    snapshot = EnvSnapshot([("K", "ok"), ("k", "bad" + N)], Platform.WINDOWS)
    fake = _FakeSubprocess(
        LocalSubprocess(str(tmp_path)),
        SubprocessError(SubprocessErrorCode.SPAWN_ERROR, "x"),
        snapshot,
    )
    assert await _run_bash(fake, tmp_path, "true") == "Failed to start the shell sh"
    env = fake.calls[0][1].env  # type: ignore[union-attr]
    assert env is not None and env.get("K") == "ok" and "k" not in env
    assert (
        spawn_rejection(fake.calls[0][0], str(tmp_path), env, [], list(env), Platform.WINDOWS)
        is None
    )


def test_bash_shell_module_has_both_transports() -> None:
    assert bash_shell.ShellConfig("s", ("-c",), "argv").transport == "argv"
