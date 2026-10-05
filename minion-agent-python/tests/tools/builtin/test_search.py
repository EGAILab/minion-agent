"""WP-13.4 `find`/`grep` binding witnesses (spec/tools.md WP-13.4 "Witnesses and evidence" items
2-6, and the negative controls they kill). The canonical scenarios
(`tests/conformance/test_builtin_search_conformance.py`) run the real pinned engines; these use the
explicit, uncertified `EngineOverride` to drive a scripted engine (`search_fake_engine.py`)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import LocalFileSystem, Platform
from minion_agent.execution.result import Ok
from minion_agent.execution.subprocess import ExitStatus, LocalSubprocess
from minion_agent.execution.world import ExecutionWorldIdentity
from minion_agent.runtime import RunAbortController
from minion_agent.tools.builtin import search_engines
from minion_agent.tools.builtin._node_path import NodePath
from minion_agent.tools.builtin._readline import LineSplitter
from minion_agent.tools.builtin.find import (
    _exists,
    _windows_full_path,
    create_find_tool,
    relativize,
)
from minion_agent.tools.builtin.grep import _js_index, create_grep_tool, truncate_line
from minion_agent.tools.builtin.paths import BuiltinToolError
from minion_agent.tools.builtin.search_engines import (
    PINS,
    Artifact,
    EngineOverride,
    EnginePin,
    EngineStore,
    ProvisioningError,
    host_platform,
    provision_search_engines,
)

FAKE = str(Path(__file__).with_name("search_fake_engine.py"))
OVERRIDE = EngineOverride({"fd": [sys.executable, FAKE], "rg": [sys.executable, FAKE]})
SEP_CLASS = "[/\\\\]"


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    log = tmp_path / "engine.log"
    monkeypatch.setenv("FAKE_ENGINE_LOG", str(log))
    for name in (
        "FAKE_ENGINE_STDOUT_FILE",
        "FAKE_ENGINE_STDERR",
        "FAKE_ENGINE_EXIT",
        "FAKE_ENGINE_SLEEP",
    ):
        monkeypatch.delenv(name, raising=False)

    def script(
        stdout: str | bytes = b"", stderr: str = "", exit: int = 0, sleep: float = 0
    ) -> None:
        data = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
        out = tmp_path / "engine.stdout"
        out.write_bytes(data)
        monkeypatch.setenv("FAKE_ENGINE_STDOUT_FILE", str(out))
        monkeypatch.setenv("FAKE_ENGINE_STDERR", stderr)
        monkeypatch.setenv("FAKE_ENGINE_EXIT", str(exit))
        monkeypatch.setenv("FAKE_ENGINE_SLEEP", str(sleep))

    def argv() -> list[list[str]]:
        if not log.exists():
            return []
        return [json.loads(line)["argv"] for line in log.read_text(encoding="utf-8").splitlines()]

    root = tmp_path / "root"
    root.mkdir()
    return {"script": script, "argv": argv, "root": root}


def _tools(root: Path, engines: Any = OVERRIDE) -> tuple[Any, Any]:
    fs, sp = LocalFileSystem(str(root)), LocalSubprocess(str(root))
    return create_find_tool(fs, sp, engines), create_grep_tool(fs, sp, engines)


async def _run(tool: Any, args: dict[str, Any], signal: Any = None) -> tuple[str, dict[str, Any]]:
    try:
        result = await tool.execute("c", args, signal)
    except BuiltinToolError as error:
        return "ERR " + str(error), {}
    return result.content[0].text, result.details


# ---- witness 4: argument vectors ----


async def test_find_argument_vectors(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    find, _ = _tools(root)
    native = str(root)
    cases = [
        ({"pattern": "*.ts"}, ["--max-results", "1000", "--", "*.ts"]),
        ({"pattern": "*.ts", "limit": 2.5}, ["--max-results", "2.5", "--", "*.ts"]),
        ({"pattern": "*.ts", "limit": -0.0}, ["--max-results", "0", "--", "*.ts"]),
        ({"pattern": "*.ts", "limit": math.inf}, ["--max-results", "Infinity", "--", "*.ts"]),
    ]
    for args, tail in cases:
        await _run(find, args)
        assert engine["argv"]()[-1] == [
            "--glob",
            "--color=never",
            "--hidden",
            "--no-require-git",
            *tail,
            native,
        ]
    windows = os.name == "nt"
    full = {
        "src/x.ts": f"**{SEP_CLASS}src{SEP_CLASS}x.ts" if windows else "**/src/x.ts",
        "/abs/x": f"{SEP_CLASS}abs{SEP_CLASS}x" if windows else "/abs/x",
        "**/x": f"**{SEP_CLASS}x" if windows else "**/x",
        "**": "**",
    }
    for pattern, effective in full.items():
        await _run(find, {"pattern": pattern})
        observed = engine["argv"]()[-1]
        assert observed[observed.index("--") + 1] == effective
        assert ("--full-path" in observed) == ("/" in pattern)
    (root / ".git").mkdir()
    await _run(find, {"pattern": "*.ts"})
    assert "--no-require-git" not in engine["argv"]()[-1]


async def test_grep_argument_vectors(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    _, grep = _tools(root)
    base = ["--json", "--line-number", "--color=never", "--hidden"]
    cases = [
        ({"pattern": "foo"}, base),
        (
            {"pattern": "foo", "ignoreCase": True, "literal": True, "glob": "*.ts"},
            [*base, "--ignore-case", "--fixed-strings", "--glob", "*.ts"],
        ),
        ({"pattern": "foo", "ignoreCase": False, "literal": False, "glob": ""}, base),
        ({"pattern": "-v"}, base),
    ]
    for args, prefix in cases:
        await _run(grep, args)
        assert engine["argv"]()[-1] == [*prefix, "--", args["pattern"], str(root)]


async def test_an_unpaired_surrogate_reaches_the_engine_as_replacement(
    engine: dict[str, Any],
) -> None:
    find, grep = _tools(engine["root"])
    await _run(find, {"pattern": "a\ud800b"})
    await _run(grep, {"pattern": "\udc00x\ud83d\ude00"})
    find_argv, grep_argv = engine["argv"]()[-2:]
    assert "a\ufffdb" in find_argv
    assert "\ufffdx\U0001f600" in grep_argv


# ---- witness 6: order preservation within one invocation ----


async def test_find_reproduces_an_unsorted_engine_stream(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    sep = os.sep
    engine["script"](stdout=f"{root}{sep}z.ts\n{root}{sep}a.ts\n")
    find, _ = _tools(root)
    assert await _run(find, {"pattern": "*.ts"}) == ("z.ts\na.ts", {})


def _match(path: str, line: int, text: str | None = "x\n") -> str:
    data: dict[str, Any] = {"path": {"text": path}, "line_number": line}
    if text is not None:
        data["lines"] = {"text": text}
    return json.dumps({"type": "match", "data": data})


async def test_grep_reproduces_an_unsorted_engine_stream(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    stream = "\n".join(
        [
            _match(str(root / "z.ts"), 2),
            _match(str(root / "a.ts"), 1),
            _match(str(root / "z.ts"), 1),
        ]
    )
    engine["script"](stdout=stream + "\n", exit=0)
    _, grep = _tools(root)
    assert await _run(grep, {"pattern": "x"}) == ("z.ts:2: x\na.ts:1: x\nz.ts:1: x", {})


# ---- find: output collection and errors ----


async def test_find_trims_skips_blank_and_keeps_a_partial_failure(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    engine["script"](
        stdout=f"  {root}{os.sep}a.ts  \r\n\n{root}{os.sep}d{os.sep}\n", stderr="warn", exit=1
    )
    find, _ = _tools(root)
    assert await _run(find, {"pattern": "*"}) == ("a.ts\nd/", {})


async def test_find_engine_failures(engine: dict[str, Any]) -> None:
    find, _ = _tools(engine["root"])
    engine["script"](stderr="  [fd error]: boom \n", exit=1)
    assert await _run(find, {"pattern": "*"}) == ("ERR [fd error]: boom", {})
    engine["script"](exit=2)
    assert await _run(find, {"pattern": "*"}) == ("ERR fd exited with code 2", {})
    engine["script"]()
    assert await _run(find, {"pattern": "*"}) == ("No files found matching pattern", {})


async def test_find_tests_emptiness_before_trimming(engine: dict[str, Any]) -> None:
    """Pi joins the raw lines and tests that for emptiness; trimming happens per line afterwards.
    Whitespace-only output with a failing exit is therefore a success with an empty body."""
    engine["script"](stdout="  \n", stderr="warn", exit=1)
    find, _ = _tools(engine["root"])
    assert await _run(find, {"pattern": "*"}) == ("", {})


async def test_find_keeps_entries_that_format_identically(engine: dict[str, Any]) -> None:
    """Witness 7 at binding level (`WP134-CON-R002`): two distinct names that Pi's trim makes equal
    both print -- multiplicity is kept, nothing deduplicates."""
    root: Path = engine["root"]
    sep = os.sep
    engine["script"](stdout=f"{root}{sep}src{sep}same.ts\n{root}{sep}src{sep}same.ts \n")
    find, _ = _tools(root)
    assert await _run(find, {"pattern": "same*"}) == ("src/same.ts\nsrc/same.ts", {})


async def test_spawn_failure_texts(tmp_path: Path) -> None:
    missing = EngineOverride(
        {"fd": [str(tmp_path / "no-such-engine")], "rg": [str(tmp_path / "no-such-engine")]}
    )
    (tmp_path / "root").mkdir()
    find, grep = _tools(tmp_path / "root", missing)
    assert (await _run(find, {"pattern": "*"}))[0].startswith("ERR Failed to run fd: ")
    assert (await _run(grep, {"pattern": "x"}))[0].startswith("ERR Failed to run ripgrep: ")


async def test_override_without_the_engine(tmp_path: Path) -> None:
    find, grep = _tools(tmp_path, EngineOverride({}))
    assert await _run(find, {"pattern": "*"}) == (
        "ERR fd is not configured in the engine override.",
        {},
    )
    assert await _run(grep, {"pattern": "x"}) == (
        "ERR ripgrep (rg) is not configured in the engine override.",
        {},
    )


# ---- cancellation: find settles at once, grep after the engine exits ----
# (grep's abort listener starts at spawn)


async def test_find_abort_settles_at_once_and_stops_the_engine(engine: dict[str, Any]) -> None:
    engine["script"](sleep=20)
    find, _ = _tools(engine["root"])
    controller = RunAbortController()
    asyncio.get_running_loop().call_later(0.5, controller.abort)
    started = time.monotonic()
    assert await _run(find, {"pattern": "*"}, controller.signal) == ("ERR Operation aborted", {})
    assert time.monotonic() - started < 5
    await asyncio.sleep(0.5)  # the abandoned work has terminated the engine


async def test_grep_abort_after_spawn_settles_after_exit(engine: dict[str, Any]) -> None:
    engine["script"](sleep=20)
    _, grep = _tools(engine["root"])
    controller = RunAbortController()
    asyncio.get_running_loop().call_later(0.5, controller.abort)
    started = time.monotonic()
    assert await _run(grep, {"pattern": "x"}, controller.signal) == ("ERR Operation aborted", {})
    assert time.monotonic() - started < 5


async def test_pre_aborted_calls(engine: dict[str, Any]) -> None:
    find, grep = _tools(engine["root"])
    controller = RunAbortController()
    controller.abort()
    assert await _run(find, {"pattern": "*"}, controller.signal) == ("ERR Operation aborted", {})
    assert await _run(grep, {"pattern": "x"}, controller.signal) == ("ERR Operation aborted", {})


class _AbortingOverride:
    """Aborts the call's signal while the engine is being resolved -- before grep's listener."""

    def __init__(self, controller: RunAbortController) -> None:
        self.controller = controller

    async def resolve(self, engine: str) -> list[str]:
        self.controller.abort()
        return [sys.executable, FAKE]


async def test_grep_does_not_observe_an_abort_before_spawn(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    engine["script"](stdout=_match(str(root / "a.ts"), 1) + "\n")
    controller = RunAbortController()
    fs, sp = LocalFileSystem(str(root)), LocalSubprocess(str(root))
    grep = create_grep_tool(fs, sp, _AbortingOverride(controller))  # type: ignore[arg-type]
    assert await _run(grep, {"pattern": "x"}, controller.signal) == ("a.ts:1: x", {})


class _ScriptedStream:
    """An in-memory engine pipe: its chunks, then EOF, with no real I/O between reads."""

    def __init__(self, chunks: list[bytes], on_read: Any = None) -> None:
        self.chunks, self.on_read = chunks, on_read

    async def read_chunk(self) -> Any:
        if not self.chunks:
            return Ok(None)
        if self.on_read is not None:
            self.on_read()
        return Ok(self.chunks.pop(0))

    async def close(self) -> None:
        return None


class _ScriptedProcess:
    def __init__(self, stdout: _ScriptedStream) -> None:
        self.stdout, self.stderr, self.terminated = stdout, _ScriptedStream([]), 0

    async def wait(self) -> Any:
        return Ok(ExitStatus(0))

    async def terminate(self) -> None:
        self.terminated += 1


class _ScriptedSubprocess(LocalSubprocess):
    """Spawns a scripted process whose first stdout read aborts the call's signal: the abort lands
    after grep's listener registration, and the run completes within the same event-loop turns,
    before any polling tick (WP134-IMPL-R001)."""

    def __init__(self, cwd: str, controller: RunAbortController, abort_on: str) -> None:
        super().__init__(cwd)
        self.controller, self.abort_on = controller, abort_on
        self.processes: list[_ScriptedProcess] = []

    async def spawn(self, argv: Any, options: Any = None) -> Any:
        if self.abort_on == "spawn":
            self.controller.abort()
        on_read = self.controller.abort if self.abort_on == "first_read" else None
        line = _match(str(Path(self.cwd) / "a.ts"), 1) + "\n"
        process = _ScriptedProcess(_ScriptedStream([line.encode()], on_read))
        self.processes.append(process)
        return Ok(process)


async def test_grep_keeps_an_abort_that_lands_just_before_completion(tmp_path: Path) -> None:
    """Pi grep.ts: the listener, registered after spawn, sets `aborted` synchronously and the close
    handler rejects, even when the engine then exits 0 with a match; onAbort kills the child."""
    controller = RunAbortController()
    sp = _ScriptedSubprocess(str(tmp_path), controller, "first_read")
    grep = create_grep_tool(LocalFileSystem(str(tmp_path)), sp, OVERRIDE)
    assert await _run(grep, {"pattern": "x"}, controller.signal) == ("ERR Operation aborted", {})
    assert sp.processes[0].terminated == 1


async def test_grep_ignores_an_abort_before_its_listener_registration(tmp_path: Path) -> None:
    """An abort during spawn precedes the registration: Pi's listener never fires for it."""
    controller = RunAbortController()
    sp = _ScriptedSubprocess(str(tmp_path), controller, "spawn")
    grep = create_grep_tool(LocalFileSystem(str(tmp_path)), sp, OVERRIDE)
    assert await _run(grep, {"pattern": "x"}, controller.signal) == ("a.ts:1: x", {})
    assert sp.processes[0].terminated == 0


async def test_find_aborted_during_engine_resolution(engine: dict[str, Any]) -> None:
    controller = RunAbortController()
    fs, sp = LocalFileSystem(str(engine["root"])), LocalSubprocess(str(engine["root"]))
    find = create_find_tool(fs, sp, _AbortingOverride(controller))  # type: ignore[arg-type]
    assert await _run(find, {"pattern": "*"}, controller.signal) == ("ERR Operation aborted", {})


# ---- grep: collection, context, limits ----


async def test_grep_counts_uncollected_matches_and_skips_noise(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    stream = [
        "not json",
        "",
        "[1]",
        json.dumps({"type": "begin"}),
        json.dumps({"type": "match", "data": "x"}),
        json.dumps({"type": "match", "data": {"path": {"bytes": "AAE="}, "line_number": 1}}),
    ]
    engine["script"](stdout="\n".join(stream) + "\n")
    _, grep = _tools(root)
    assert await _run(grep, {"pattern": "x"}) == ("", {})


async def test_grep_limits_and_kill_for_limit(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    many = "\n".join(_match(str(root / "a.ts"), n) for n in range(1, 6)) + "\n"
    engine["script"](stdout=many, exit=2)
    _, grep = _tools(root)
    text, details = await _run(grep, {"pattern": "x", "limit": 0})
    assert text == "a.ts:1: x\n\n[1 matches limit reached. Use limit=2 for more, or refine pattern]"
    assert details == {"matchLimitReached": 1}
    text, details = await _run(grep, {"pattern": "x", "limit": 2.5})
    assert text.startswith(
        "a.ts:1: x\na.ts:2: x\na.ts:3: x\n\n[2.5 matches limit reached. Use limit=5"
    ), text
    assert details == {"matchLimitReached": 2.5}
    engine["script"](stdout=many, exit=0)
    assert (await _run(grep, {"pattern": "x", "limit": math.inf}))[0].count("\n") == 4


async def test_grep_engine_failures(engine: dict[str, Any]) -> None:
    _, grep = _tools(engine["root"])
    engine["script"](exit=2)
    assert await _run(grep, {"pattern": "x"}) == ("ERR ripgrep exited with code 2", {})
    engine["script"](stderr="rg: bad\n", exit=2)
    assert await _run(grep, {"pattern": "x"}) == ("ERR rg: bad", {})
    engine["script"](exit=1)
    assert await _run(grep, {"pattern": "x"}) == ("No matches found", {})


async def test_grep_path_not_found_and_file_path(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    (root / "f.txt").write_text("a\r\nb match\r\n")
    _, grep = _tools(root)
    text, _ = await _run(grep, {"pattern": "x", "path": "missing"})
    assert text == f"ERR Path not found: {root / 'missing'}"
    engine["script"](stdout=_match(str(root / "f.txt"), 2, "b match\r\n") + "\n")
    assert await _run(grep, {"pattern": "x", "path": "f.txt"}) == ("f.txt:2: b match", {})


async def test_grep_context_reconstruction(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    (root / "c.txt").write_bytes(b"\xef\xbb\xbfL1\r\nL2\rL3\n")
    _, grep = _tools(root)
    engine["script"](stdout=_match(str(root / "c.txt"), 2) + "\n")
    assert (await _run(grep, {"pattern": "x", "context": 1}))[
        0
    ] == "c.txt-1- \ufeffL1\nc.txt:2: L2\nc.txt-3- L3"
    assert (await _run(grep, {"pattern": "x", "context": -1}))[0] == "c.txt:2: x"
    assert (await _run(grep, {"pattern": "x", "context": -0.0}))[0] == "c.txt:2: x"
    assert (await _run(grep, {"pattern": "x", "context": 0.5}))[0] == "c.txt-1.5- \nc.txt-2.5- "
    assert (await _run(grep, {"pattern": "x", "context": math.inf}))[0] == (
        "c.txt-1- \ufeffL1\nc.txt:2: L2\nc.txt-3- L3\nc.txt-4- "
    )
    engine["script"](stdout=_match(str(root / "gone.txt"), 3, None) + "\n")
    assert (await _run(grep, {"pattern": "x"}))[0] == "gone.txt:3: (unable to read file)"
    engine["script"](
        stdout=_match(str(root / "c.txt"), 1, None)
        + "\n"
        + _match(str(root / "c.txt"), 2, None)
        + "\n"
    )
    assert (await _run(grep, {"pattern": "x"}))[0] == "c.txt:1: \ufeffL1\nc.txt:2: L2"


async def test_grep_line_truncation_and_notices(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    long = "a" * 499 + "\U0001f600" + "tail"
    engine["script"](stdout=_match(str(root / "a.ts"), 1, long + "\n") + "\n")
    _, grep = _tools(root)
    text, details = await _run(grep, {"pattern": "x"})
    assert (
        text
        == "a.ts:1: "
        + "a" * 499
        + "\ud83d... [truncated]\n\n"
        + "[Some lines truncated to 500 chars. Use read tool to see full lines]"
    )
    assert details == {"linesTruncated": True}


async def test_find_byte_truncation_details(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    names = "".join(f"{root}{os.sep}{'n' * 100}{i:04d}\n" for i in range(600))
    engine["script"](stdout=names)
    find, _ = _tools(root)
    text, details = await _run(find, {"pattern": "*"})
    assert text.endswith("\n\n[50.0KB limit reached]")
    truncation = details["truncation"]
    assert set(truncation) == {
        "content",
        "truncated",
        "truncatedBy",
        "totalLines",
        "totalBytes",
        "outputLines",
        "outputBytes",
        "lastLinePartial",
        "firstLineExceedsLimit",
        "maxLines",
        "maxBytes",
    }
    assert truncation["truncatedBy"] == "bytes" and truncation["maxLines"] == 9007199254740991


# ---- DIV-003: the engine store and provisioning ----


async def test_unprovisioned_store_texts(tmp_path: Path) -> None:
    store = EngineStore(root=tmp_path / "store", platform="linux-x64")
    with pytest.raises(BuiltinToolError) as error:
        await store.resolve("fd")
    assert str(error.value) == (
        "fd is not provisioned: the certified fd 10.4.2 engine is missing or failed verification. "
        "Run provision_search_engines() to provision it."
    )
    store.root.mkdir()
    store.binary_path("rg").write_bytes(b"not the engine")
    with pytest.raises(BuiltinToolError) as error:
        await store.resolve("rg")
    assert str(error.value).startswith(
        "ripgrep (rg) is not provisioned: the certified ripgrep 15.2.0 engine"
    )
    darwin = EngineStore(root=tmp_path, platform="darwin-arm64")
    with pytest.raises(BuiltinToolError) as error:
        await darwin.resolve("fd")
    assert (
        str(error.value)
        == "fd is not available on this platform: no certified fd engine for darwin-arm64."
    )


async def test_a_store_needs_the_local_execution_world(tmp_path: Path) -> None:
    remote = ExecutionWorldIdentity("remote-box")
    fs = LocalFileSystem(str(tmp_path), execution_world=remote)
    find = create_find_tool(
        fs, LocalSubprocess(str(tmp_path), execution_world=remote), EngineStore(root=tmp_path)
    )
    assert await _run(find, {"pattern": "*"}) == (
        "ERR fd is not available on this platform: no certified fd engine for remote-box "
        "(non-local execution world).",
        {},
    )


def _fake_pins(tmp_path: Path, *, corrupt_binary: bool = False) -> dict[str, Any]:
    pins: dict[str, Any] = {}
    for engine, kind in (("fd", "zip"), ("rg", "tar")):
        binary = f"{engine}-binary".encode()
        buffer = io.BytesIO()
        if kind == "zip":
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr(f"pkg/{engine}", binary)
            name = f"{engine}.zip"
        else:
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                info = tarfile.TarInfo(f"pkg/{engine}")
                info.size = len(binary)
                archive.addfile(info, io.BytesIO(binary))
                archive.addfile(tarfile.TarInfo("pkg/dir") if False else _dir_info())
            name = f"{engine}.tar.gz"
        data = buffer.getvalue()
        (tmp_path / name).write_bytes(data)
        digest = hashlib.sha256(binary).hexdigest()
        pins[engine] = EnginePin(
            PINS[engine].label,
            PINS[engine].engine,
            PINS[engine].version,
            {
                "test-x64": Artifact(
                    f"https://example.invalid/{name}",
                    hashlib.sha256(data).hexdigest(),
                    f"pkg/{engine}",
                    "0" * 64 if corrupt_binary else digest,
                ),
            },
        )
    return pins


def _dir_info() -> tarfile.TarInfo:
    info = tarfile.TarInfo("pkg/dir")
    info.type = tarfile.DIRTYPE
    return info


def test_provisioning_installs_verifies_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search_engines, "PINS", _fake_pins(tmp_path))
    store = EngineStore(root=tmp_path / "store", platform="test-x64")
    assert provision_search_engines(store, source=tmp_path) == {
        "fd": "installed",
        "rg": "installed",
    }
    assert store.is_verified("fd") and store.is_verified("rg")
    assert provision_search_engines(store, source=tmp_path) == {"fd": "present", "rg": "present"}
    assert not list(store.root.glob(".*partial"))


def test_provisioning_refuses_bad_artifacts_and_binaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search_engines, "PINS", _fake_pins(tmp_path))
    (tmp_path / "fd.zip").write_bytes(b"tampered")
    store = EngineStore(root=tmp_path / "store", platform="test-x64")
    with pytest.raises(ProvisioningError, match="artifact SHA-256 mismatch"):
        provision_search_engines(store, source=tmp_path)
    assert not store.binary_path("fd").exists()
    monkeypatch.setattr(search_engines, "PINS", _fake_pins(tmp_path, corrupt_binary=True))
    with pytest.raises(ProvisioningError, match="binary SHA-256 mismatch"):
        provision_search_engines(store, source=tmp_path)
    assert not store.binary_path("fd").exists()
    with pytest.raises(ProvisioningError, match="not available on this platform"):
        provision_search_engines(
            EngineStore(root=tmp_path, platform="darwin-arm64"), source=tmp_path
        )


def test_provisioning_rejects_a_non_file_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins = _fake_pins(tmp_path)
    rg = pins["rg"].platforms["test-x64"]
    pins["rg"] = EnginePin(
        "ripgrep (rg)",
        "ripgrep",
        "15.2.0",
        {"test-x64": Artifact(rg.url, rg.artifact_sha256, "pkg/dir", rg.binary_sha256)},
    )
    monkeypatch.setattr(search_engines, "PINS", pins)
    with pytest.raises(ProvisioningError, match="is not a file"):
        provision_search_engines(
            EngineStore(root=tmp_path / "store", platform="test-x64"), source=tmp_path
        )


def test_an_interrupted_install_leaves_nothing_that_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search_engines, "PINS", _fake_pins(tmp_path))
    store = EngineStore(root=tmp_path / "store", platform="test-x64")

    def interrupted(src: str, dst: str) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(search_engines.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        provision_search_engines(store, source=tmp_path)
    assert not store.is_verified("fd")
    assert not list(store.root.iterdir())


def test_the_fixed_name_appears_only_by_the_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine's fixed name never exists as a partly written file: it is created only by the
    rename of a fully written temporary file."""
    monkeypatch.setattr(search_engines, "PINS", _fake_pins(tmp_path))
    store = EngineStore(root=tmp_path / "store", platform="test-x64")
    rename = os.replace
    renamed: list[str] = []

    def spy(src: str, dst: str) -> None:
        assert not Path(dst).exists(), f"{dst} existed before the atomic rename"
        assert Path(src).name.endswith(".partial"), f"{src} is not a temporary file"
        renamed.append(Path(dst).name)
        rename(src, dst)

    monkeypatch.setattr(search_engines.os, "replace", spy)
    provision_search_engines(store, source=tmp_path)
    assert sorted(renamed) == sorted(store.binary_path(e).name for e in ("fd", "rg"))


class DownloadAttemptedError(Exception):
    pass


async def test_a_tool_call_never_downloads_or_uses_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`DIV-003`: an unprovisioned engine is an error at call time -- no download, and no `PATH`
    lookup even when same-named executables are on `PATH`."""
    on_path = tmp_path / "bin"
    on_path.mkdir()
    for name in ("fd", "rg"):
        for suffix in ("", ".exe", ".cmd"):
            planted = on_path / f"{name}{suffix}"
            planted.write_text(
                "@echo planted\n" if suffix == ".cmd" else "#!/bin/sh\necho planted\n"
            )
            planted.chmod(0o755)
    monkeypatch.setenv("PATH", f"{on_path}{os.pathsep}{os.environ.get('PATH', '')}")

    def download(url: str) -> bytes:
        raise DownloadAttemptedError(url)

    monkeypatch.setattr(search_engines, "_fetch", download)
    root = tmp_path / "root"
    root.mkdir()
    find, grep = _tools(root, EngineStore(root=tmp_path / "empty-store"))
    find_text, _ = await _run(find, {"pattern": "*"})
    grep_text, _ = await _run(grep, {"pattern": "x"})
    assert find_text.startswith("ERR fd is not provisioned: "), find_text
    assert grep_text.startswith("ERR ripgrep (rg) is not provisioned: "), grep_text


def test_an_unpinned_platform_never_verifies(tmp_path: Path) -> None:
    assert not EngineStore(root=tmp_path, platform="darwin-arm64").is_verified("fd")


async def test_repository_marker_probe_per_platform(tmp_path: Path) -> None:
    """`pathExists` mapping (`WP134-CON-R001`): non-following `file_info` on Windows,
    `probe_dir_entry` on POSIX; any error is false. Both mappings are exercised on either host."""
    fs = LocalFileSystem(str(tmp_path))
    (tmp_path / ".git").mkdir()
    for platform in (Platform.WINDOWS, Platform.POSIX):
        assert await _exists(fs, platform, str(tmp_path / ".git"))
        assert not await _exists(fs, platform, str(tmp_path / "missing" / ".git"))


async def test_grep_byte_truncation_notice(engine: dict[str, Any]) -> None:
    root: Path = engine["root"]
    stream = "".join(_match(str(root / "w.txt"), n, "w" * 490 + "\n") + "\n" for n in range(1, 121))
    engine["script"](stdout=stream)
    _, grep = _tools(root)
    text, details = await _run(grep, {"pattern": "x", "limit": 200})
    assert text.endswith("\n\n[50.0KB limit reached]")
    assert set(details) == {"truncation"}
    assert details["truncation"]["truncatedBy"] == "bytes"


def test_host_platform_and_default_store() -> None:
    assert host_platform() in {"win32-x64", "linux-x64"} or "-" in host_platform()
    assert EngineStore().root.parts[-2:] == (".minion", "search-engines")


# ---- pure helpers ----


def test_windows_full_path_normalization() -> None:
    """DIV-002: each genuine `**/` component is optional in one top-level alternation (fd's glob has
    no nested alternation); adjacent components collapse; the user's braces are distributed."""
    s = SEP_CLASS
    d = f"**{s}"  # the pattern-initial component, never optional

    def alternation(*variants: str) -> str:
        return "{" + ",".join(d + v for v in variants) + "}"

    assert _windows_full_path("**/src/**/*.spec.ts") == alternation(
        f"src{s}**{s}*.spec.ts", f"src{s}*.spec.ts"
    )
    assert _windows_full_path("**/src/**/**/*.spec.ts") == _windows_full_path("**/src/**/*.spec.ts")
    assert _windows_full_path("**/a/**/b/**/c") == alternation(
        f"a{s}**{s}b{s}**{s}c", f"a{s}b{s}**{s}c", f"a{s}**{s}b{s}c", f"a{s}b{s}c"
    )
    assert _windows_full_path("**/{src/**/b.ts,none}") == alternation(
        f"src{s}**{s}b.ts", f"src{s}b.ts", "none"
    )
    assert _windows_full_path("**/{a,b}/**/x") == alternation(
        f"a{s}**{s}x", f"a{s}x", f"b{s}**{s}x", f"b{s}x"
    )
    # Plain `,` `{` `}` keep their literal meaning inside the generated alternation.
    assert _windows_full_path("**/a,b}/**/x") == alternation(
        f"a[,]b[}}]{s}**{s}x", f"a[,]b[}}]{s}x"
    )
    # Without a genuine `**/` component -- or when the braces or a class are not well formed --
    # the text is exactly Pi's `replaceAll("/", "[/\\]")`.
    for pi_scope in (
        "**/src/*.ts",
        "**/a\\/b/[/]x",
        "**/[",
        "**/a\\*/b",
        "**/x\\/**\\/y",
        "**/[/**/]z",
        "/abs/x",
        "**/x",
        "**/src/**",
        "**/{a,b}/c",
        "**/{a/**/{b,c}}/x",
        "**/{a/**/b",
    ):
        assert _windows_full_path(pi_scope) == pi_scope.replace("/", s)


def test_relativize_and_node_paths() -> None:
    win, posix = NodePath(Platform.WINDOWS), NodePath(Platform.POSIX)
    assert relativize("C:\\r\\a\\b.ts", "C:\\r", win) == "a/b.ts"
    assert relativize("c:\\R\\a\\", "C:\\r", win) == "a/"
    assert relativize("C:\\r\\a/", "C:\\r", win) == "a/"
    assert relativize("D:\\x\\y", "C:\\r", win) == "D:/x/y"
    assert relativize("rel/x", "C:\\r", win) == "rel/x"
    assert relativize("/r/a/b", "/r", posix) == "a/b"
    assert relativize("/r/d/", "/r", posix) == "d/"
    assert win.relative("C:\\r", "c:\\R") == "" and posix.relative("/r", "/r") == ""
    assert win.is_absolute("\\x") and win.is_absolute("C:/x") and not win.is_absolute("C:x")
    assert posix.is_absolute("/x") and not posix.is_absolute("x")
    assert win.dirname("C:\\a\\b") == "C:\\a" and posix.dirname("/a/b") == "/a"
    assert win.join("C:\\a", ".git") == "C:\\a\\.git" and posix.join("/a", ".git") == "/a/.git"
    assert win.basename("C:\\a\\b.ts") == "b.ts" and posix.basename("/a/b/") == "b"


def test_readline_splitting() -> None:
    splitter = LineSplitter()
    assert splitter.feed(b"a\r") == ["a"]
    assert splitter.feed(b"\nb\rc\r\nd\n\xe2\x82") == ["b", "c", "d"]
    assert splitter.feed(b"\xac\xff") == []
    assert splitter.finish() == ["\u20ac\ufffd"]
    tail = LineSplitter()
    assert tail.feed(b"x\r") == ["x"] and tail.feed(b"y\n") == ["y"] and tail.finish() == []


def test_truncate_line_and_js_index() -> None:
    assert truncate_line("x" * 500) == ("x" * 500, False)
    assert truncate_line("x" * 501) == ("x" * 500 + "... [truncated]", True)
    assert (
        _js_index(["a", "b"], 1.0) == "b"
        and _js_index(["a"], 0.5) == ""
        and _js_index(["a"], -1) == ""
    )


def test_find_and_grep_require_one_execution_world(tmp_path: Path) -> None:
    other = LocalFileSystem(str(tmp_path), execution_world=ExecutionWorldIdentity("elsewhere"))
    with pytest.raises(Exception, match=r"find requires ctx\.fs and ctx\.subprocess"):
        create_find_tool(other, LocalSubprocess(str(tmp_path)), OVERRIDE)
    with pytest.raises(Exception, match=r"grep requires ctx\.fs and ctx\.subprocess"):
        create_grep_tool(other, LocalSubprocess(str(tmp_path)), OVERRIDE)
