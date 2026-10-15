"""Execute the L12-D012 `path-resolution` canonical cases on this host's platform (spec/execution.md
section 3.2; record 12-l12-d012-path-colon.md sections 9 and 10) through the real `ctx.fs`
`absolute_path` seam, plus the permanent controls the contract requires of every runner. L12-D013
(record 12-l12-d013-rooted.md sections 7 and 8) adds the rooted and UNC-root cases, the extended
child cwd form and controls D013-C1..C5."""

from __future__ import annotations

import json
import ntpath
import os
import subprocess as subprocess_module
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from minion_agent.execution import filesystem

from . import path_resolution_runner as runner
from .path_resolution_runner import (
    DOCUMENT,
    HOST_PLATFORM,
    instantiate,
    precondition_refusal,
    run_baseline_case,
    run_host_case,
    text,
)

CASES: list[dict[str, Any]] = DOCUMENT["path_resolution"]["cases"]
MINE = [c for c in CASES if c["platform"] == HOST_PLATFORM]
BASELINE = [c for c in MINE if c["host"] is None]
HOST = [c for c in MINE if c["host"] is not None]
ALL_HOST = [c for c in CASES if c["host"] is not None]
D012_HOST = [c for c in ALL_HOST if "/rooted" not in c["id"]]
ROOTED_WIN = [c for c in CASES if c["platform"] == "win32" and "/rooted" in c["id"]]
BY_ID = {c["id"]: c for c in CASES}
WIN = pytest.mark.skipif(sys.platform != "win32", reason="the win32 resolver branch")
SYNTHETIC_CWD = "E:\\sandbox\\child"  # the in-process port witnesses' process cwd (E: is not A:/C:)


@pytest.mark.parametrize("case", BASELINE, ids=lambda c: c["id"])
async def test_baseline_case(case: dict[str, Any]) -> None:
    assert await run_baseline_case(case) == text(case["expect"])


@pytest.mark.skipif(sys.platform != "win32", reason="host-state cases are win32 only")
@pytest.mark.parametrize("case", HOST, ids=lambda c: c["id"])
def test_host_case_in_crafted_child(case: dict[str, Any], tmp_path: Path) -> None:
    observed, expected = run_host_case(case, tmp_path)
    assert observed == expected


def test_every_case_of_this_platform_is_selected() -> None:
    """L12-D012's 55/25 plus L12-D013's 15 win32 and 4 linux rooted cases."""
    assert len(MINE) == (70 if HOST_PLATFORM == "win32" else 29)
    assert len(BASELINE) + len(HOST) == len(MINE)


# --- the binding's port, driven in-process with injected host state (both platforms) ------------


def _injected(case: dict[str, Any]) -> tuple[Any, Any, str]:
    host = case["host"]
    letter = SYNTHETIC_CWD[0] if host["drive"] == "cwd" else host["drive"]
    entry = None if host["entry"] is None else instantiate(host["entry"], SYNTHETIC_CWD)

    def inherited(device: str) -> str | None:
        return entry if device.upper() == f"{letter}:" else None

    path = instantiate(case["input"], SYNTHETIC_CWD)
    return inherited, (lambda: SYNTHETIC_CWD), path


def _port(case: dict[str, Any]) -> str:
    inherited, process_cwd, path = _injected(case)
    return filesystem.node_win32_resolve(
        [text(case["provider_cwd"]), path], inherited=inherited, process_cwd=process_cwd
    )


@pytest.mark.parametrize("case", D012_HOST, ids=lambda c: c["id"])
def test_port_reproduces_every_host_class(case: dict[str, Any]) -> None:
    assert _port(case) == instantiate(case["expect"], SYNTHETIC_CWD)


# --- L12-D013 rooted cases through the real resolver, with injected host state (win32) -----------


def _synthetic_process_cwd(case: dict[str, Any] | None) -> str:
    extended = case is not None and (case["host"] or {}).get("cwd_form") == "extended"
    return runner.EXTENDED_PREFIX + SYNTHETIC_CWD if extended else SYNTHETIC_CWD


def _resolve(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> str:
    """The real `resolve_local_path` (Pi's absolute-or-relative decision included), whose native
    process cwd is the case's synthetic cwd spelling. No inherited entry may be consulted."""

    def no_entry(device: str) -> str | None:
        raise AssertionError(f"inherited ={device} consulted for a rooted input")

    monkeypatch.setattr(filesystem, "_native_inherited_view", no_entry)
    monkeypatch.setattr(filesystem, "_native_process_cwd_view", lambda: _synthetic_process_cwd(case))
    path = instantiate(case["input"], SYNTHETIC_CWD)
    return filesystem.resolve_local_path(text(case["provider_cwd"]), path)


def _expected(case: dict[str, Any]) -> str:
    return instantiate(case["expect"], SYNTHETIC_CWD)


@WIN
@pytest.mark.parametrize("case", ROOTED_WIN, ids=lambda c: c["id"])
def test_resolver_reproduces_every_rooted_class(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _resolve(case, monkeypatch) == _expected(case)


# --- D013-C1..C5 (record 12-l12-d013-rooted.md sections 7.2 and 8.2): green baseline, then the --
# --- wrong behavior, killed by its named case while its paired positive stays green --------------


def _c1_provider_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    """WRONG: a device-less root takes the provider cwd's drive (merged Python's M1)."""
    monkeypatch.setattr(filesystem, "_native_process_cwd_view", lambda: "C:\\x")


def _ntpath_unc_root(real: Any) -> Any:
    def wrong(path: str) -> tuple[str, int, bool]:
        if len(path) > 1 and path[0] in "\\/" and path[1] in "\\/":
            device = ntpath.splitdrive(path)[0]
            return device, len(device), True
        return real(path)  # type: ignore[no-any-return]

    return wrong


def _c2_incomplete_unc(monkeypatch: pytest.MonkeyPatch) -> None:
    """WRONG: an incomplete UNC (`\\\\leaf`, `\\\\srv\\`) is kept as a UNC device (ntpath's split,
    M2)."""
    monkeypatch.setattr(filesystem, "_win32_root", _ntpath_unc_root(filesystem._win32_root))


def _c3_separator_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """WRONG: a complete UNC root keeps its raw server/share separator run (M3)."""
    real = filesystem._win32_root

    def wrong(path: str) -> tuple[str, int, bool]:
        device, root_end, is_absolute = real(path)
        if device.startswith("\\\\") and root_end:
            return path[:root_end].replace("/", "\\"), root_end, is_absolute
        return device, root_end, is_absolute

    monkeypatch.setattr(filesystem, "_win32_root", wrong)


def _c4_namespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """WRONG: the extended process cwd loses its `\\\\?\\` prefix (the wrong cwd operand)."""
    monkeypatch.setattr(filesystem, "_native_process_cwd_view", lambda: SYNTHETIC_CWD)


D013_WIN_CONTROLS = [
    (_c1_provider_drive, "win32/host/rooted/01", "win32/baseline/rooted-01"),
    (_c2_incomplete_unc, "win32/host/rooted/06", "win32/baseline/rooted-01"),
    (_c2_incomplete_unc, "win32/host/rooted/08", "win32/baseline/rooted-01"),
    (_c3_separator_run, "win32/baseline/rooted-03", "win32/baseline/rooted-02"),
    (_c4_namespace_stripped, "win32/host/rooted-extended/01", "win32/host/rooted/01"),
]


@WIN
@pytest.mark.parametrize(
    ("wrong", "killed_by", "positive"),
    D013_WIN_CONTROLS,
    ids=["C1", "C2-06", "C2-08", "C3", "C4"],
)
def test_control_d013_is_killed_by_its_named_case(
    wrong: Any, killed_by: str, positive: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, pair = BY_ID[killed_by], BY_ID[positive]
    assert _resolve(case, monkeypatch) == _expected(case)  # green baseline
    assert _resolve(pair, monkeypatch) == _expected(pair)
    with monkeypatch.context() as wrong_state:
        _resolve(case, wrong_state)  # install the case's own host state, then the wrong behavior
        wrong(wrong_state)
        assert filesystem.resolve_local_path(
            text(case["provider_cwd"]), instantiate(case["input"], SYNTHETIC_CWD)
        ) != _expected(case), f"{wrong.__name__} survived {killed_by}"
    with monkeypatch.context() as pair_state:
        _resolve(pair, pair_state)
        wrong(pair_state)
        assert filesystem.resolve_local_path(
            text(pair["provider_cwd"]), instantiate(pair["input"], SYNTHETIC_CWD)
        ) == _expected(pair), f"{wrong.__name__} also broke its positive {positive}"


@pytest.mark.skipif(sys.platform == "win32", reason="D013-C5 is the POSIX normalization control")
async def test_control_d013_c5_two_leading_slashes_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D013-C5: keeping exactly two POSIX leading slashes (merged Python's M4) is killed by
    linux/baseline/rooted-02 (`//leaf` is `/leaf`); rooted-01 (`/leaf`) stays green."""
    case, pair = BY_ID["linux/baseline/rooted-02"], BY_ID["linux/baseline/rooted-01"]
    assert await run_baseline_case(case) == text(case["expect"])
    assert await run_baseline_case(pair) == text(pair["expect"])
    monkeypatch.setattr(filesystem, "node_posix_single_root", lambda path: path)
    assert await run_baseline_case(case) != text(case["expect"])
    assert await run_baseline_case(pair) == text(pair["expect"])


def test_ported_posix_single_root() -> None:
    """The pure M4 helper on both platforms: Node's `path.posix.resolve` keeps one leading `/`."""
    single = filesystem.node_posix_single_root
    assert [single(p) for p in ("//leaf", "///leaf", "//", "/leaf", "/", "a//b")] == [
        "/leaf",
        "/leaf",
        "/",
        "/leaf",
        "/",
        "a//b",
    ]


# --- permanent controls (record section 9, runner requirement 3) ---------------------------------


def _fall_through(device: str, inherited: Any, process_cwd: Any) -> str:
    """WRONG: the first source that names the device (revision 1's rejected paraphrase)."""
    entry, cwd = inherited(device), process_cwd()
    for base in (entry, cwd):
        if base and base[:2].lower() == device.lower():
            return str(base)
    return f"{device}\\"


def _mismatch_always_root(device: str, inherited: Any, process_cwd: Any) -> str:
    """WRONG: any device mismatch selects the root (ignores the third-character conjunction)."""
    base = inherited(device) or process_cwd()
    return f"{device}\\" if base[:2].lower() != device.lower() else str(base)


def _empty_is_a_value(device: str, inherited: Any, process_cwd: Any) -> str:
    """WRONG: an empty entry is used as the base instead of selecting the process cwd."""
    entry = inherited(device)
    base = entry if entry is not None else process_cwd()
    if base[:2].lower() != device.lower() and base[2:3] == "\\":
        return f"{device}\\"
    return str(base)


@pytest.mark.parametrize(
    ("wrong", "killed_by"),
    [
        (_fall_through, "win32/host/cwd-drive/08"),
        (_mismatch_always_root, "win32/host/cwd-drive/09"),
        (_empty_is_a_value, "win32/host/cwd-drive/02"),
    ],
    ids=["fall-through", "mismatch-always-root", "empty-is-a-value"],
)
def test_control_wrong_selection_is_killed_by_its_named_case(
    wrong: Any, killed_by: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = BY_ID[killed_by]
    expected = instantiate(case["expect"], SYNTHETIC_CWD)
    assert _port(case) == expected  # the real selection passes
    monkeypatch.setattr(filesystem, "_select_drive_base", wrong)
    assert _port(case) != expected, f"{wrong.__name__} survived {killed_by}"


def test_control_pairs_discriminate() -> None:
    """The wrong-drive entry (08) and the absent entry (01) differ on the cwd's own drive, and the
    empty entry (02) equals the absent one: the pairs the controls rely on."""
    observed = {n: _port(BY_ID[f"win32/host/cwd-drive/{n}"]) for n in ("01", "02", "08")}
    assert observed["01"] == observed["02"] != observed["08"]


def test_precondition_helper_truth_table() -> None:
    """The precondition helper alone (supporting evidence; the entry-point enforcement controls
    below are the L12D012-PY-I001 witnesses)."""
    case = BY_ID["win32/baseline/12"]
    assert precondition_refusal(case, inherited=lambda _: None, process_cwd="E:\\w") is None
    for entry in ("A:\\deep", ""):  # present -- including present but EMPTY -- refuses
        refusal = precondition_refusal(
            case, inherited=lambda d, v=entry: v if d == "A:" else None, process_cwd="E:\\w"
        )
        assert refusal == "inherited =A: entry is present"
    assert precondition_refusal(case, inherited=lambda _: None, process_cwd="A:\\w") is not None
    host = BY_ID["win32/host/a-drive/01"]
    assert precondition_refusal(host, inherited=lambda _: None, process_cwd="C:\\w") is not None


# --- entry-point enforcement controls (L12D012-PY-I001) ------------------------------------------
# The ACTUAL runner entry points run under scripted host state (the inherited-entry lookup and the
# process cwd are module-level in the runner; a host case's child cwd is its sandbox argument). No
# case path and no scripted cwd is ever created, accessed or used as a real cwd: a refused run stops
# before the provider or the child, and an admitted host run stops AT the child launch (a spy).


class _Reached(Exception):
    """The admitted host run reached the child launch (the spy stops it there)."""


class _Observed:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.provider: list[str] = []
        self.child: list[list[str]] = []
        real = filesystem.LocalFileSystem.absolute_path

        async def absolute_path(fs: Any, path: str, signal: Any = None) -> Any:
            self.provider.append(path)
            return await real(fs, path, signal)

        def run(args: list[str], **_: Any) -> Any:
            self.child.append(args)
            raise _Reached

        monkeypatch.setattr(filesystem.LocalFileSystem, "absolute_path", absolute_path)
        monkeypatch.setattr(subprocess_module, "run", run)


def _script(monkeypatch: pytest.MonkeyPatch, module: Any, entry: str | None, cwd: str) -> None:
    monkeypatch.setattr(module, "real_inherited", lambda d: entry if d == "A:" else None)
    monkeypatch.setattr(module, "process_cwd", lambda: cwd)


async def _baseline_refused(module: Any, observed: _Observed) -> bool:
    """The refusal witness: the entry point refuses, and the provider is never invoked."""
    try:
        await module.run_baseline_case(BY_ID["win32/baseline/12"])
    except AssertionError as error:
        return str(error).startswith("PRECONDITION REFUSED") and observed.provider == []
    return False


def _host_refused(module: Any, observed: _Observed, case_id: str, sandbox: str) -> bool:
    try:
        module.run_host_case(BY_ID[case_id], Path(sandbox))
    except _Reached:
        return False
    except AssertionError as error:
        return str(error).startswith("PRECONDITION REFUSED") and observed.child == []
    return False


FORBIDDEN_BASELINE = [
    ("", "E:\\w"),
    ("A:\\deep", "E:\\w"),
    (None, "A:\\w"),
    (None, "X:\\w"),
    (None, "C:\\w"),
]
FORBIDDEN_HOST = [
    ("win32/host/a-drive/01", "C:\\never-created"),
    ("win32/host/a-drive/08", "A:\\never-created"),
    ("win32/host/cwd-drive/01", "C:\\never-created"),
]


@pytest.mark.parametrize(("entry", "cwd"), FORBIDDEN_BASELINE)
async def test_baseline_entry_refuses_before_the_provider(
    entry: str | None, cwd: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Present (including present-but-EMPTY) =A:, or a process cwd on A:/X:/C:: refused, and the
    real provider is never reached."""
    observed = _Observed(monkeypatch)
    _script(monkeypatch, runner, entry, cwd)
    assert await _baseline_refused(runner, observed)


async def test_baseline_entry_admits_and_reaches_the_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: admissible state reaches the real `absolute_path` exactly once."""
    observed = _Observed(monkeypatch)
    _script(monkeypatch, runner, None, "E:\\w")
    value = await runner.run_baseline_case(BY_ID["win32/baseline/12"])
    assert observed.provider == ["a::/b"]
    if HOST_PLATFORM == "win32":
        assert value == text(BY_ID["win32/baseline/12"]["expect"])


@pytest.mark.parametrize(("case_id", "sandbox"), FORBIDDEN_HOST)
def test_host_entry_refuses_before_the_child(
    case_id: str, sandbox: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = _Observed(monkeypatch)
    assert _host_refused(runner, observed, case_id, sandbox)


def test_host_entry_admits_and_reaches_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: an admissible child cwd reaches the child launch (stopped by the spy)."""
    observed = _Observed(monkeypatch)
    with pytest.raises(_Reached):
        runner.run_host_case(BY_ID["win32/host/a-drive/01"], Path("E:\\never-created"))
    assert len(observed.child) == 1


def _scripted_child(monkeypatch: pytest.MonkeyPatch, reported: str) -> list[dict[str, Any]]:
    """A child launch that only REPORTS `reported` as its cwd (nothing is launched or accessed)."""
    calls: list[dict[str, Any]] = []

    def run(args: list[str], **kwargs: Any) -> Any:
        calls.append(kwargs)
        out = json.dumps({"cwd": reported, "path": "", "value": "V"})
        return subprocess_module.CompletedProcess(args, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess_module, "run", run)
    return calls


@pytest.mark.parametrize(
    "reported",
    ["E:\\never-created", "\\\\?\\e:\\never-created", "\\\\.\\E:\\never-created"],
)
def test_host_entry_refuses_a_child_without_the_extended_spelling(
    reported: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L12-D013: an extended case whose child does not observe exactly `\\\\?\\` + the sandbox
    path REFUSES (no case-folding, no other namespace)."""
    calls = _scripted_child(monkeypatch, reported)
    with pytest.raises(AssertionError, match=r"^PRECONDITION REFUSED \(cwd form\)"):
        runner.run_host_case(BY_ID["win32/host/rooted-extended/01"], Path("E:\\never-created"))
    assert [c["cwd"] for c in calls] == ["\\\\?\\E:\\never-created"]


def test_host_entry_admits_the_exact_extended_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive: the exact spelling is admitted, and templates fill from the ORDINARY path."""
    _scripted_child(monkeypatch, "\\\\?\\E:\\never-created")
    observed, expected = runner.run_host_case(
        BY_ID["win32/host/rooted-extended/01"], Path("E:\\never-created")
    )
    assert (observed, expected) == ("V", "\\\\?\\E:\\leaf")


def _without_enforcement(marker: str) -> Any:
    """The runner with ONE entry point's refusal branch removed, compiled in memory (nothing is
    written); `marker` is that branch's exact source."""
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert source.count(marker) == 1
    mutated = source.replace(marker, marker.replace("if refusal is not None:", "if False:"))
    module = types.ModuleType("path_resolution_runner_without_enforcement")
    module.__file__ = runner.__file__
    exec(compile(mutated, runner.__file__, "exec"), module.__dict__)
    return module


_BASELINE_BRANCH = (
    '    if refusal is not None:\n        raise AssertionError(f"PRECONDITION REFUSED: {refusal}")'
)
_HOST_BRANCH = (
    "    if refusal is not None:\n"
    '        raise AssertionError(f"PRECONDITION REFUSED (child cwd): {refusal}")'
)


@pytest.mark.parametrize(("entry", "cwd"), FORBIDDEN_BASELINE[:1] + FORBIDDEN_BASELINE[2:3])
async def test_control_baseline_enforcement_omission_is_killed(
    entry: str | None, cwd: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known-bad (the reviewer's mutant): the baseline entry point without its refusal branch runs
    the forbidden case through the provider, so the refusal witness above FAILS on it."""
    mutant = _without_enforcement(_BASELINE_BRANCH)
    observed = _Observed(monkeypatch)
    _script(monkeypatch, mutant, entry, cwd)
    assert not await _baseline_refused(mutant, observed)
    assert observed.provider == ["a::/b"]


@pytest.mark.parametrize(("case_id", "sandbox"), FORBIDDEN_HOST[:1])
def test_control_host_enforcement_omission_is_killed(
    case_id: str, sandbox: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = _without_enforcement(_HOST_BRANCH)
    observed = _Observed(monkeypatch)
    assert not _host_refused(mutant, observed, case_id, sandbox)
    assert len(observed.child) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="the per-drive entry exists only on win32")
def test_host_runner_child_environment_carries_exactly_the_case_entry(tmp_path: Path) -> None:
    from .path_resolution_runner import host_child_env

    env = host_child_env(BY_ID["win32/host/a-drive/08"], str(tmp_path))
    assert env["=A:"] == "X:\\other"
    assert [k for k in env if k.startswith("=")] == ["=A:"]
    assert "=A:" not in os.environ
