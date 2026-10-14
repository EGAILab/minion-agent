"""L12-D011 negative controls (spec/execution.md section 21; record section 3.4), with the credit rules of
workflow section 9.7 enforced mechanically (L12D011-I002):

1. BASELINE: the unmutated copy runs first with the same command and environment; it must exit 0, report no
   FAILED/ERROR line, and report every intended witness node of every control as PASSED. Otherwise the whole run
   is INVALID and no mutant is credited.
2. Each mutant's edits must match a unique anchor, and every mutated file must COMPILE (in memory) before any test.
3. A mutant is KILLED only when pytest exits 1 (tests failed), no ERROR line is reported, and at least one
   intended witness node FAILED with its DISCRIMINATING reason (a per-witness regular expression). Exit codes
   2/3/4/5, ERROR lines, a missing baseline node, a timeout or a non-matching reason are INVALID or NOT KILLED.

    python scripts/l12_d011_negative_controls.py --work <new dir inside the project root> [NAME ...]
    python scripts/l12_d011_negative_controls.py --list
    python scripts/l12_d011_negative_controls.py --self-test     (the credit rules, on synthetic results)

Containment (Owner rule 2026-10-10): `--work` must be a NEW directory strictly inside the project root
(MINION_PROJECT_ROOT, default the Windows checkout root; a container passes its private /tmp root). Every
target is proven -- lexically inside, no link or reparse point on any existing ancestor, real path inside --
immediately before each mkdir/copy/write. Nothing is ever deleted. Needs MINION_SEARCH_ENGINE_ARTIFACTS for the
tool cases. Set `E5_BARE_PYTEST=1` in a container without the pytest-cov configuration.
"""

# ruff: noqa: E501 -- the fault table quotes source lines verbatim as unique anchors

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = Path(
    os.environ.get("MINION_PROJECT_ROOT", "E:/AI/Projects/OpenMinds/Minions/Minion-Agent")
).resolve()
EXE = Path("minion_agent/execution")
SUB, NUL = EXE / "subprocess.py", EXE / "spawn_nul.py"
SEARCH = Path("minion_agent/tools/builtin/_search.py")
BASH = Path("minion_agent/tools/builtin/bash.py")
UNIT = "tests/execution/test_spawn_nul.py"
CANON = "tests/conformance/test_spawn_nul_conformance.py"
TESTS = [UNIT, CANON]


def canon(case: str) -> str:
    return f"{CANON}::test_spawn_case[{case}]"


@dataclass(frozen=True)
class Control:
    edits: list[tuple[Path, str, str]]
    witnesses: dict[
        str, str
    ]  # intended pytest node id -> the discriminating failure reason (regex)
    expected: str


_CHECK = "        if rejected is not None:\n            return Err(rejected)\n"
_ORDER = f"{UNIT}::test_env_check_order_both_platforms"
CONTROLS: dict[str, Control] = {
    "check-after-native-spawn": Control(
        [(SUB, _CHECK, "")],
        {f"{UNIT}::test_no_native_spawn_after_rejection[argv0-options0]": r"native spawn reached",
         canon("name/file"): r"ValueError: embedded null"},
        "the native spawn is reached (the spy fires; Python's ValueError leaks)",
    ),
    "baseline-first-posix-order": Control(
        [(NUL, "    owned_set = set(owned)\n", "    return names\n    owned_set = set(owned)\n")],
        {_ORDER: r"assert \['A', 'Z'"},
        "POSIX order follows baseline presentation, not overlay-first",
    ),
    "sort-all-posix-order": Control(
        [(NUL, "    owned_set = set(owned)\n", "    return sorted(names, key=_utf16_key)\n    owned_set = set(owned)\n")],
        {_ORDER: r"assert \['01', '10'"},
        "POSIX order sorts every name",
    ),
    "naive-integer-index": Control(
        [(NUL, '_INDEX = re.compile(r"0|[1-9][0-9]*")\n', '_INDEX = re.compile(r"[0-9]+")\n')],
        {_ORDER: r"assert \['01', '2'"},
        "01 is taken for an integer index",
    ),
    "unbounded-index-conversion": Control(
        [(NUL, "    return len(name) <= 10 and _INDEX.fullmatch(name) is not None and int(name) <= _MAX_INDEX\n",
          "    return _INDEX.fullmatch(name) is not None and int(name) <= _MAX_INDEX\n")],
        {f"{UNIT}::test_index_domain_neighborhood_and_long_names": r"LONG NAME CONVERTED",
         f"{UNIT}::test_long_numeric_env_name_is_an_ordinary_name_at_the_provider": r"LONG NAME CONVERTED"},
        "a long numeric name is converted and raises (L12D011-I001)",
    ),
    "superseded-baseline-value": Control(
        [(SUB, "            effective_env,\n            [name for name, _ in baseline],\n",
          "            {**(opts.env or {}), **dict(baseline)},\n            [name for name, _ in baseline],\n")],
        {canon("env/superseded-baseline-value-control"): r"a control must reach the OS"},
        "the superseded baseline value is checked instead of the effective one",
    ),
    "raw-cwd-not-resolved": Control(
        [(SUB, "            argv,\n            resolved_cwd,\n", "            argv,\n            opts.cwd or self.cwd,\n")],
        {canon("cwd/file-url-percent-00"): r"ValueError: embedded null",
         canon("cwd/literal-relative"): r"ived 'd\\\\x00'"},
        "the raw spelling is checked: %00 is missed, a relative cwd is reported unresolved",
    ),
    "double-url-decode": Control(
        [(SUB, "        resolved_cwd = resolve_local_path(self.cwd, opts.cwd) if opts.cwd is not None else self.cwd\n",
          "        from urllib.parse import unquote\n        resolved_cwd = unquote(resolve_local_path(self.cwd, opts.cwd)) if opts.cwd is not None else self.cwd\n")],
        {canon("cwd/file-url-percent-2500-control"): r"a control must reach the OS"},
        "%2500 is decoded twice into a NUL",
    ),
    "default-cwd-unchecked": Control(
        [(NUL, "    if _NUL in cwd:\n", "    if _NUL in cwd and cwd != cwd:\n")],
        {canon("cwd/provider-default-nul"): r"ValueError: embedded null"},
        "a NUL in the provider default cwd reaches the native spawn",
    ),
    "cwd-before-args": Control(
        [(NUL, "    rejected = argv_rejection(argv)\n    if rejected is not None:\n        return rejected\n    if _NUL in cwd:\n        return _rejection(diagnostic(\"options.cwd\", cwd, _CWD_REASON))\n",
          "    if _NUL in cwd:\n        return _rejection(diagnostic(\"options.cwd\", cwd, _CWD_REASON))\n    rejected = argv_rejection(argv)\n    if rejected is not None:\n        return rejected\n")],
        {canon("precedence/args-before-cwd"): r"assert \"The property"},
        "the cwd is reported before an invalid argument",
    ),
    "cut-at-or-above-128": Control(
        [(NUL, "    if len(received) > _CUT:\n", "    if len(received) >= _CUT:\n")],
        {canon("len/inspected-128"): r"x00'\.\.\.\" =="},
        "an inspected value of exactly 128 units is cut",
    ),
    "quote-selection": Control(
        [(NUL, "        if 34 not in piece:\n            quote = 34\n", "        if 34 not in piece:\n            quote = 39\n")],
        {canon("fmt/single-quote"): r"'it\\\\'s"},
        "a value with ' is single-quoted and escaped",
    ),
    "tool-string-matching": Control(
        [(SEARCH, "        if spawned.error.nul_input:\n", "        if spawned.error.message.startswith(\"The \"):\n")],
        {f"{UNIT}::test_engine_keeps_its_prefix_for_a_non_nul_failure": r"PREFIX LOST"},
        "an OS message starting with 'The ' loses its prefix",
    ),
    "tool-checks-projected-argv": Control(
        [(SEARCH, "    rejected = argv_rejection(argv)\n", "    rejected = argv_rejection([scalar_command(arg) for arg in argv])\n")],
        {f"{UNIT}::test_engine_diagnostic_shows_the_unprojected_value": r"PROJECTED DIAGNOSTIC"},
        "the diagnostic shows U+FFFD for a lone surrogate",
    ),
    "bash-checks-stdin-command": Control(
        [(BASH, "            [config.shell, *config.args] if via_stdin else [config.shell, *config.args, command]\n",
          "            [config.shell, *config.args, command]\n")],
        {f"{UNIT}::test_bash_stdin_transport_command_is_not_argv": r"assert \"The argument.*== 'Failed to st"},
        "a stdin-transported command is rejected as argv",
    ),
}  # fmt: skip

_LINE = re.compile(r"^(PASSED|FAILED|ERROR) (\S+)(?: - (.*))?$")


@dataclass
class Outcome:
    returncode: int | None  # None: timeout
    passed: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    summary: str = ""


def parse(returncode: int | None, stdout: str) -> Outcome:
    out = Outcome(returncode)
    lines = stdout.splitlines()
    for line in lines:
        match = _LINE.match(line)
        if match:
            kind, node, reason = match.group(1), match.group(2), match.group(3) or ""
            if kind == "PASSED":
                out.passed.add(node)
            elif kind == "FAILED":
                out.failed[node] = reason
            else:
                out.errors[node] = reason
    out.summary = lines[-1] if lines else ""
    return out


def baseline_verdict(base: Outcome, controls: dict[str, Control]) -> str | None:
    """None when the baseline is green and every intended witness PASSED; else the INVALID reason."""
    if base.returncode != 0:
        return f"baseline exit {base.returncode}"
    if base.failed or base.errors:
        return f"baseline not green: {sorted(base.failed)[:3] + sorted(base.errors)[:3]}"
    missing = sorted({n for c in controls.values() for n in c.witnesses} - base.passed)
    return f"intended witnesses not PASSED in the baseline: {missing}" if missing else None


def credit(control: Control, base_passed: set[str], result: Outcome) -> dict[str, object]:
    """The section 9.7 credit for one mutant run (pure; exercised by --self-test)."""
    missing = sorted(n for n in control.witnesses if n not in base_passed)
    if missing:
        return {"verdict": "INVALID", "why": f"witness not PASSED in the baseline: {missing}"}
    if result.returncode is None:
        return {"verdict": "INVALID", "why": "timeout"}
    if result.returncode == 0:
        return {"verdict": "NOT KILLED", "why": "the mutant survived (exit 0)"}
    if result.returncode != 1:
        return {
            "verdict": "INVALID",
            "why": f"pytest exit {result.returncode} (not a test failure)",
        }
    if result.errors:
        return {
            "verdict": "INVALID",
            "why": f"collection/setup errors: {sorted(result.errors)[:3]}",
        }
    observed = {n: result.failed[n] for n in control.witnesses if n in result.failed}
    matched = {n: r for n, r in observed.items() if re.search(control.witnesses[n], r)}
    if not matched:
        return {
            "verdict": "NOT KILLED",
            "why": "no intended witness failed for its discriminating reason",
            "observed": observed,
        }
    return {"verdict": "KILLED", "observed": matched}


def _checked(target: Path) -> Path:
    """Lexical, no-link/no-reparse ancestry, and real-path proofs, immediately before a mutation."""
    target = Path(os.path.abspath(target))
    if ".." in target.parts or target == PROJECT or not target.is_relative_to(PROJECT):
        raise RuntimeError(f"outside the project root: {target}")
    for entry in [*reversed(target.parents), target]:
        try:
            meta = entry.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(meta.st_mode) or getattr(meta, "st_file_attributes", 0) & 0x400:
            raise RuntimeError(f"link or reparse point in {entry}")
    if not Path(os.path.realpath(target)).is_relative_to(PROJECT.resolve()):
        raise RuntimeError(f"really outside the project root: {target}")
    return target


def _command() -> list[str]:
    base = [
        sys.executable,
        "-m",
        "pytest",
        *TESTS,
        "-q",
        "-rA",
        "-p",
        "no:cacheprovider",
        "--no-header",
    ]
    return base + (["-o", "addopts="] if os.environ.get("E5_BARE_PYTEST") else ["--no-cov"])


def _copy_src(dest: Path) -> None:
    for directory, dirs, files in os.walk(ROOT / "src"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel = Path(directory).relative_to(ROOT / "src")
        _checked(dest / rel).mkdir()
        for file in files:
            shutil.copyfile(Path(directory) / file, _checked(dest / rel / file))


def _pytest(src: Path) -> Outcome:
    env = {**os.environ, "PYTHONPATH": str(src), "COLUMNS": "4000", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        done = subprocess.run(
            _command(), cwd=ROOT, env=env, capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        return Outcome(None)
    return parse(done.returncode, done.stdout)


def run_mutant(name: str, work: Path, base_passed: set[str]) -> dict[str, object]:
    control = CONTROLS[name]
    src = work / name / "src"
    _checked(work / name).mkdir()
    _copy_src(src)
    texts: dict[Path, str] = {}
    for target, anchor, replacement in control.edits:
        path = src / target
        text = texts.get(path) or path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        if text.count(anchor) != 1:
            return {
                "name": name,
                "verdict": "INVALID",
                "why": f"anchor matched {text.count(anchor)} times in {target}",
            }
        texts[path] = text.replace(anchor, replacement)
    for path, text in texts.items():
        try:
            compile(text, str(path), "exec")
        except SyntaxError as error:
            return {"name": name, "verdict": "INVALID", "why": f"mutant does not compile: {error}"}
        _checked(path).write_bytes(text.encode("utf-8"))
    result = _pytest(src)
    return {
        "name": name,
        "expected": control.expected,
        **credit(control, base_passed, result),
        "exit": result.returncode,
        "errors": len(result.errors),
        "summary": result.summary,
    }


def self_test() -> int:
    failed = 0

    def check(ok: bool, text: str) -> None:
        nonlocal failed
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {text}")

    node = f"{UNIT}::test_engine_keeps_its_prefix_for_a_non_nul_failure"
    control = CONTROLS["tool-string-matching"]
    good = Outcome(1, failed={node: "AssertionError: PREFIX LOST: The fake"})
    check(credit(control, {node}, good)["verdict"] == "KILLED", "a valid intended kill is credited")
    counterfeit = Outcome(1, failed={node: "NameError: counterfeit setup failure"})
    check(
        credit(control, {node}, counterfeit)["verdict"] == "NOT KILLED",
        "a counterfeit failure reason is not credited",
    )
    for code in (2, 3, 4, 5):
        check(
            credit(control, {node}, Outcome(code, failed={node: "AssertionError: PREFIX LOST: x"}))[
                "verdict"
            ]
            == "INVALID",
            f"pytest exit {code} is INVALID even with a matching FAILED line",
        )
    check(
        credit(
            control,
            {node},
            Outcome(
                1,
                failed={node: "AssertionError: PREFIX LOST"},
                errors={"tests/x.py": "ImportError"},
            ),
        )["verdict"]
        == "INVALID",
        "an ERROR line makes the run INVALID",
    )
    check(
        credit(control, set(), good)["verdict"] == "INVALID",
        "a witness missing from the baseline is INVALID",
    )
    check(credit(control, {node}, Outcome(None))["verdict"] == "INVALID", "a timeout is INVALID")
    check(
        credit(control, {node}, Outcome(0, passed={node}))["verdict"] == "NOT KILLED",
        "a surviving mutant is NOT KILLED",
    )
    every = {n for c in CONTROLS.values() for n in c.witnesses}
    check(
        baseline_verdict(Outcome(0, passed=every), CONTROLS) is None,
        "a green baseline with every witness PASSED is accepted",
    )
    check(
        baseline_verdict(Outcome(1, passed=every, failed={node: "x"}), CONTROLS) is not None,
        "a red baseline is INVALID",
    )
    check(
        baseline_verdict(Outcome(0, passed=every - {node}), CONTROLS) is not None,
        "a filtered/missing witness makes the baseline INVALID",
    )
    parsed = parse(
        1,
        f"PASSED {UNIT}::a\nFAILED {node} - AssertionError: PREFIX LOST: y\nERROR {CANON} - ImportError: z\n1 failed",
    )
    check(
        parsed.passed == {f"{UNIT}::a"} and node in parsed.failed and CANON in parsed.errors,
        "the -rA report is parsed",
    )
    print("DRIVER SELF-TEST " + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--list"]:
        print(
            json.dumps(
                {
                    n: {"witnesses": c.witnesses, "expected": c.expected}
                    for n, c in CONTROLS.items()
                },
                indent=1,
            )
        )
        return 0
    if argv[:1] == ["--self-test"]:
        return self_test()
    if argv[:1] != ["--work"] or len(argv) < 2:
        print("usage: --work <new dir inside the project root> [NAME ...]", file=sys.stderr)
        return 2
    work = _checked(Path(argv[1]))
    if work.exists():
        print(f"--work must be NEW: {work}", file=sys.stderr)
        return 2
    _checked(work).mkdir(parents=True)
    names = argv[2:] or list(CONTROLS)
    _checked(work / "baseline").mkdir()
    _copy_src(work / "baseline" / "src")
    base = _pytest(work / "baseline" / "src")
    invalid = baseline_verdict(base, {n: CONTROLS[n] for n in names})
    report: dict[str, object] = {
        "os": "win32" if os.name == "nt" else "linux",
        "baseline": {
            "exit": base.returncode,
            "passed": len(base.passed),
            "failed": sorted(base.failed),
            "errors": sorted(base.errors),
            "summary": base.summary,
            "verdict": invalid or "GREEN",
        },
    }
    if invalid:
        report["results"] = []
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return 1
    report["results"] = [run_mutant(name, work, base.passed) for name in names]
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if all(r["verdict"] == "KILLED" for r in report["results"]) else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
