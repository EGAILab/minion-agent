"""L12-D011 negative controls (spec/execution.md section 21; record section 3.4), with the credit rules of
workflow section 9.7 enforced mechanically on OBSERVED evidence (L12D011-I002; convergence episode
CE-L12D011-01, revision 1, assurance record `12-l12-d011-convergence-ce01.md` section 7):

1. W3 FIRST: the evidence fixture (`l12_d011_evidence_fixture.py`) runs with the same pytest command and the
   evidence reporter (`l12_d011_evidence.py`); every fixture node must classify as declared (`FIXTURE`). This
   is also the executable confirmation of the native NUL origin on this platform. Otherwise every control is
   INVALID.
2. BASELINE: the unmutated copy must exit 0, report no FAILED/ERROR line, report every intended witness node
   PASSED, and carry the reporter's START/END lines with this run's token.
3. Each mutant's edits must match a unique anchor and every mutated file must COMPILE (in memory) first.
4. A mutant is KILLED only when pytest exits 1, no ERROR line is reported, the evidence is complete (START/END
   with the token, END count = records, no duplicate node/phase, a record for every FAILED intended node and a
   FAILED line for every recorded intended node), NO intended witness has a forbidden observation, and at least
   one intended witness is GENUINE:
   - ASSERTION witness: the exception IS builtins.AssertionError, raised in the call phase, innermost frame in
     the declared file and function, at a line inside the declared assertion statement; reason `re.match`ed
     on the first line of str(exception);
   - NATIVE_NUL witness: the exception IS builtins.ValueError, call phase, innermost frame in the stdlib
     `subprocess` module's Popen._execute_child / Popen._posix_spawn, the seam frame LocalSubprocess.spawn of
     the run's source on the stack; first line fully matching `embedded null (character|byte)`.
   A wrong kind, phase or origin is FORBIDDEN (the run is INVALID: the intended failure did not fire); the right
   kind and origin with another reason is a MISMATCH (not killed). The rendered `-rA` reason never decides.

    python scripts/l12_d011_negative_controls.py --work <new dir inside the project root> [NAME ...]
    python scripts/l12_d011_negative_controls.py --list
    python scripts/l12_d011_negative_controls.py --self-test  (W1/W2, meta-controls D1-D10, then W3)

Trust boundary: cooperating test components. A test that deliberately rebinds builtins, forges a traceback or
prints token-bearing records is out of scope; tests run under output capture, and records carry a fresh token.

Containment (Owner rule 2026-10-10): `--work` must be a NEW directory strictly inside the project root
(MINION_PROJECT_ROOT, default the Windows checkout root; a container passes its private /tmp root). Every
target is proven -- lexically inside, no link or reparse point on any existing ancestor, real path inside --
immediately before each mkdir/copy/write. Nothing is ever deleted. The self-test and W3 write nothing. Needs
MINION_SEARCH_ENGINE_ARTIFACTS for the tool cases. Set `E5_BARE_PYTEST=1` in a container without the pytest-cov
configuration.
"""

# ruff: noqa: E501 -- the fault table quotes source lines verbatim as unique anchors

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
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
FIXTURE_FILE = "scripts/l12_d011_evidence_fixture.py"
PLUGIN = "l12_d011_evidence"
EVIDENCE = "L12D011-EVIDENCE"
TOKEN_ENV = "L12D011_EVIDENCE_TOKEN"
NATIVE_FRAMES = {"Popen._execute_child", "Popen._posix_spawn"}
SEAM_QUALNAME = "LocalSubprocess.spawn"

GENUINE, MISMATCH, FORBIDDEN = "GENUINE", "MISMATCH", "FORBIDDEN"


@dataclass(frozen=True)
class Assertion:
    """The intended failure is THIS assertion statement (anchor: unique within `qualname` in `file`)."""

    file: str
    qualname: str
    anchor: str
    reason: str


@dataclass(frozen=True)
class NativeNul:
    """The intended failure is the native call's own NUL rejection, reached through the seam."""

    reason: str = r"embedded null (character|byte)"


Witness = Assertion | NativeNul


def canon(case: str) -> str:
    return f"{CANON}::test_spawn_case[{case}]"


_SPAWN_CASE = "test_spawn_case"
_MESSAGE = 'assert observed["message"] == text(expected["message"])'
_REACH = 'assert not observed["nul_input"], "a control must reach the OS, not the NUL check"'


def _unit(qualname: str, anchor: str, reason: str) -> Assertion:
    return Assertion(UNIT, qualname, anchor, reason)


def _canon(anchor: str, reason: str) -> Assertion:
    return Assertion(CANON, _SPAWN_CASE, anchor, reason)


@dataclass(frozen=True)
class Control:
    edits: list[tuple[Path, str, str]]
    witnesses: dict[str, Witness]  # intended pytest node id -> its declared failure
    expected: str


_CHECK = "        if rejected is not None:\n            return Err(rejected)\n"
_ORDER = f"{UNIT}::test_env_check_order_both_platforms"
_ORDER_STMT = "assert env_check_order(baseline, overlay, Platform.POSIX) == ["
_LONG = 'raise AssertionError(f"LONG NAME CONVERTED: {error}") from error'
CONTROLS: dict[str, Control] = {
    "check-after-native-spawn": Control(
        [(SUB, _CHECK, "")],
        {f"{UNIT}::test_no_native_spawn_after_rejection[argv0-options0]": _unit("test_no_native_spawn_after_rejection.<locals>.spy", 'raise AssertionError("native spawn reached")', r"native spawn reached"),
         canon("name/file"): NativeNul()},
        "the native spawn is reached (the spy fires; Python's ValueError leaks)",
    ),
    "baseline-first-posix-order": Control(
        [(NUL, "    owned_set = set(owned)\n", "    return names\n    owned_set = set(owned)\n")],
        {_ORDER: _unit("test_env_check_order_both_platforms", _ORDER_STMT, r"assert \['A', 'Z'")},
        "POSIX order follows baseline presentation, not overlay-first",
    ),
    "sort-all-posix-order": Control(
        [(NUL, "    owned_set = set(owned)\n", "    return sorted(names, key=_utf16_key)\n    owned_set = set(owned)\n")],
        {_ORDER: _unit("test_env_check_order_both_platforms", _ORDER_STMT, r"assert \['01', '10'")},
        "POSIX order sorts every name",
    ),
    "naive-integer-index": Control(
        [(NUL, '_INDEX = re.compile(r"0|[1-9][0-9]*")\n', '_INDEX = re.compile(r"[0-9]+")\n')],
        {_ORDER: _unit("test_env_check_order_both_platforms", _ORDER_STMT, r"assert \['01', '2'")},
        "01 is taken for an integer index",
    ),
    "unbounded-index-conversion": Control(
        [(NUL, "    return len(name) <= 10 and _INDEX.fullmatch(name) is not None and int(name) <= _MAX_INDEX\n",
          "    return _INDEX.fullmatch(name) is not None and int(name) <= _MAX_INDEX\n")],
        {f"{UNIT}::test_index_domain_neighborhood_and_long_names": _unit("test_index_domain_neighborhood_and_long_names", _LONG, r"LONG NAME CONVERTED"),
         f"{UNIT}::test_long_numeric_env_name_is_an_ordinary_name_at_the_provider": _unit("test_long_numeric_env_name_is_an_ordinary_name_at_the_provider", _LONG, r"LONG NAME CONVERTED")},
        "a long numeric name is converted and raises (L12D011-I001)",
    ),
    "superseded-baseline-value": Control(
        [(SUB, "            effective_env,\n            [name for name, _ in baseline],\n",
          "            {**(opts.env or {}), **dict(baseline)},\n            [name for name, _ in baseline],\n")],
        {canon("env/superseded-baseline-value-control"): _canon(_REACH, r"a control must reach the OS")},
        "the superseded baseline value is checked instead of the effective one",
    ),
    "raw-cwd-not-resolved": Control(
        [(SUB, "            argv,\n            resolved_cwd,\n", "            argv,\n            opts.cwd or self.cwd,\n")],
        {canon("cwd/file-url-percent-00"): NativeNul(),
         canon("cwd/literal-relative"): _canon(_MESSAGE, r"assert .*ived 'd\\\\x00'")},
        "the raw spelling is checked: %00 is missed, a relative cwd is reported unresolved",
    ),
    "double-url-decode": Control(
        [(SUB, "        resolved_cwd = resolve_local_path(self.cwd, opts.cwd) if opts.cwd is not None else self.cwd\n",
          "        from urllib.parse import unquote\n        resolved_cwd = unquote(resolve_local_path(self.cwd, opts.cwd)) if opts.cwd is not None else self.cwd\n")],
        {canon("cwd/file-url-percent-2500-control"): _canon(_REACH, r"a control must reach the OS")},
        "%2500 is decoded twice into a NUL",
    ),
    "default-cwd-unchecked": Control(
        [(NUL, "    if _NUL in cwd:\n", "    if _NUL in cwd and cwd != cwd:\n")],
        {canon("cwd/provider-default-nul"): NativeNul()},
        "a NUL in the provider default cwd reaches the native spawn",
    ),
    "cwd-before-args": Control(
        [(NUL, "    rejected = argv_rejection(argv)\n    if rejected is not None:\n        return rejected\n    if _NUL in cwd:\n        return _rejection(diagnostic(\"options.cwd\", cwd, _CWD_REASON))\n",
          "    if _NUL in cwd:\n        return _rejection(diagnostic(\"options.cwd\", cwd, _CWD_REASON))\n    rejected = argv_rejection(argv)\n    if rejected is not None:\n        return rejected\n")],
        {canon("precedence/args-before-cwd"): _canon(_MESSAGE, r"assert \"The property")},
        "the cwd is reported before an invalid argument",
    ),
    "cut-at-or-above-128": Control(
        [(NUL, "    if len(received) > _CUT:\n", "    if len(received) >= _CUT:\n")],
        {canon("len/inspected-128"): _canon(_MESSAGE, r"assert .*x00'\.\.\.\" ==")},
        "an inspected value of exactly 128 units is cut",
    ),
    "quote-selection": Control(
        [(NUL, "        if 34 not in piece:\n            quote = 34\n", "        if 34 not in piece:\n            quote = 39\n")],
        {canon("fmt/single-quote"): _canon(_MESSAGE, r"assert .*'it\\\\'s")},
        "a value with ' is single-quoted and escaped",
    ),
    "tool-string-matching": Control(
        [(SEARCH, "        if spawned.error.nul_input:\n", "        if spawned.error.message.startswith(\"The \"):\n")],
        {f"{UNIT}::test_engine_keeps_its_prefix_for_a_non_nul_failure": _unit("test_engine_keeps_its_prefix_for_a_non_nul_failure", 'assert str(caught.value) == "Failed to run fd: The fake"', r"PREFIX LOST")},
        "an OS message starting with 'The ' loses its prefix",
    ),
    "tool-checks-projected-argv": Control(
        [(SEARCH, "    rejected = argv_rejection(argv)\n", "    rejected = argv_rejection([scalar_command(arg) for arg in argv])\n")],
        {f"{UNIT}::test_engine_diagnostic_shows_the_unprojected_value": _unit("test_engine_diagnostic_shows_the_unprojected_value", "assert str(caught.value).endswith(", r"PROJECTED DIAGNOSTIC")},
        "the diagnostic shows U+FFFD for a lone surrogate",
    ),
    "bash-checks-stdin-command": Control(
        [(BASH, "            [config.shell, *config.args] if via_stdin else [config.shell, *config.args, command]\n",
          "            [config.shell, *config.args, command]\n")],
        {f"{UNIT}::test_bash_stdin_transport_command_is_not_argv": _unit("test_bash_stdin_transport_command_is_not_argv", 'assert await _run_bash(fake, tmp_path, "echo a" + N) == "Failed to start the shell wsl-bash"', r"assert \"The argument.*== 'Failed to st")},
        "a stdin-transported command is rejected as argv",
    ),
}  # fmt: skip


def _fx(qualname: str, anchor: str, reason: str) -> Assertion:
    return Assertion(FIXTURE_FILE, qualname, anchor, reason)


_MARK = r"FIXTURE MARK"
# W3: fixture node -> (declaration, the classification it must receive). Pairs C001/C002 counterfeits with
# genuine observations; the genuine native node confirms the native origin on the running platform.
FIXTURE: dict[str, tuple[Witness, str]] = {
    "test_genuine_assertion": (_fx("test_genuine_assertion", 'assert "a" == "b", "FIXTURE MARK: genuine"', _MARK), GENUINE),
    "test_marker_not_at_start": (_fx("test_marker_not_at_start", 'assert "a" == "b", "wrapper: FIXTURE MARK"', _MARK), MISMATCH),
    "test_wrong_statement": (_fx("test_wrong_statement", 'assert "a" == "a", "FIXTURE MARK: declared"', _MARK), FORBIDDEN),
    "test_fake_assertion_class": (_fx("test_fake_assertion_class", 'raise fake("FIXTURE MARK: fake class")', _MARK), FORBIDDEN),
    "test_assert_named_class": (_fx("test_assert_named_class", 'raise fake("setup helper failed")', r"assert \['A', 'Z'"), FORBIDDEN),
    "test_assertion_subclass": (_fx("test_assertion_subclass", 'raise Sub("FIXTURE MARK: subclass")', _MARK), FORBIDDEN),
    "test_pytest_fail": (_fx("test_pytest_fail", 'pytest.fail("FIXTURE MARK: via pytest.fail")', _MARK), FORBIDDEN),
    "test_marker_name_error": (_fx("test_marker_name_error", 'raise NameError("FIXTURE MARK: counterfeit setup failure")', _MARK), FORBIDDEN),
    "test_rewritten_elided": (_fx("test_rewritten_elided", "assert got == want", r"assert \"The property\.\.\."), GENUINE),
    "test_helper_value_error": (NativeNul(), FORBIDDEN),
    "test_genuine_native_through_seam": (NativeNul(), GENUINE),
    "test_native_without_seam": (NativeNul(), FORBIDDEN),
}  # fmt: skip

_LINE = re.compile(r"^(PASSED|FAILED|ERROR) (\S+)(?: - (.*))?$")
_EVIDENCE_LINE = re.compile(rf"^{EVIDENCE} (START|RECORD|END) (.*)$")


@dataclass
class Outcome:
    returncode: int | None  # None: timeout
    passed: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    records: dict[str, dict[str, Any]] = field(default_factory=dict)  # node -> its one record
    evidence_problem: str | None = "no evidence parsed"


def _evidence(lines: list[str], token: str) -> tuple[dict[str, dict[str, Any]], str | None]:
    """The reporter's records, or the reason the evidence is not usable (fail closed)."""
    starts, ends, records = [], [], []
    for line in lines:
        match = _EVIDENCE_LINE.match(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group(2))
        except ValueError:
            return {}, f"malformed evidence line: {line[:120]}"
        if not isinstance(payload, dict) or payload.get("token") != token:
            return {}, "evidence with a missing or foreign token"
        {"START": starts, "RECORD": records, "END": ends}[match.group(1)].append(payload)
    if len(starts) != 1 or len(ends) != 1:
        return {}, f"evidence framing: {len(starts)} START, {len(ends)} END"
    if ends[0].get("count") != len(records):
        return {}, f"evidence count {ends[0].get('count')} != {len(records)} records"
    by_node: dict[str, dict[str, Any]] = {}
    for record in records:
        node = record.get("node")
        if not isinstance(node, str) or node in by_node:
            return {}, f"duplicate or unnamed evidence record: {node!r}"
        by_node[node] = record
    return by_node, None


def parse(returncode: int | None, stdout: str, token: str = "") -> Outcome:
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
    out.records, out.evidence_problem = _evidence(lines, token)
    out.summary = next((ln for ln in reversed(lines) if not ln.startswith(EVIDENCE)), "")
    return out


def _same(a: str | Path, b: str | Path) -> bool:
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def _scope(tree: ast.AST, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The function whose code qualname is `qualname` (`outer.<locals>.inner`, `Class.method`)."""

    def walk(node: ast.AST, prefix: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if name == qualname and not isinstance(child, ast.ClassDef):
                    return child
                inner = f"{name}.<locals>." if not isinstance(child, ast.ClassDef) else f"{name}."
                found = walk(child, inner)
            else:
                found = walk(child, prefix)
            if found is not None:
                return found
        return None

    return walk(tree, "")


def statement_span(witness: Assertion) -> tuple[int, int]:
    """The line span of the ONE assert/raise/expression statement in the declared function that starts with
    the anchor (nested functions excluded). Raises when absent or ambiguous (R14)."""
    source = (ROOT / witness.file).read_text(encoding="utf-8")
    scope = _scope(ast.parse(source), witness.qualname)
    if scope is None:
        raise LookupError(f"no function {witness.qualname} in {witness.file}")
    found: list[ast.stmt] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, (ast.Assert, ast.Raise, ast.Expr)):
                text = ast.get_source_segment(source, child) or ""
                if text.startswith(witness.anchor):
                    found.append(child)
            visit(child)

    visit(scope)
    if len(found) != 1:
        raise LookupError(
            f"anchor matched {len(found)} statements in {witness.file}::{witness.qualname}"
        )
    return found[0].lineno, found[0].end_lineno or found[0].lineno


def classify(witness: Witness, record: dict[str, Any], seam_file: Path) -> tuple[str, str]:
    """GENUINE / MISMATCH / FORBIDDEN for one intended witness's observed record (R1-R10)."""
    if record.get("when") != "call":
        return FORBIDDEN, f"phase {record.get('when')!r}"
    if "error" in record:
        return FORBIDDEN, f"unobservable failure: {record['error']}"
    frames = record.get("frames")
    first = record.get("first_line")
    if not isinstance(frames, list) or not frames or not isinstance(first, str):
        return FORBIDDEN, "incomplete observation"
    inner = frames[-1]
    if isinstance(witness, Assertion):
        if record.get("is_assertion") is not True:
            return (
                FORBIDDEN,
                f"type is not the builtin AssertionError (observed {record.get('type')})",
            )
        low, high = statement_span(witness)
        if not (
            _same(inner["file"], ROOT / witness.file)
            and inner["qualname"] == witness.qualname
            and low <= inner["line"] <= high
        ):
            return (
                FORBIDDEN,
                f"raised at {inner['qualname']}:{inner['line']}, not the declared statement",
            )
        if re.match(witness.reason, first) is None:
            return MISMATCH, f"reason {first[:160]!r}"
        return GENUINE, first[:160]
    if record.get("is_value_error") is not True:
        return FORBIDDEN, f"type is not the builtin ValueError (observed {record.get('type')})"
    if not (_same(inner["file"], subprocess.__file__) and inner["qualname"] in NATIVE_FRAMES):
        return FORBIDDEN, f"raised at {inner['qualname']}, not the native call"
    if not any(_same(f["file"], seam_file) and f["qualname"] == SEAM_QUALNAME for f in frames):
        return FORBIDDEN, "the seam LocalSubprocess.spawn is not on the stack"
    if re.fullmatch(witness.reason, first) is None:
        return MISMATCH, f"reason {first[:160]!r}"
    return GENUINE, f"{first} at {inner['qualname']}"


def baseline_verdict(base: Outcome, controls: dict[str, Control]) -> str | None:
    """None when the baseline is green, every intended witness PASSED and the evidence is present."""
    if base.returncode != 0:
        return f"baseline exit {base.returncode}"
    if base.failed or base.errors:
        return f"baseline not green: {sorted(base.failed)[:3] + sorted(base.errors)[:3]}"
    if base.evidence_problem is not None:
        return f"baseline evidence: {base.evidence_problem}"
    missing = sorted({n for c in controls.values() for n in c.witnesses} - base.passed)
    return f"intended witnesses not PASSED in the baseline: {missing}" if missing else None


def credit(
    control: Control, base_passed: set[str], result: Outcome, seam_file: Path
) -> dict[str, object]:
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
    if result.evidence_problem is not None:
        return {"verdict": "INVALID", "why": f"evidence: {result.evidence_problem}"}
    observed: dict[str, dict[str, str]] = {}
    for node, witness in control.witnesses.items():
        record = result.records.get(node)
        if node in result.failed and record is None:
            return {"verdict": "INVALID", "why": f"FAILED without an evidence record: {node}"}
        if record is None:
            continue
        if node not in result.failed:
            return {"verdict": "INVALID", "why": f"evidence record without a FAILED line: {node}"}
        verdict, why = classify(witness, record, seam_file)
        observed[node] = {"classification": verdict, "detail": why, "type": str(record.get("type"))}
    forbidden = {n: o for n, o in observed.items() if o["classification"] == FORBIDDEN}
    genuine = {n: o for n, o in observed.items() if o["classification"] == GENUINE}
    if forbidden:
        return {
            "verdict": "INVALID",
            "why": "an intended witness did not fire as declared",
            "observed": observed,
        }
    if not genuine:
        return {
            "verdict": "NOT KILLED",
            "why": "no intended witness failed as declared",
            "observed": observed,
        }
    return {"verdict": "KILLED", "observed": observed}


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


def _command(paths: list[str]) -> list[str]:
    base = [
        sys.executable,
        "-m",
        "pytest",
        *paths,
        "-q",
        "-rA",
        "-p",
        "no:cacheprovider",
        "-p",
        PLUGIN,
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


def _pytest(src: Path, paths: list[str]) -> Outcome:
    token = secrets.token_hex(16)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(src), str(SCRIPTS)]),
        "COLUMNS": "4000",
        "PYTHONDONTWRITEBYTECODE": "1",
        TOKEN_ENV: token,
    }
    try:
        done = subprocess.run(
            _command(paths), cwd=ROOT, env=env, capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        return Outcome(None)
    return parse(done.returncode, done.stdout, token)


def _seam(src: Path) -> Path:
    return src / SUB


def fixture_check() -> tuple[bool, list[str]]:
    """W3: run the evidence fixture with the real command, reporter, parser and classifier; no writes."""
    result = _pytest(ROOT / "src", [FIXTURE_FILE])
    lines, ok = [], True
    if result.returncode != 1 or result.errors or result.evidence_problem is not None:
        return False, [
            f"fixture run: exit {result.returncode}, errors {sorted(result.errors)}, "
            f"evidence {result.evidence_problem}"
        ]
    for name, (witness, expected) in FIXTURE.items():
        node = f"{FIXTURE_FILE}::{name}"
        record = result.records.get(node)
        if node not in result.failed or record is None:
            ok = False
            lines.append(f"FAIL {name}: not FAILED with one record")
            continue
        got, why = classify(witness, record, _seam(ROOT / "src"))
        ok &= got == expected
        lines.append(
            f"{'PASS' if got == expected else 'FAIL'} W3 {name}: {got} (expected {expected}) {why}"
        )
    if set(result.records) != {f"{FIXTURE_FILE}::{n}" for n in FIXTURE}:
        ok = False
        lines.append(f"FAIL W3 record set: {sorted(result.records)}")
    return ok, lines


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
    result = _pytest(src, TESTS)
    return {
        "name": name,
        "expected": control.expected,
        **credit(control, base_passed, result, _seam(src)),
        "exit": result.returncode,
        "errors": len(result.errors),
        "summary": result.summary,
    }


# ---------------------------------------------------------------------------------------------- self-test


def _load(name: str, path: Path, source: str | None = None) -> types.ModuleType:
    """Load a module from `path`, or from a mutated `source` in memory (nothing is written)."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    text = source if source is not None else path.read_text(encoding="utf-8")
    exec(compile(text, str(path), "exec"), module.__dict__)
    return module


def _record(node: str, observation: dict[str, Any], when: str = "call") -> dict[str, Any]:
    return {"node": node, "when": when, **observation}


def _raised(make: Any) -> BaseException:
    try:
        make()
    except BaseException as error:
        return error
    raise AssertionError("did not raise")


def _helper_value_error() -> None:
    raise ValueError("embedded null character")


def rows(driver: types.ModuleType, plugin: types.ModuleType) -> list[tuple[str, bool]]:
    """W1: the credit matrix R1-R13 on REAL in-process observations and synthetic run outcomes."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from minion_agent.execution import subprocess as seam

    seam_file = Path(seam.__file__)
    out: list[tuple[str, bool]] = []
    tool = driver.CONTROLS["tool-string-matching"]
    t_node = next(iter(tool.witnesses))
    t_wit = tool.witnesses[t_node]
    t_low, _ = driver.statement_span(t_wit)
    native = driver.CONTROLS["default-cwd-unchecked"]
    n_node = next(iter(native.witnesses))

    def assertion_at(line: int, message: str, *, qualname: str = t_wit.qualname) -> dict[str, Any]:
        frame = {"file": str(ROOT / t_wit.file), "qualname": qualname, "line": line}
        return {"is_assertion": True, "is_value_error": False, "type": "builtins.AssertionError",
                "first_line": message, "frames": [frame]}  # fmt: skip

    def verdict(control: Any, node: str, records: dict[str, Any], *, code: int = 1,
                failed: bool = True, errors: dict[str, str] | None = None, problem: str | None = None,
                base: set[str] | None = None) -> str:  # fmt: skip
        outcome = driver.Outcome(code, failed={node: "x"} if failed else {}, errors=errors or {},
                                 records=records, evidence_problem=problem)  # fmt: skip
        return str(
            driver.credit(control, {node} if base is None else base, outcome, seam_file)["verdict"]
        )

    good = {t_node: _record(t_node, assertion_at(t_low, "PREFIX LOST: The fake"))}
    out.append(
        (
            "R1 genuine assertion at the declared statement is KILLED",
            verdict(tool, t_node, good) == "KILLED",
        )
    )

    def native_fire() -> None:
        real = seam.spawn_rejection
        seam.spawn_rejection = lambda *a, **k: None  # type: ignore[assignment]  # the mutant, in memory
        try:
            asyncio.run(
                seam.LocalSubprocess(str(ROOT)).spawn(
                    [sys.executable], seam.SpawnOptions(cwd="d\0")
                )
            )
        finally:
            seam.spawn_rejection = real

    real_native = plugin.observe(_raised(native_fire))
    out.append(("R2 genuine native fire through the seam (real, in-process) is KILLED",
                verdict(native, n_node, {n_node: _record(n_node, real_native)}) == "KILLED"))  # fmt: skip
    helper = plugin.observe(_raised(_helper_value_error))
    out.append(("R3 C001 same-text helper ValueError is INVALID",
                verdict(native, n_node, {n_node: _record(n_node, helper)}) == "INVALID"))  # fmt: skip

    def inside_seam() -> None:  # the same text raised by code the seam runs BEFORE the native call
        real = seam.spawn_rejection
        seam.spawn_rejection = lambda *a, **k: _helper_value_error()  # type: ignore[assignment]
        try:
            asyncio.run(
                seam.LocalSubprocess(str(ROOT)).spawn([sys.executable], seam.SpawnOptions())
            )
        finally:
            seam.spawn_rejection = real

    pre_boundary = plugin.observe(_raised(inside_seam))
    out.append(("R3 C001 same-text ValueError inside the seam before the native call is INVALID",
                verdict(native, n_node, {n_node: _record(n_node, pre_boundary)}) == "INVALID"))  # fmt: skip

    def no_seam() -> None:
        asyncio.run(asyncio.create_subprocess_exec(sys.executable, "a\0"))

    bypass = plugin.observe(_raised(no_seam))
    out.append(("R4 native fire without the seam frame is INVALID",
                verdict(native, n_node, {n_node: _record(n_node, bypass)}) == "INVALID"))  # fmt: skip
    fake_cls = type("AssertionError", (Exception,), {"__module__": "builtins"})
    named_cls = type("assert ['A', 'Z']", (Exception,), {"__module__": "builtins"})
    sub_cls = type("Sub", (AssertionError,), {})
    counterfeits = {
        "C002 fake AssertionError class": fake_cls("PREFIX LOST: The fake"),
        "C002 assert-named class": named_cls("PREFIX LOST: setup helper failed"),
        "AssertionError subclass": sub_cls("PREFIX LOST: subclass"),
        "NameError with the marker": NameError("PREFIX LOST: counterfeit"),
        "ModuleNotFoundError (real text)": _raised(lambda: importlib.import_module("PREFIX LOST")),
        "UnboundLocalError with the marker": UnboundLocalError("PREFIX LOST: local"),
        "ImportError with the marker": ImportError("PREFIX LOST: missing helper"),
    }
    for label, error in counterfeits.items():
        observation = plugin.observe(error)
        # Even when raised at the declared statement itself.
        observation["frames"] = assertion_at(t_low, "")["frames"]
        out.append((f"R5 {label} is observed as no assertion and is INVALID",
                    observation["is_assertion"] is False
                    and verdict(tool, t_node, {t_node: _record(t_node, observation)}) == "INVALID"))  # fmt: skip
    elsewhere = {t_node: _record(t_node, assertion_at(t_low + 1, "PREFIX LOST: other"))}
    helper_fn = {t_node: _record(t_node, assertion_at(t_low, "PREFIX LOST: x", qualname="helper"))}
    out.append(
        (
            "R6 a genuine assertion in another statement is INVALID",
            verdict(tool, t_node, elsewhere) == "INVALID",
        )
    )
    out.append(
        (
            "R6 a genuine assertion in another function is INVALID",
            verdict(tool, t_node, helper_fn) == "INVALID",
        )
    )
    late = {t_node: _record(t_node, assertion_at(t_low, "wrapper: PREFIX LOST"))}
    other = {t_node: _record(t_node, assertion_at(t_low, "assert 'x' == 'y'"))}
    out.append(
        (
            "R7 the marker not at the start is NOT KILLED",
            verdict(tool, t_node, late) == "NOT KILLED",
        )
    )
    out.append(("R7 another reason is NOT KILLED", verdict(tool, t_node, other) == "NOT KILLED"))
    mixed_ctl = driver.CONTROLS["check-after-native-spawn"]
    m_spy, m_canon = list(mixed_ctl.witnesses)
    spy_wit = mixed_ctl.witnesses[m_spy]
    spy_low, _ = driver.statement_span(spy_wit)
    spy_obs = {"is_assertion": True, "is_value_error": False, "type": "builtins.AssertionError",
               "first_line": "native spawn reached",
               "frames": [{"file": str(ROOT / spy_wit.file), "qualname": spy_wit.qualname, "line": spy_low}]}  # fmt: skip
    mixed = driver.Outcome(1, failed={m_spy: "x", m_canon: "x"},
                           records={m_spy: _record(m_spy, spy_obs), m_canon: _record(m_canon, helper)},
                           evidence_problem=None)  # fmt: skip
    both_good = driver.Outcome(1, failed={m_spy: "x", m_canon: "x"},
                               records={m_spy: _record(m_spy, spy_obs), m_canon: _record(m_canon, real_native)},
                               evidence_problem=None)  # fmt: skip
    out.append(("R8 a genuine witness does not mask a forbidden one (mixed is INVALID)",
                driver.credit(mixed_ctl, {m_spy, m_canon}, mixed, seam_file)["verdict"] == "INVALID"))  # fmt: skip
    out.append(("R1/R2 both witnesses genuine is KILLED",
                driver.credit(mixed_ctl, {m_spy, m_canon}, both_good, seam_file)["verdict"] == "KILLED"))  # fmt: skip
    out.append(("R9 FAILED without an evidence record is INVALID (no rendered fallback)",
                verdict(tool, t_node, {}) == "INVALID"))  # fmt: skip
    out.append(
        (
            "R9 a record without a FAILED line is INVALID",
            verdict(tool, t_node, good, failed=False) == "INVALID",
        )
    )
    stdout = f"FAILED {t_node} - AssertionError: PREFIX LOST: The fake\n"
    line = json.dumps({"token": "T", **good[t_node]})
    framed = f'{EVIDENCE} START {{"token": "T"}}\n{EVIDENCE} RECORD {line}\n{EVIDENCE} END {{"token": "T", "count": 1}}\n'
    out.append(
        (
            "R9 framed evidence with the run's token parses",
            driver.parse(1, stdout + framed, "T").evidence_problem is None,
        )
    )
    for label, text in {
        "a foreign token": framed.replace('"T"', '"U"', 1),
        "a missing START": framed.split("\n", 1)[1],
        "a missing END": framed.rsplit(f"{EVIDENCE} END", 1)[0],
        "a wrong END count": framed.replace('"count": 1', '"count": 2'),
        "a duplicate record": framed.replace(
            f"{EVIDENCE} END", f"{EVIDENCE} RECORD {line}\n{EVIDENCE} END"
        ),
    }.items():
        parsed = driver.parse(1, stdout + text, "T")
        out.append((f"R9 {label} makes the evidence INVALID",
                    parsed.evidence_problem is not None
                    and str(driver.credit(tool, {t_node}, parsed, seam_file)["verdict"]) == "INVALID"))  # fmt: skip
    setup = {t_node: _record(t_node, assertion_at(t_low, "PREFIX LOST: x"), when="setup")}
    out.append(
        (
            "R10 a non-call phase on an intended node is INVALID",
            verdict(tool, t_node, setup) == "INVALID",
        )
    )
    unobservable = {t_node: {"node": t_node, "when": "call", "error": "RuntimeError()"}}
    out.append(
        ("R10 an unobservable failure is INVALID", verdict(tool, t_node, unobservable) == "INVALID")
    )
    out.append(
        (
            "R10 an ERROR entry is INVALID",
            verdict(tool, t_node, good, errors={"tests/x.py": "e"}) == "INVALID",
        )
    )
    for code in (2, 3, 4, 5):
        out.append((f"R10 pytest exit {code} is INVALID even with a genuine record",
                    verdict(tool, t_node, good, code=code) == "INVALID"))  # fmt: skip
    timeout = driver.Outcome(None)
    out.append(
        (
            "R10 a timeout is INVALID",
            driver.credit(tool, {t_node}, timeout, seam_file)["verdict"] == "INVALID",
        )
    )
    out.append(("R11 no intended failure (only unrelated nodes) is NOT KILLED",
                verdict(tool, t_node, {}, failed=False) == "NOT KILLED"))  # fmt: skip
    out.append(("R12 a witness missing from the baseline is INVALID",
                verdict(tool, t_node, good, base=set()) == "INVALID"))  # fmt: skip
    every = {n for c in driver.CONTROLS.values() for n in c.witnesses}
    green = driver.Outcome(0, passed=every, evidence_problem=None)
    out.append(("R12 a green, selected baseline with evidence is accepted",
                driver.baseline_verdict(green, driver.CONTROLS) is None))  # fmt: skip
    for label, base in {
        "a red baseline": driver.Outcome(
            1, passed=every, failed={t_node: "x"}, evidence_problem=None
        ),
        "a filtered witness": driver.Outcome(0, passed=every - {t_node}, evidence_problem=None),
        "a baseline without evidence": driver.Outcome(0, passed=every),
    }.items():
        out.append(
            (f"R12 {label} is INVALID", driver.baseline_verdict(base, driver.CONTROLS) is not None)
        )
    out.append(
        (
            "R13 exit 0 is NOT KILLED",
            verdict(tool, t_node, {}, code=0, failed=False) == "NOT KILLED",
        )
    )
    return out


def declarations(driver: types.ModuleType) -> list[tuple[str, bool]]:
    """W2 (R14): every declaration resolves to exactly one statement in the CURRENT source."""
    out = []
    witnesses = [
        (f"{n}: {node}", w) for n, c in driver.CONTROLS.items() for node, w in c.witnesses.items()
    ]
    witnesses += [(f"fixture {n}", w) for n, (w, _) in driver.FIXTURE.items()]
    for label, witness in witnesses:
        if isinstance(witness, driver.NativeNul):
            out.append(
                (
                    f"R14 {label} declares NATIVE_NUL",
                    witness.reason == r"embedded null (character|byte)",
                )
            )
            continue
        try:
            driver.statement_span(witness)
            out.append((f"R14 {label} resolves to one statement", True))
        except LookupError as error:
            out.append((f"R14 {label}: {error}", False))
    out.append(("R14 all 15 controls are declared", len(driver.CONTROLS) == 15))
    return out


# Meta-controls (known-bad -> FAIL on the named row): (target, anchor, replacement, row prefix that must fail).
META: dict[str, tuple[str, str, str, str]] = {
    "D1 kind by class name": ("plugin", '"is_assertion": type(exc) is _ASSERTION,', '"is_assertion": type(exc).__name__ == "AssertionError",', "R5 C002 fake AssertionError class"),
    "D2 blanket ValueError rejection": ("driver", '    if record.get("is_value_error") is not True:\n', "    if True:\n", "R2 genuine native"),
    "D3 unanchored reason": ("driver", "        if re.match(witness.reason, first) is None:\n", "        if re.search(witness.reason, first) is None:\n", "R7 the marker not at the start"),
    "D4 origin dropped": ("driver", "    if not (_same(inner[\"file\"], subprocess.__file__) and inner[\"qualname\"] in NATIVE_FRAMES):\n", "    if False:\n", "R3 C001"),
    "D4 statement origin dropped": ("driver", "            and low <= inner[\"line\"] <= high\n", "            and True\n", "R6 a genuine assertion in another statement"),
    "D5 seam dropped": ("driver", "    if not any(_same(f[\"file\"], seam_file) and f[\"qualname\"] == SEAM_QUALNAME for f in frames):\n", "    if False:\n", "R4 native fire without the seam"),
    "D6 masking": ("driver", "    if forbidden:\n        return {\n", "    if forbidden and not genuine:\n        return {\n", "R8 a genuine witness does not mask"),
    "D7 rendered fallback": ("driver", '            return {"verdict": "INVALID", "why": f"FAILED without an evidence record: {node}"}\n', '            if re.search(str(getattr(witness, "reason", "")), result.failed[node]):\n                return {"verdict": "KILLED", "observed": {}}\n            continue\n', "R9 FAILED without an evidence record"),
    "D8 token dropped": ("driver", '        if not isinstance(payload, dict) or payload.get("token") != token:\n', "        if not isinstance(payload, dict):\n", "R9 a foreign token"),
    "D9 exit ignored": ("driver", "    if result.returncode != 1:\n", "    if False:\n", "R10 pytest exit 2"),
    "D10 baseline selection skipped": ("driver", "    missing = sorted(n for n in control.witnesses if n not in base_passed)\n", "    missing: list[str] = []\n", "R12 a witness missing from the baseline"),
}  # fmt: skip


def in_process_checks() -> list[tuple[str, bool]]:
    """W1 + W2 + D1-D10, all in memory (W4 runs exactly this in the suite)."""
    driver_path, plugin_path = Path(__file__).resolve(), SCRIPTS / f"{PLUGIN}.py"
    driver = _load("l12_d011_driver_under_test", driver_path)
    plugin = _load("l12_d011_plugin_under_test", plugin_path)
    results = rows(driver, plugin) + declarations(driver)
    sources = {
        "driver": driver_path.read_text(encoding="utf-8"),
        "plugin": plugin_path.read_text(encoding="utf-8"),
    }
    for name, (target, anchor, replacement, row) in META.items():
        text = sources[target].replace("\r\n", "\n")
        if text.count(anchor) != 1:
            results.append((f"{name}: anchor matched {text.count(anchor)} times", False))
            continue
        mutated = text.replace(anchor, replacement)
        bad_driver = _load(
            f"meta_driver_{len(results)}", driver_path, mutated if target == "driver" else None
        )
        bad_plugin = _load(
            f"meta_plugin_{len(results)}", plugin_path, mutated if target == "plugin" else None
        )
        named = [ok for label, ok in rows(bad_driver, bad_plugin) if label.startswith(row)]
        results.append((f"{name}: known-bad FAILS on '{row}'", bool(named) and not all(named)))
    return results


def self_test() -> int:
    results = in_process_checks()
    ok_w3, w3_lines = fixture_check()
    for label, ok in results:
        print(f"{'PASS' if ok else 'FAIL'} {label}")
    print("\n".join(w3_lines))
    failed = sum(not ok for _, ok in results) + (not ok_w3)
    print(
        f"DRIVER SELF-TEST {'FAILED' if failed else 'PASSED'} ({len(results)} in-process checks, W3 {'PASS' if ok_w3 else 'FAIL'})"
    )
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--list"]:
        print(json.dumps({n: {"witnesses": {k: repr(w) for k, w in c.witnesses.items()}, "expected": c.expected}
                          for n, c in CONTROLS.items()}, indent=1))  # fmt: skip
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
    names = argv[2:] or list(CONTROLS)
    report: dict[str, object] = {"os": "win32" if os.name == "nt" else "linux"}
    ok_w3, w3_lines = fixture_check()
    report["fixture"] = {"verdict": "PASS" if ok_w3 else "FAIL", "rows": w3_lines}
    # Unconfirmed evidence (including this platform's native origin) credits nothing.
    if not ok_w3:
        report["results"] = []
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return 1
    _checked(work).mkdir(parents=True)
    _checked(work / "baseline").mkdir()
    _copy_src(work / "baseline" / "src")
    base = _pytest(work / "baseline" / "src", TESTS)
    invalid = baseline_verdict(base, {n: CONTROLS[n] for n in names})
    report["baseline"] = {
        "exit": base.returncode,
        "passed": len(base.passed),
        "failed": sorted(base.failed),
        "errors": sorted(base.errors),
        "evidence": base.evidence_problem or "present",
        "summary": base.summary,
        "verdict": invalid or "GREEN",
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
