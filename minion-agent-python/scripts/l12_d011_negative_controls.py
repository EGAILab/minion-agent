"""L12-D011 negative controls (spec/execution.md section 21; record section 3.4). Each fault is applied
to a copy of `src` under a NEW work directory, and the spawn-NUL suites run against it; a control
counts as killed only when its INTENDED witness fails (workflow section 9.7) -- an unrelated failure,
or a collection/setup/import error, is not a kill.

    python scripts/l12_d011_negative_controls.py --work <new dir inside the project root> [NAME ...]
    python scripts/l12_d011_negative_controls.py --list

Containment (Owner rule 2026-10-10): `--work` must be a NEW directory strictly inside the project root
(MINION_PROJECT_ROOT, default the Windows checkout root; a container passes its private /tmp root).
Every target is proven -- lexically inside, no link or reparse point on any existing ancestor, real
path inside -- immediately before each mkdir/copy/write. Nothing is ever deleted; the caller removes
the work directory with its own guarded cleanup. Needs MINION_SEARCH_ENGINE_ARTIFACTS for the tool
cases. Set `E5_BARE_PYTEST=1` in a container without the pytest-cov configuration.
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
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = Path(
    os.environ.get("MINION_PROJECT_ROOT", "E:/AI/Projects/OpenMinds/Minions/Minion-Agent")
).resolve()
EXE = Path("minion_agent/execution")
SUB, NUL, SEARCH = (
    EXE / "subprocess.py",
    EXE / "spawn_nul.py",
    Path("minion_agent/tools/builtin/_search.py"),
)
BASH = Path("minion_agent/tools/builtin/bash.py")
UNIT = "tests/execution/test_spawn_nul.py"
CANON = "tests/conformance/test_spawn_nul_conformance.py"
TESTS = [UNIT, CANON]


def canon(case: str) -> str:
    return f"{CANON}::test_spawn_case[{case}]"


@dataclass(frozen=True)
class Control:
    edits: list[tuple[Path, str, str]]
    witnesses: list[str]
    expected: str


_CHECK = "        if rejected is not None:\n            return Err(rejected)\n"
CONTROLS: dict[str, Control] = {
    "check-after-native-spawn": Control(
        [(SUB, _CHECK, "")],
        [f"{UNIT}::test_no_native_spawn_after_rejection[argv0-options0]", canon("name/file")],
        "the native spawn is reached (the spy fires; Python's ValueError leaks)",
    ),
    "baseline-first-posix-order": Control(
        [(NUL, "    owned_set = set(owned)\n", "    return names\n    owned_set = set(owned)\n")],
        [f"{UNIT}::test_env_check_order_both_platforms"],
        "POSIX order follows baseline presentation, not overlay-first",
    ),
    "sort-all-posix-order": Control(
        [
            (
                NUL,
                "    owned_set = set(owned)\n",
                "    return sorted(names, key=_utf16_key)\n    owned_set = set(owned)\n",
            )
        ],
        [f"{UNIT}::test_env_check_order_both_platforms"],
        "POSIX order sorts every name",
    ),
    "naive-integer-index": Control(
        [(NUL, '_INDEX = re.compile(r"0|[1-9][0-9]*")\n', '_INDEX = re.compile(r"[0-9]+")\n')],
        [f"{UNIT}::test_env_check_order_both_platforms"],
        "01 / 4294967295 are taken for integer indices",
    ),
    "superseded-baseline-value": Control(
        [
            (
                SUB,
                "            effective_env,\n            [name for name, _ in baseline],\n",
                "            {**(opts.env or {}), **dict(baseline)},\n            [name for name, _ in baseline],\n",
            )
        ],
        [canon("env/superseded-baseline-value-control")],
        "the superseded baseline value is checked instead of the effective one",
    ),
    "raw-cwd-not-resolved": Control(
        [
            (
                SUB,
                "            argv,\n            resolved_cwd,\n",
                "            argv,\n            opts.cwd or self.cwd,\n",
            )
        ],
        [canon("cwd/file-url-percent-00"), canon("cwd/literal-relative")],
        "the raw spelling is checked: %00 is missed, a relative cwd is reported unresolved",
    ),
    "double-url-decode": Control(
        [
            (
                SUB,
                "        resolved_cwd = resolve_local_path(self.cwd, opts.cwd) if opts.cwd is not None else self.cwd\n",
                "        from urllib.parse import unquote\n        resolved_cwd = unquote(resolve_local_path(self.cwd, opts.cwd)) if opts.cwd is not None else self.cwd\n",
            )
        ],
        [canon("cwd/file-url-percent-2500-control")],
        "%2500 is decoded twice into a NUL",
    ),
    "default-cwd-unchecked": Control(
        [(NUL, "    if _NUL in cwd:\n", "    if _NUL in cwd and cwd != cwd:\n")],
        [canon("cwd/provider-default-nul")],
        "a NUL in the provider default cwd reaches the native spawn",
    ),
    "cwd-before-args": Control(
        [
            (
                NUL,
                '    rejected = argv_rejection(argv)\n    if rejected is not None:\n        return rejected\n    if _NUL in cwd:\n        return _rejection(diagnostic("options.cwd", cwd, _CWD_REASON))\n',
                '    if _NUL in cwd:\n        return _rejection(diagnostic("options.cwd", cwd, _CWD_REASON))\n    rejected = argv_rejection(argv)\n    if rejected is not None:\n        return rejected\n',
            )
        ],
        [canon("precedence/args-before-cwd")],
        "the cwd is reported before an invalid argument",
    ),
    "cut-at-or-above-128": Control(
        [(NUL, "    if len(received) > _CUT:\n", "    if len(received) >= _CUT:\n")],
        [canon("len/inspected-128")],
        "an inspected value of exactly 128 units is cut",
    ),
    "quote-selection": Control(
        [
            (
                NUL,
                "        if 34 not in piece:\n            quote = 34\n",
                "        if 34 not in piece:\n            quote = 39\n",
            )
        ],
        [canon("fmt/single-quote")],
        "a value with ' is single-quoted and escaped",
    ),
    "tool-string-matching": Control(
        [
            (
                SEARCH,
                "        if spawned.error.nul_input:\n",
                '        if spawned.error.message.startswith("The "):\n',
            )
        ],
        [f"{UNIT}::test_engine_keeps_its_prefix_for_a_non_nul_failure"],
        "an OS message starting with 'The ' loses its prefix",
    ),
    "tool-checks-projected-argv": Control(
        [
            (
                SEARCH,
                "    rejected = argv_rejection(argv)\n",
                "    rejected = argv_rejection([scalar_command(arg) for arg in argv])\n",
            )
        ],
        [f"{UNIT}::test_engine_diagnostic_shows_the_unprojected_value"],
        "the diagnostic shows U+FFFD for a lone surrogate",
    ),
    "bash-checks-stdin-command": Control(
        [
            (
                BASH,
                "            [config.shell, *config.args] if via_stdin else [config.shell, *config.args, command]\n",
                "            [config.shell, *config.args, command]\n",
            )
        ],
        [f"{UNIT}::test_bash_stdin_transport_command_is_not_argv"],
        "a stdin-transported command is rejected as argv",
    ),
}

_SUMMARY = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$")


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
        "-rfE",
        "-p",
        "no:cacheprovider",
        "--no-header",
    ]
    return base + (["-o", "addopts="] if os.environ.get("E5_BARE_PYTEST") else ["--no-cov"])


def run(name: str, work: Path) -> dict[str, object]:
    control = CONTROLS[name]
    src = work / name / "src"
    _checked(work / name).mkdir()
    for directory, dirs, files in os.walk(ROOT / "src"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel = Path(directory).relative_to(ROOT / "src")
        _checked(src / rel).mkdir()
        for file in files:
            shutil.copyfile(Path(directory) / file, _checked(src / rel / file))
    texts: dict[Path, str] = {}
    for target, anchor, replacement in control.edits:
        path = src / target
        text = texts.get(path) or path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        if text.count(anchor) != 1:
            return {"name": name, "error": f"anchor matched {text.count(anchor)} times in {target}"}
        texts[path] = text.replace(anchor, replacement)
    for path, text in texts.items():
        _checked(path).write_bytes(text.encode("utf-8"))
    env = {**os.environ, "PYTHONPATH": str(src), "COLUMNS": "4000"}
    try:
        result = subprocess.run(
            _command(), cwd=ROOT, env=env, capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        return {"name": name, "error": "hung (timeout); not a kill"}
    lines = result.stdout.splitlines()
    failures = {m.group(2): (m.group(1), m.group(3) or "") for m in map(_SUMMARY.match, lines) if m}
    errors = sorted(n for n, (kind, _) in failures.items() if kind == "ERROR" or "::" not in n)
    killing = {
        n: failures[n][1] for n in control.witnesses if failures.get(n, ("",))[0] == "FAILED"
    }
    return {
        "name": name,
        "killed": bool(killing) and not errors,
        "intended_witnesses": control.witnesses,
        "expected": control.expected,
        "observed": killing,
        "other_failures": sorted(set(failures) - set(killing) - set(errors)),
        "collection_or_setup_errors": errors,
        "summary": lines[-1] if lines else "",
    }


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
    if argv[:1] != ["--work"] or len(argv) < 2:
        print("usage: --work <new dir inside the project root> [NAME ...]", file=sys.stderr)
        return 2
    work = _checked(Path(argv[1]))
    if work.exists():
        print(f"--work must be NEW: {work}", file=sys.stderr)
        return 2
    _checked(work).mkdir(parents=True)
    results = [run(name, work) for name in (argv[2:] or list(CONTROLS))]
    print(
        json.dumps(
            {"os": "win32" if os.name == "nt" else "linux", "results": results},
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0 if all(r.get("killed") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
