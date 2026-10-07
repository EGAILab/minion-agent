"""WP-13.4 `find`/`grep` negative controls (spec/tools.md WP-13.4 "Negative controls"). Each fault
is applied to a temporary copy of `src`; the search suites (the binding witnesses and the canonical
scenarios over the pinned engines) run against it, and the control counts as killed only when its
INTENDED witness fails (workflow section 9.7) -- an unrelated failure, or a collection, setup or
import error, is not a kill.

    python scripts/wp134_search_negative_controls.py [--list] [NAME ...]

Needs `MINION_SEARCH_ENGINE_ARTIFACTS` (the official engine artifacts, as for the conformance suite).
A control whose witness exists only on one platform is reported `not_applicable` on the other.
Set `E5_BARE_PYTEST=1` in a container that lacks the project's pytest-cov configuration. Exit status
0 iff every applicable selected control was killed by its intended witness.
"""

# ruff: noqa: E501 -- the fault table quotes source lines verbatim as unique anchors

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = Path("minion_agent/tools/builtin")
FIND, GREP = PKG / "find.py", PKG / "grep.py"
SEARCH = PKG / "_search.py"
ENGINES, READLINE, NODE_PATH = (
    PKG / "search_engines.py",
    PKG / "_readline.py",
    PKG / "_node_path.py",
)
UNIT = "tests/tools/builtin/test_search.py"
CANON = "tests/conformance/test_builtin_search_conformance.py"
TESTS = [UNIT, CANON]
HOST = "win32" if os.name == "nt" else "linux"


def _scenario(name: str) -> str:
    return f"{CANON}::test_builtin_search_scenario[{name}]"


@dataclass(frozen=True)
class Control:
    edits: list[tuple[Path, str, str]]
    witnesses: list[str]  # pytest node ids; any one of them failing is the intended kill
    expected: str  # the discriminating failure the witness should report
    platforms: tuple[str, ...] = ("win32", "linux")


_UNVERIFIED = r"""        if not await asyncio.to_thread(self.is_verified, engine):
            raise not_provisioned(engine)
"""

CONTROLS: dict[str, Control] = {
    "sort_find_results": Control(
        [
            (
                FIND,
                "        limit_reached = len(relativized) >= effective_limit\n",
                "        relativized.sort()\n        limit_reached = len(relativized) >= effective_limit\n",
            )
        ],
        [f"{UNIT}::test_find_reproduces_an_unsorted_engine_stream"],
        "witness 6: z.ts then a.ts comes back as a.ts then z.ts",
    ),
    "sort_grep_results": Control(
        [
            (
                GREP,
                "        for file_path, line_number, line_text in matches:\n",
                "        for file_path, line_number, line_text in sorted(matches, key=lambda m: (m[0], m[1])):\n",
            )
        ],
        [f"{UNIT}::test_grep_reproduces_an_unsorted_engine_stream"],
        "witness 6: the engine's z.ts/a.ts/z.ts stream is reordered",
    ),
    "dedupe_find_entries": Control(
        [
            (
                FIND,
                "            if line:\n                relativized.append(relativize(line, search_path, node))\n",
                "            if line and relativize(line, search_path, node) not in relativized:\n                relativized.append(relativize(line, search_path, node))\n",
            )
        ],
        [
            f"{UNIT}::test_find_keeps_entries_that_format_identically",
            _scenario("builtin-search-find-duplicate-trailing-space-duplicate"),
        ],
        "witness 7: `same.ts` and `same.ts ` give one entry instead of two",
    ),
    "following_git_probe_on_windows": Control(
        [
            (
                FIND,
                "        return not isinstance(await fs.file_info(path), Err)\n",
                "        return not isinstance(await fs.canonical_path(path), Err)\n",
            )
        ],
        [_scenario("builtin-search-find-junction-dangling-git-junction")],
        "witness 7: a dangling .git junction reads as absent, so --no-require-git applies the parent .gitignore",
        ("win32",),
    ),
    "no_require_git_passed_to_rg": Control(
        [
            (
                GREP,
                '        args = ["--json", "--line-number", "--color=never", "--hidden"]\n',
                '        args = ["--json", "--line-number", "--color=never", "--hidden", "--no-require-git"]\n',
            )
        ],
        [f"{UNIT}::test_grep_argument_vectors"],
        "witness 4: rg's argv gains --no-require-git",
    ),
    "no_require_git_omitted_for_fd": Control(
        [(FIND, '            args.append("--no-require-git")\n', "            pass\n")],
        [f"{UNIT}::test_find_argument_vectors"],
        "witness 4: fd's argv outside a repository lacks --no-require-git",
    ),
    "merge_context_windows": Control(
        [
            (
                GREP,
                "                label = number_to_string(current)\n",
                '                label = number_to_string(current)\n                if any(o.startswith((f"{relative}:{label}: ", f"{relative}-{label}- ")) for o in output):\n                    current += 1\n                    continue\n',
            )
        ],
        [_scenario("builtin-search-grep-plain-context-overlap")],
        "overlapping windows print each line once instead of once per match",
    ),
    "integer_find_limit": Control(
        [
            (
                FIND,
                "        effective_limit = DEFAULT_LIMIT if limit is None else limit\n",
                "        effective_limit = DEFAULT_LIMIT if limit is None else int(limit)\n",
            )
        ],
        [f"{UNIT}::test_find_argument_vectors"],
        "witness 5: limit 2.5 reaches fd as --max-results 2",
    ),
    "integer_grep_limit": Control(
        [
            (
                GREP,
                "        effective_limit = math_max(1, DEFAULT_LIMIT if limit is None else limit)\n",
                "        effective_limit = math_max(1, DEFAULT_LIMIT if limit is None else int(limit))\n",
            )
        ],
        [f"{UNIT}::test_grep_limits_and_kill_for_limit"],
        "witness 5: limit 2.5 stops at 2 matches with a `2 matches` notice",
    ),
    "integer_grep_context": Control(
        [
            (
                GREP,
                "        context_value = context if (context and context > 0) else 0\n",
                "        context_value = int(context) if (context and context > 0) else 0\n",
            )
        ],
        [f"{UNIT}::test_grep_context_reconstruction"],
        "witness 5: context 0.5 prints the matched line instead of the fractional 1.5/2.5 window",
    ),
    "drop_uncollected_match_count": Control(
        [
            (
                GREP,
                '            state["count"] += 1\n            data = _field(event, "data")\n',
                '            data = _field(event, "data")\n',
            ),
            (
                GREP,
                "                matches.append((file_path, line_number, line_text))\n",
                '                matches.append((file_path, line_number, line_text))\n                state["count"] += 1\n',
            ),
        ],
        [f"{UNIT}::test_grep_counts_uncollected_matches_and_skips_noise"],
        "a match without path text gives `No matches found` instead of an empty result",
    ),
    "strip_bom_in_context": Control(
        [
            (
                GREP,
                "                    text = decode_utf8(read.value).replace(",
                '                    text = decode_utf8(read.value).lstrip("\\ufeff").replace(',
            )
        ],
        [f"{UNIT}::test_grep_context_reconstruction"],
        "the first context line loses its U+FEFF",
    ),
    "cr_not_a_line_break": Control(
        [
            (
                READLINE,
                '            if char == "\\n" or char == "\\r":\n',
                '            if char == "\\n":\n',
            )
        ],
        [f"{UNIT}::test_readline_splitting"],
        "a lone CR no longer ends a line",
    ),
    "truncate_line_in_code_points": Control(
        [
            (
                GREP,
                '    units = to_units(line)\n    if len(units) <= GREP_MAX_LINE_LENGTH:\n        return line, False\n    return from_units(units[:GREP_MAX_LINE_LENGTH]) + "... [truncated]", True\n',
                '    if len(line) <= GREP_MAX_LINE_LENGTH:\n        return line, False\n    return line[:GREP_MAX_LINE_LENGTH] + "... [truncated]", True\n',
            )
        ],
        [f"{UNIT}::test_grep_line_truncation_and_notices"],
        "witness 5: the cut keeps the whole emoji instead of splitting its surrogate pair at unit 500",
    ),
    "trim_before_empty_test": Control(
        [
            (
                FIND,
                '        output = "\\n".join(lines)\n',
                '        output = js_trim("\\n".join(lines))\n',
            )
        ],
        [f"{UNIT}::test_find_tests_emptiness_before_trimming"],
        "whitespace-only output with exit 1 becomes the stderr error",
    ),
    "path_fallback": Control(
        [
            (
                ENGINES,
                _UNVERIFIED,
                "        if not await asyncio.to_thread(self.is_verified, engine):\n            import shutil\n\n            found = shutil.which(engine)\n            if found is None:\n                raise not_provisioned(engine)\n            return [found]\n",
            )
        ],
        [f"{UNIT}::test_a_tool_call_never_downloads_or_uses_path"],
        "DIV-003: the planted PATH executable runs instead of the not-provisioned error",
    ),
    "download_from_tool_call": Control(
        [
            (
                ENGINES,
                _UNVERIFIED,
                "        if not await asyncio.to_thread(self.is_verified, engine):\n            await asyncio.to_thread(provision_search_engines, self)\n",
            )
        ],
        [f"{UNIT}::test_a_tool_call_never_downloads_or_uses_path"],
        "DIV-003: DownloadAttemptedError -- the tool call fetched an artifact",
    ),
    "trust_existing_file": Control(
        [
            (
                ENGINES,
                "        return _sha256_file(self.binary_path(engine)) == artifact.binary_sha256\n",
                "        return self.binary_path(engine).is_file()\n",
            )
        ],
        [f"{UNIT}::test_unprovisioned_store_texts"],
        "DIV-003: a file at the fixed name that fails its SHA-256 is used",
    ),
    "non_atomic_install": Control(
        [
            (
                ENGINES,
                '    handle, temporary = tempfile.mkstemp(dir=store.root, prefix=f".{engine}-", suffix=".partial")\n',
                '    temporary = str(store.binary_path(engine))\n    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o644)\n',
            )
        ],
        [f"{UNIT}::test_the_fixed_name_appears_only_by_the_atomic_rename"],
        "DIV-003: the fixed name exists (partly written) before the rename",
    ),
    "case_sensitive_windows_relativize": Control(
        [
            (
                NODE_PATH,
                "                result = ntpath.relpath(target_n, start_n)\n",
                '                s_parts, t_parts = start_n.split("\\\\"), target_n.split("\\\\")\n'
                "                common = 0\n"
                "                while common < min(len(s_parts), len(t_parts)) and s_parts[common] == t_parts[common]:\n"
                "                    common += 1\n"
                '                result = "\\\\".join([".."] * (len(s_parts) - common) + t_parts[common:]) or "."\n',
            )
        ],
        [f"{UNIT}::test_relativize_and_node_paths"],
        "`c:\\R\\a\\` under `C:\\r` relativizes through `..` instead of to `a/`",
    ),
    # Remediation 1 (WP134-IMPL-R001, R002): the reviewed defects, as single-point mutants.
    # CE-L13-WP134-01 (agreed checkpoint revision 2): the converged rules, as single-point mutants.
    "find_decides_after_release": Control(
        [
            (
                FIND,
                "                await run.run(on_line, complete)\n",
                "                await run.run(on_line)\n                complete()\n",
            )
        ],
        [
            f"{UNIT}::test_abort_window_partition[stdout_close-find]",
            f"{UNIT}::test_abort_window_partition[stderr_close-find]",
        ],
        "R001: an abort during the stream release, after engine completion, gives `Operation aborted` instead of a.ts",
    ),
    "grep_decides_after_release": Control(
        [
            (
                GREP,
                "            await run.run(on_line, window.close)\n",
                "            await run.run(on_line)\n            window.close()\n",
            )
        ],
        [
            f"{UNIT}::test_abort_window_partition[stdout_close-grep]",
            f"{UNIT}::test_abort_window_partition[stderr_close-grep]",
        ],
        "R001: an abort during the stream release, after engine completion, gives `Operation aborted` instead of the match",
    ),
    "rerun_spawn_not_reverified": Control(
        [
            (
                FIND,
                '            fd = await resolve_engine(engines, subprocess, "fd")\n',
                "",
            )
        ],
        [
            f"{UNIT}::test_the_diagnostic_rerun_spawn_is_verified_too[True]",
        ],
        "R004: the rule-5 re-run spawns a binary replaced after the first verification instead of "
        "giving the not-provisioned error",
    ),
    "stop_ack_join_before_completion": Control(
        [
            (
                SEARCH,
                "            status = await self.process.wait()\n",
                "            if self._stopping is not None:\n"
                "                await self._stopping\n"
                "            status = await self.process.wait()\n",
            )
        ],
        [
            f"{UNIT}::test_a_held_stop_acknowledgement_does_not_extend_the_window"
            "[while_only_the_stop_ack_is_pending]"
        ],
        "R001 (targeted closure 1): joining the limit-stop acknowledgement before completion keeps "
        "the window open, so an abort while only the acknowledgement is pending gives "
        "`Operation aborted` instead of the limit result",
    ),
    "grep_latch_lost_at_completion": Control(
        [
            (
                GREP,
                '        if state["aborted"] or window.observed:\n',
                '        if state["aborted"]:\n',
            )
        ],
        [
            f"{UNIT}::test_grep_keeps_an_abort_that_lands_just_before_completion",
            f"{UNIT}::test_abort_window_partition[stdout_data-grep]",
        ],
        "R001: an abort inside the window, unseen by the polling watcher, gives the match",
    ),
    "grep_observes_pre_registration_abort": Control(
        [
            (
                GREP,
                "        window = AbortWindow(signal)\n",
                "        window = AbortWindow(signal)\n        window.open = signal is not None\n",
            )
        ],
        [
            f"{UNIT}::test_abort_window_partition[spawn-grep]",
            f"{UNIT}::test_grep_ignores_an_abort_before_its_listener_registration",
        ],
        "R001: an abort before grep's listener exists aborts the call",
    ),
    "zero_directory_left_broken": Control(
        [(FIND, "    return render(0, count)\n", "    return _pi_windows_rewrite(pattern)\n")],
        [
            f"{UNIT}::test_windows_full_path_normalization",
            _scenario("builtin-search-find-components-mixed-star-crossing"),
            _scenario("builtin-search-find-plain-full-path-spec"),
        ],
        "DIV-002: Pi's Windows text; `src/**/a*.ts` misses src/a.ts and src/a/sub/b.ts",
    ),
    "whole_pattern_linux_conversion": Control(
        [
            (
                FIND,
                "            if line:\n                relativized.append(relativize(line, search_path, node))\n",
                '            if line and ("/" not in pattern or __import__("pathlib").PurePosixPath(line.replace("\\\\", "/")).full_match(prefixed)):\n'
                "                relativized.append(relativize(line, search_path, node))\n",
            )
        ],
        [_scenario("builtin-search-find-components-mixed-star-crossing")],
        "C001: Linux semantics for the whole pattern drops src/a/sub/b.ts (the retained single-* crossing)",
        ("win32",),
    ),
    "alt_start_not_recursive": Control(
        [
            (
                FIND,
                '                and (is_sep(i - 1) or tokens[i - 1][0] in ("open", "comma"))\n',
                "                and is_sep(i - 1)\n",
            )
        ],
        [
            f"{UNIT}::test_windows_full_path_normalization",
            _scenario("builtin-search-find-components-alt-start-doublestar"),
        ],
        "R002: `src/{**/b.spec.ts,none}` misses src/b.spec.ts on Windows",
    ),
    "adjacent_components_not_collapsed": Control(
        [
            (
                FIND,
                "                while j + 2 <= hi and is_double_star(j) and is_sep(j + 2):\n                    j += 3\n",
                "",
            )
        ],
        [
            f"{UNIT}::test_windows_full_path_normalization",
            _scenario("builtin-search-find-components-adjacent-doublestar"),
        ],
        "R002: `src/**/**/*.spec.ts` loses a branch on Windows",
    ),
    "empty_alternative_zero_form": Control(
        [
            (
                FIND,
                '                    out[-1:] = [f"{{{sep},{sep}**{sep}}}"]\n',
                '                    out.append(f"{{,**{sep}}}")\n',
            )
        ],
        [
            f"{UNIT}::test_windows_full_path_normalization",
            _scenario("builtin-search-find-components-brace-alternative-doublestar"),
        ],
        "R002: the empty alternative never matches, so the direct file is missed",
    ),
    "lex_original_pattern": Control(
        [
            (
                FIND,
                "    tokens = _lex(_pi_windows_rewrite(pattern))\n",
                '    tokens = [("class", _SEPARATOR_CLASS) if v == "/" else (k, v.replace("/", _SEPARATOR_CLASS)) for k, v in _lex(pattern)]\n',
            )
        ],
        [
            f"{UNIT}::test_windows_full_path_normalization",
            _scenario("builtin-search-find-components-class-reshaped-negated"),
        ],
        "R002: a class Pi's rewrite reshapes (`src/[!]/**/...`) is corrected instead of keeping Pi's Windows meaning",
    ),
    "diagnostics_from_corrected_text": Control(
        [
            (
                FIND,
                "        run, lines, rejected = await run_fd(effective, effective != pi_pattern)\n",
                "        run, lines, rejected = await run_fd(effective, False)\n",
            )
        ],
        [
            f"{UNIT}::test_a_rejected_corrected_pattern_reports_pis_diagnostic",
            _scenario("builtin-search-find-components-rejected-invalid-range"),
        ],
        "R002 rule 5: fd's diagnostic shows the generated pattern text instead of Pi's",
    ),
    "literal_brace_wrapping": Control(
        [
            (
                FIND,
                "        tokens.append((kind, char))\n",
                '        tokens.append((kind, "[}]" if char == "}" and kind == "lit" else char))\n',
            )
        ],
        [_scenario("builtin-search-find-components-rejected-unopened-brace")],
        "R002: an unmatched `}` made literal turns Pi's `unopened alternate group` error into a success",
        ("win32",),
    ),
}


def _command() -> list[str]:
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rfE", "--tb=short"]
    strict = ["-W", "error::pytest.PytestUnraisableExceptionWarning"]
    if os.environ.get("E5_BARE_PYTEST"):
        return [
            *base,
            "-c",
            os.devnull,
            f"--rootdir={ROOT}",
            "-o",
            "asyncio_mode=auto",
            *strict,
            *TESTS,
        ]
    return [*base, "--no-cov", *strict, *TESTS]


_SUMMARY = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$")


def run(name: str) -> dict[str, object]:
    control = CONTROLS[name]
    if HOST not in control.platforms:
        return {
            "name": name,
            "not_applicable": f"witness exists on {', '.join(control.platforms)} only",
        }
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        shutil.copytree(ROOT / "src", src, ignore=shutil.ignore_patterns("__pycache__"))
        texts: dict[Path, str] = {}
        for target, anchor, replacement in control.edits:
            path = src / target
            text = texts.get(path) or path.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if text.count(anchor) != 1:
                return {
                    "name": name,
                    "error": f"anchor matched {text.count(anchor)} times in {target}",
                }
            texts[path] = text.replace(anchor, replacement)
        for path, text in texts.items():
            path.write_bytes(text.encode("utf-8"))
        env = {**os.environ, "PYTHONPATH": str(src), "COLUMNS": "4000"}  # untruncated -r reasons
        try:
            result = subprocess.run(
                _command(), cwd=ROOT, env=env, capture_output=True, text=True, timeout=900
            )
        except subprocess.TimeoutExpired:
            return {"name": name, "error": "hung (timeout); not a kill"}
    lines = result.stdout.splitlines()
    failures = {m.group(2): (m.group(1), m.group(3) or "") for m in map(_SUMMARY.match, lines) if m}
    errors = sorted(
        node for node, (kind, _) in failures.items() if kind == "ERROR" or "::" not in node
    )
    killing = {
        node: failures[node][1]
        for node in control.witnesses
        if failures.get(node, ("",))[0] == "FAILED"
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
        print(json.dumps({n: {"witnesses": c.witnesses, "expected": c.expected, "platforms": c.platforms}
                          for n, c in CONTROLS.items()}, indent=1))  # fmt: skip
        return 0
    if not os.environ.get("MINION_SEARCH_ENGINE_ARTIFACTS"):
        print(
            "MINION_SEARCH_ENGINE_ARTIFACTS must name the pinned engine artifacts", file=sys.stderr
        )
        return 2
    results = [run(name) for name in (argv or list(CONTROLS))]
    print(json.dumps({"os": HOST, "results": results}, indent=1, ensure_ascii=False))
    return 0 if all(r.get("killed") or "not_applicable" in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
