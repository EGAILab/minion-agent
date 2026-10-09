"""Compiled, intended-witness controls; only edits an isolated scratch copy.

Run sequentially with the same pinned ICU environment as Cargo gates. A build
error, missing selector, panic abort or infrastructure failure is INVALID, never
a semantic kill. No candidate file is changed. Python is orchestration only.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

CORE = "crates/minion-agent/src/skills/"
ICU = "crates/minion-agent-pinned-icu/src/lib.rs"
CONTROLS = [
    ("declared-priority-removed", CORE+"discovery.rs", 'entry.name != "SKILL.md"', "true", 1, "canonical_skill_discovery", "skills-d02-root-skill-md-short-circuits", False),
    ("nested-markdown-enabled", CORE+"discovery.rs", "&& root_files", "&& true", 1, "canonical_skill_discovery", "skills-d07-root-markdown-only-at-root", False),
    ("raw-collation-replaced-by-lexical", ICU, "indices.sort_by(|&a, &b| collator.strcoll(&names[a], &names[b]));", "indices.sort_by(|&a, &b| names[a].as_string_debug().cmp(&names[b].as_string_debug()));", 1, "raw_name_collation_is_not_lowercased_or_lexical", None, False),
    ("invalid-pattern-aborts-discovery", CORE+"discovery.rs", "if matcher.add_units(&pattern).is_err() {", 'if matcher.add_units(&pattern).is_err() { return Err(SkillDiscoveryError("invalid ignore pattern".into()));', 1, "canonical_skill_discovery", "skills-i10-invalid-patterns-dropped-valid-kept", False),
    ("invalid-path-diagnostic-dropped", CORE+"discovery.rs", 'SkillDiagnosticCode::InvalidPath,\n                "entry path cannot be matched against ignore rules",', 'SkillDiagnosticCode::InvalidMetadata,\n                "entry path cannot be matched against ignore rules",', 1, "invalid_ignore_path_emits_one_diagnostic_and_skips_only_that_entry", None, True),
    ("nesting-off-by-one", CORE+"frontmatter.rs", "const MAX_DEPTH: usize = 64;", "const MAX_DEPTH: usize = 65;", 1, "collection_depth_is_exactly_64_and_root_counts", None, True),
    ("nesting-bound-removed", CORE+"frontmatter.rs", "const MAX_DEPTH: usize = 64;", "const MAX_DEPTH: usize = usize::MAX;", 1, "collection_depth_is_exactly_64_and_root_counts", None, True),
    ("continuation-strips-unicode-whitespace", CORE+"frontmatter.rs", "line.bytes().take_while(|&c| c == b' ').count()", "line.chars().take_while(|c| c.is_whitespace()).map(char::len_utf8).sum()", 1, "canonical_skill_discovery", "skills-f07-unicode-whitespace-in-continuation", False),
    ("unterminated-final-blank-counts", CORE+"frontmatter.rs", "let e = trailing.saturating_sub(usize::from(\n            self.pos == self.lines.len() && !self.terminated && trailing > 0,\n        ));", "let e = trailing;", 1, "canonical_skill_discovery", "skills-f08-block-chomping-at-frontmatter-end", False),
    ("ignore-parent-binding-depth-bound", CORE+"ignore.rs", "pending.push(current.clone());", "if pending.len() > 64 { return Err(()); } pending.push(current.clone());", 1, "ignore_parent_chain_is_stack_independent_and_cache_is_invalidated", None, True),
    ("walk-binding-depth-bound", CORE+"discovery.rs", "let root_files = frame.root_files;", "if entry.path.code_units().iter().filter(|&&u| u == 47 || u == 92).count() > 64 { continue; } let root_files = frame.root_files;", 1, "walk_has_no_binding_depth_bound", None, False),
    ("mapping-called-twice", CORE+"discovery.rs", "for skill in loaded.skills {", "for skill in loaded.skills { let _ = map(skill.clone(), &input.source).map_err(SourcedSkillError::Mapping)?;", 1, "sourced_records_are_writable_and_preserve_opaque_source_identity", None, False),
    ("mapping-failure-swallowed", CORE+"discovery.rs", "skill: map(skill, &input.source).map_err(SourcedSkillError::Mapping)?,", "skill: match map(skill, &input.source) { Ok(value) => value, Err(_) => continue },", 1, "application_mapping_failure_is_not_a_diagnostic", None, False),
    ("filesystem-name-interpolation-is-lossy", CORE+"discovery.rs", "message.extend_from_slice(parent.code_units());", "message.extend(String::from_utf16_lossy(parent.code_units()).encode_utf16());", 1, "filesystem_origin_names_and_diagnostic_interpolation_preserve_units", None, False),
    ("nonascii-to-ascii-fold-allowed", ICU, "(unit >= 128 && first < 128)", "false", 1, "utf16_prefix_and_nonunicode_case_folding_are_lossless", None, True),
    ("ignore-pattern-prefix-is-lossy", CORE+"ignore.rs", "let negative = pattern.first() == Some(&33);", "let projected: Vec<_> = String::from_utf16_lossy(pattern).encode_utf16().collect(); let pattern = projected.as_slice(); let negative = pattern.first() == Some(&33);", 1, "utf16_prefix_and_nonunicode_case_folding_are_lossless", None, True),
    ("ignore-cache-not-invalidated", CORE+"ignore.rs", "self.cache.clear();\n        Ok(true)", "Ok(true)", 1, "ignore_parent_chain_is_stack_independent_and_cache_is_invalidated", None, True),
]

def run(root, control, log):
    name, _, _, _, _, witness, scenario, library = control
    env = os.environ.copy()
    env.pop("WP141_SCENARIO", None)
    if scenario:
        env["WP141_SCENARIO"] = scenario
    command = ["cargo", "test", "-p", "minion-agent"]
    command += ["--lib"] if library else ["--test", "skill_discovery"]
    command += [witness, "--", "--nocapture"]
    result = subprocess.run(command, cwd=root, env=env, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout, encoding="utf-8")
    compiled = "Finished `test` profile" in result.stdout
    failed_witness = re.search(r"^test .*" + re.escape(witness) + r" \.\.\. FAILED$", result.stdout, re.M)
    killed = result.returncode == 101 and compiled and failed_witness and "panicked at" in result.stdout and (not scenario or scenario in result.stdout)
    return result.returncode, bool(killed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Rust workspace")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--only", choices=[c[0] for c in CONTROLS])
    args = parser.parse_args()
    source = args.source.resolve()
    scratch = args.scratch.resolve()
    if os.name == "nt" and scratch.drive.upper() != "E:":
        raise SystemExit("scratch must remain on E:")
    scratch.mkdir(parents=True, exist_ok=True)
    controls = [c for c in CONTROLS if not args.only or c[0] == args.only]
    with tempfile.TemporaryDirectory(prefix="wp141-controls-", dir=scratch) as directory:
        copy = Path(directory)
        root = copy / "minion-agent-rust"
        shutil.copytree(source, root, ignore=shutil.ignore_patterns("target", ".git"))
        shutil.copytree(source.parent / "conformance", copy / "conformance")
        data = copy / "minion-agent-python/tests/skills/data"
        data.mkdir(parents=True)
        for filename in ("frontmatter-corpus.json", "ignore-corpus.json"):
            shutil.copy2(source.parent / "minion-agent-python/tests/skills/data" / filename, data / filename)
        # The existing certified filesystem unit-test binary includes this
        # Node-derived data even when only a skills unit witness is selected.
        shutil.copytree(source.parent / "minion-agent-python/tests/execution/data", copy / "minion-agent-python/tests/execution/data")
        # Cargo can reuse a binary from a previous identical temporary tree,
        # including its compile-time CARGO_MANIFEST_DIR. That path has since
        # been deleted. Rebuild this package once for the new fixture root.
        subprocess.run(["cargo", "clean", "-p", "minion-agent"], cwd=root, check=True)
        invalid = 0
        killed = 0
        for control in controls:
            name, filename, old, new, count, *_ = control
            path = root / filename
            original = path.read_text(encoding="utf-8")
            if original.count(old) != count or old == new:
                print(f"INVALID {name}: anchor count {original.count(old)}, expected {count}", flush=True)
                invalid += 1
                continue
            baseline, _ = run(root, control, scratch / f"{name}-baseline.log")
            if baseline != 0:
                print(f"INVALID {name}: baseline exit {baseline}", flush=True)
                invalid += 1
                continue
            try:
                path.write_text(original.replace(old, new), encoding="utf-8")
                status, valid_kill = run(root, control, scratch / f"{name}-mutant.log")
            finally:
                path.write_text(original, encoding="utf-8")
            print(f"{'KILLED' if valid_kill else 'INVALID'} {name}: mutant exit {status}", flush=True)
            killed += valid_kill
            invalid += not valid_kill
        print(f"{killed}/{len(controls)} valid intended-witness kills; {invalid} invalid/surviving", flush=True)
        return int(invalid != 0)

if __name__ == "__main__":
    raise SystemExit(main())
