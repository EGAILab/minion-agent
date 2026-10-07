"""Rust search mutations in a disposable scratch tree; compile failures are NOT kills.

Requires the same pinned ICU/artifact environment as the positive gates. TEMP/TMP
must point to task scratch (E: on Windows). Candidate source is never modified.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CONTROLS = {
    "retry-closes-first-window": ("find.rs", "                !retry", "                true", "diagnostic_rerun_abort_during_wait_settles_before_stop_ack"),
    "retry-without-signal": ("find.rs", "            request.signal.clone(),", "            if first {request.signal.clone()} else {None},", "diagnostic_rerun_abort_during_wait_settles_before_stop_ack"),
    "retry-skips-between-abort": ("find.rs", "        if request.signal.as_ref().is_some_and(|s| s.is_cancelled()) {", "        if false {", "diagnostic_rerun_abort_between_runs_prevents_spawn"),
    "retry-awaits-aborted-stop-ack": ("find.rs", "window.aborted.load(Ordering::SeqCst)\n                        || (window.active", "false\n                        || (window.active", "diagnostic_rerun_abort_during_wait_settles_before_stop_ack"),
    "unc-share-forced-absolute": ("search_paths.rs", "common < 2", "common < 3", "logical_surrogate_root_is_not_its_native_replacement"),
    "automatic-call-time-provisioning": ("search_engines.rs", "if world != &ExecutionWorldIdentity::local() || !self.verified(engine, pin).await {", "let artifacts=std::env::var_os(\"MINION_SEARCH_ENGINE_ARTIFACTS\").unwrap(); let _=provision_search_engines(self,Some(Path::new(&artifacts))).await; if world != &ExecutionWorldIdentity::local() || !self.verified(engine, pin).await {", "managed_store_verifies_every_use_and_provisioning_is_idempotent"),
    "decision-after-disposal": ("search_run.rs", "    result\n}", "    result.map(|mut outcome| {outcome.aborted=signal.as_ref().is_some_and(|s|s.is_cancelled());outcome})\n}", "real_factories_reproduce_the_fourteen_pi_abort_partition_cells"),
    "adjacent-recursive-not-collapsed": ("search_glob.rs", "while tokens.get(i) == Some(&Token::Stars(2)) && tokens.get(i + 1) == Some(&Token::Sep)", "while false && tokens.get(i) == Some(&Token::Stars(2)) && tokens.get(i + 1) == Some(&Token::Sep)", "canonical:builtin-search-find-components-adjacent-doublestar"),
    "empty-alternative-zero-form": ("search_glob.rs", 'format!("{{{SEP},{SEP}**{SEP}}}")', 'format!("{SEP}{{,**{SEP}}}")', "canonical:builtin-search-find-components-brace-alternative-doublestar"),
    "lex-original-pattern": ("search_glob.rs", "rewrite(&lex(&pi), false)", "{let _=pi;rewrite(&lex(pattern),false)}", "canonical:builtin-search-find-components-brace-alternative-doublestar"),
    "literal-brace-wrapping": ("search_glob.rs", "rewrite(&lex(&pi), false)", 'format!("{{{}}}",rewrite(&lex(&pi),false))', "separators_classes_and_alternation_context"),
    "whole-pattern-windows-conversion-on-linux": ("find.rs", "if platform == Platform::Windows {\n            effective", "if true {\n            effective", "canonical:builtin-search-find-components-adjacent-doublestar"),
    "context-windows-merged": ("grep.rs", "for m in matches {", "let mut previous_end=std::collections::BTreeMap::<String,f64>::new(); for m in matches { if context>0.0 {let start=js_max(1.0,m.line-context);let end=m.line+context;if previous_end.get(&m.file).is_some_and(|old|start<=*old){continue;} previous_end.insert(m.file.clone(),end);}", "canonical:builtin-search-grep-plain-context-overlap"),
    "find-abort-result-lost": ("find.rs", 'if outcome.aborted {', 'if false {', "real_factories_reproduce_the_fourteen_pi_abort_partition_cells"),
    "grep-abort-result-lost": ("grep.rs", 'if outcome.aborted {', 'if false {', "real_factories_reproduce_the_fourteen_pi_abort_partition_cells"),
    "logical-path-projected-too-early": ("search_paths.rs", 'let left = components(base.code_units(), platform);', 'let left = components(&String::from_utf16_lossy(base.code_units()).encode_utf16().collect::<Vec<_>>(), platform);', "logical_surrogate_root_is_not_its_native_replacement"),
    "following-windows-git-probe": ("find.rs", 'fs.file_info(&git, None).await.is_ok()', 'fs.probe_dir_entry(&git, None).await.is_ok()', "canonical:builtin-search-find-junction-dangling-git-junction"),
    "fixed-name-staging": ("search_engines.rs", '.join(format!(".{}.install", uuid::Uuid::new_v4()))', '.join(SearchEngine::Fd.executable())', "staging_cannot_publish_a_partial_binary_at_the_fixed_name"),
    "rg-no-require-git": ("grep.rs", '"--json".into(),', '"--json".into(), "--no-require-git".into(),', "grep_counts_uncollected_matches_without_reordering"),
    "literal-flag-omitted": ("grep.rs", 'argv.push("--fixed-strings".into());', '', "canonical:builtin-search-grep-repo-literal-dot"),
    "ignore-case-omitted": ("grep.rs", 'argv.push("--ignore-case".into());', '', "canonical:builtin-search-grep-repo-ignore-case"),
    "grep-hidden-omitted": ("grep.rs", '"--hidden".into(),', '', "canonical:builtin-search-grep-repo-hidden"),
    "cr-not-readline-delimiter": ("search_run.rs", "ch == '\\r' || ch == '\\n'", "ch == '\\n'", "readline_handles_cr_and_utf8_chunks_without_stripping_bom"),
    "brace-alt-start-not-recursive": ("search_glob.rs", 'parts.push(rewrite(&tokens[begin..end], true));', 'parts.push(rewrite(&tokens[begin..end], false));', "separators_classes_and_alternation_context"),
    "integer-find-limit": ("find.rs", 'number_to_string(limit)', 'number_to_string(limit.floor())', "canonical:builtin-search-find-repo-limit-fraction"),
    "integer-grep-limit": ("grep.rs", 'count as f64 >= limit', 'count as f64 >= limit.floor()', "canonical:builtin-search-grep-repo-limit-fraction"),
    "integer-context": ("grep.rs", 'm.line - context', 'm.line - context.floor()', "canonical:builtin-search-grep-repo-context-fraction"),
    "context-bom-dropped": ("grep.rs", 'String::from_utf8_lossy(&data)', 'String::from_utf8_lossy(&data).trim_start_matches(\'\\u{feff}\')', "canonical:builtin-search-grep-plain-bom-context"),
    "fd-no-require-git-omitted": ("find.rs", 'args.push("--no-require-git".into());', '', "find_stream_order_duplicates_and_untrimmed_empty_decision"),
    "abort-not-observed": ("search_run.rs", "&& (!preexisting||listen_to_preexisting_abort)", "&& false && (!preexisting||listen_to_preexisting_abort)", "every_exit_and_eof_abort_is_inside_the_window_but_close_is_outside"),
    "grep-preexisting-abort-observed": ("search_run.rs", "(!preexisting||listen_to_preexisting_abort)", "(!preexisting||listen_to_preexisting_abort||true)", "every_exit_and_eof_abort_is_inside_the_window_but_close_is_outside"),
    "exit-only-completion": ("search_run.rs", "out_eof && err_eof && let Some(code) = exit", "let Some(code) = exit", "completion_requires_exit_and_both_eofs"),
    "stop-ack-before-completion": ("search_run.rs", "// Settle the abort window BEFORE cleanup or a pending terminate acknowledgement.", "if let Some(task)=stop.take() {let _=task.await;} // Wrong: joins before completion.", "held_limit_stop_acknowledgement_does_not_extend_completion"),
    "per-chunk-decoder": ("search_run.rs", "self.pending.extend_from_slice(bytes);", "self.pending.clear(); self.pending.extend_from_slice(bytes);", "readline_handles_cr_and_utf8_chunks_without_stripping_bom"),
    "strip-bom": ("search_run.rs", "for ch in text.chars() {", "for ch in text.trim_start_matches('\\u{feff}').chars() {", "readline_handles_cr_and_utf8_chunks_without_stripping_bom"),
    "utf16-cut-off-by-one": ("search_text.rs", "units.truncate(500);", "units.truncate(501);", "surrogate_cut_and_empty_body_do_not_lossily_serialize"),
    "lossy-result": ("search_text.rs", "ResultString::from_code_units(text)", "ResultString::from(String::from_utf16_lossy(&text))", "surrogate_cut_and_empty_body_do_not_lossily_serialize"),
    "case-sensitive-windows-relative": ("search_paths.rs", "super::search_node_lower::lower(left) == super::search_node_lower::lower(right)", "left == right", "windows_relative_is_case_insensitive_not_prefix_matching"),
    "host-unicode-lowercase": ("search_node_lower.rs", "let points: Vec<u32>", "return String::from_utf16_lossy(units).to_lowercase().encode_utf16().collect(); #[allow(unreachable_code)] let points: Vec<u32>", "node_unicode16_lowercase_is_not_host_unicode17"),
    "sort-find-output": ("find.rs", "Ok(search_text::finish(", "let mut entries=entries; entries.sort(); Ok(search_text::finish(", "find_stream_order_duplicates_and_untrimmed_empty_decision"),
    "deduplicate-find-output": ("find.rs", "Ok(search_text::finish(", "let mut entries=entries; entries.dedup(); Ok(search_text::finish(", "find_stream_order_duplicates_and_untrimmed_empty_decision"),
    "trim-before-empty-decision": ("find.rs", 'if lines.join("\\n").is_empty() {', 'if js_trim(&lines.join("\\n")).is_empty() {', "find_stream_order_duplicates_and_untrimmed_empty_decision"),
    "omit-find-trim": ("find.rs", "let line = js_trim(line.strip_suffix('\\r').unwrap_or(line));", "let line = line.strip_suffix('\\r').unwrap_or(line);", "find_stream_order_duplicates_and_untrimmed_empty_decision"),
    "rerun-not-reverified": ("find.rs", "engines\n                .resolve(SearchEngine::Fd, subprocess.execution_world())\n                .await?", "fd.clone()", "diagnostic_rerun_is_verified_again_and_preserves_pi_text"),
    "diagnostic-is-generated-pattern": ("find.rs", "if first { effective.clone() } else { pi.clone() }", "effective.clone()", "diagnostic_rerun_is_verified_again_and_preserves_pi_text"),
    "count-only-collected-matches": ("grep.rs", "count += 1;", "if event.get(\"data\").and_then(|v|v.get(\"path\")).and_then(|v|v.get(\"text\")).is_some() {count+=1;}", "grep_counts_uncollected_matches_without_reordering"),
    "collected-output-sorted": ("grep.rs", "for m in matches {", "matches.sort_by(|a,b|a.file.cmp(&b.file)); for m in matches {", "grep_counts_uncollected_matches_without_reordering"),
    "store-existence-is-verification": ("search_engines.rs", "hash(&bytes) == pin.binary_sha256", "{let _=(bytes,pin);true}", "managed_store_verifies_every_use_and_provisioning_is_idempotent"),
    "artifact-not-verified": ("search_engines.rs", "if hash(&artifact) != pin.artifact_sha256 {", "if false {", "managed_store_verifies_every_use_and_provisioning_is_idempotent"),
    "missing-store-fallback": ("search_engines.rs", "return Err(engine.unavailable());", "return Ok(PathBuf::from(engine.executable()));", "managed_store_verifies_every_use_and_provisioning_is_idempotent"),
    "recursive-composition-removed": ("search_glob.rs", "rewrite(&lex(&pi), false)", "{let _=lex(&pi);pi}", "separators_classes_and_alternation_context"),
}

def main():
    workspace = Path(__file__).resolve().parents[1]
    selected = sys.argv[1:] or list(CONTROLS)
    with tempfile.TemporaryDirectory(prefix="minion-search-controls-") as temp:
        root = Path(temp)
        scratch = root / "minion-agent-rust"
        shutil.copytree(workspace, scratch, ignore=shutil.ignore_patterns("target", ".git"))
        shutil.copytree(workspace.parent / "conformance", root / "conformance")
        oracle = Path("minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt")
        (root / oracle).parent.mkdir(parents=True)
        shutil.copyfile(workspace.parent / oracle, root / oracle)
        source = scratch / "crates/minion-agent/src/tools/builtin"
        originals = {file: (source / file).read_text(encoding="utf-8") for file, *_ in CONTROLS.values()}
        results = []
        for name in selected:
            windows_only = {"following-windows-git-probe", "adjacent-recursive-not-collapsed", "empty-alternative-zero-form", "lex-original-pattern"}
            if (name in windows_only and sys.platform!="win32") or (name=="whole-pattern-windows-conversion-on-linux" and sys.platform=="win32"):
                print(json.dumps({"control":name,"not_applicable":"platform-specific production branch"}),flush=True)
                continue
            for file, original in originals.items():
                (source / file).write_text(original, encoding="utf-8", newline="\n")
            file, old, new, witness = CONTROLS[name]
            if old not in originals[file]:
                raise SystemExit(f"MISSING_ANCHOR: {name}: {old!r}")
            (source / file).write_text(originals[file].replace(old,new),encoding="utf-8",newline="\n")
            canonical = witness.startswith("canonical:")
            target = ["--test","builtin_search_conformance"] if canonical else ["--lib"]
            build = subprocess.run(["cargo","test","--locked","--offline","-p","minion-agent","--all-features",*target,"--no-run","--message-format=json"],cwd=scratch,text=True,encoding="utf-8",errors="replace",capture_output=True)
            events = [json.loads(line) for line in build.stdout.splitlines() if line.startswith("{")]
            executables = [e["executable"] for e in events if e.get("reason")=="compiler-artifact" and e.get("executable") and e.get("profile",{}).get("test")]
            if build.returncode or len(executables)!=1:
                print(build.stderr)
                raise SystemExit(f"BUILD_FAILURE_NOT_A_KILL: {name}")
            env = os.environ.copy()
            test = "canonical_search_real_engines_and_layer_six" if canonical else witness
            if canonical: env["MINION_SEARCH_CASE"] = witness.split(":",1)[1]
            run = subprocess.run([executables[0],test,"--nocapture"],cwd=scratch,env=env,text=True,encoding="utf-8",errors="replace",capture_output=True,timeout=75)
            transcript = run.stdout+run.stderr
            canonical_comparison = canonical and f'search case "{env["MINION_SEARCH_CASE"]}"' in transcript and "unexpected file" in transcript
            killed = run.returncode != 0 and f"{test} ... FAILED" in transcript and ("assertion" in transcript or canonical_comparison or (name in {"stop-ack-before-completion", "retry-awaits-aborted-stop-ack"} and "Elapsed" in transcript) or (name=="trim-before-empty-decision" and "whitespace-only nonempty stdout must be success" in transcript))
            item={"control":name,"witness":witness,"expected":"intended semantic assertion fails (not compilation/setup)","killed":killed,"diagnostic":run.stderr.strip()[:4000]}
            results.append(item)
            print(json.dumps(item),flush=True)
            if not killed:
                print(transcript)
                raise SystemExit(f"SURVIVED_OR_INVALID_FAILURE: {name}")
    print(json.dumps({"controls":results,"killed":len(results)},indent=2))

if __name__ == "__main__": main()
