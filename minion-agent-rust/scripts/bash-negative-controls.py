"""WP-13.3 single-point controls. Isolated sources; build failures never count.

Use the normal Cargo/pinned ICU environment. Default selects all controls;
optional names select a subset. Reports JSON, including the intended witness.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

B = "bash.rs"
S = "bash_shell.rs"
O = "bash_output.rs"
E = "environment.rs"

# Each control has an independent, observable assertion in the named witness.
CONTROLS = {
    "abort-before-timeout": (B, 'let seconds = timeout(&request)?;', 'if request.signal.as_ref().is_some_and(|s| s.is_cancelled()) { return Err(ToolCapabilityError::new("Command aborted")); }\n    let seconds = timeout(&request)?;', "bash_precheck_order_timeout_before_abort_before_shell_and_cwd"),
    "timeout-before-abort-classification": (B, 'let suffix = if aborted {', 'let suffix = if aborted && !timed_out {', "bash_timer_is_whole_ms_minimum_one_and_abort_wins_during_grace"),
    "settlement-at-exit-alone": (B, 'if exited && stdout_done && stderr_done {', 'if exited {', "bash_settlement_waits_for_output_resets_grace_and_releases_both_pipes"),
    "no-stdio-idle-grace": (B, 'Duration::from_millis(100)', 'Duration::from_secs(100)', "bash_settlement_waits_for_output_resets_grace_and_releases_both_pipes"),
    "command-keeps-pipes": (B, 'bash_shell::close(&process).await;', '', "bash_settlement_waits_for_output_resets_grace_and_releases_both_pipes"),
    "decoder-reset-per-chunk": (O, 'self.decode(&bytes, false);', 'self.pending.clear(); self.decode(&bytes, false);', "merged_decoder_bom_eof_and_raw_bytes"),
    "per-chunk-bom-strip": (O, 'self.decode(&bytes, false);', 'self.decoded_started = false; self.decode(&bytes, false);', "merged_decoder_bom_eof_and_raw_bytes"),
    "no-bom-strip": (O, "if text.starts_with('\\u{feff}') {", "if false && text.starts_with('\\u{feff}') {", "merged_decoder_bom_eof_and_raw_bytes"),
    "bom-prefix-lost-at-eof": (O, "} else if finish {\n                        text.push('\\u{fffd}');", "} else if finish {\n                        if self.pending[0] != 0xef { text.push('\\u{fffd}'); }", "merged_decoder_bom_eof_and_raw_bytes"),
    "decoded-instead-of-raw-file": (B, '&chunk,', 'String::from_utf8_lossy(&chunk).as_bytes(),', "bash_persistence_join_freezes_timeout_and_abort_and_preserves_raw_bytes"),
    "truncation-from-tail": (O, 'let truncated = self.lines() > MAX_LINES || self.decoded_bytes > MAX_BYTES;', 'let truncated = tail_by.is_some();', "pinned_boundary_probe_all_ten_rolling_rows"),
    "persistence-not-joined": (B, 'let path = match writer.await {', 'let path = match Ok::<_, ToolCapabilityError>(Some("full.log".to_owned())) {', "bash_persistence_join_freezes_timeout_and_abort_and_preserves_raw_bytes"),
    "abort-after-finalization": (B, 'let aborted = request.signal.as_ref().is_some_and(|s| s.is_cancelled());', 'let aborted = false;', "bash_timer_is_whole_ms_minimum_one_and_abort_wins_during_grace"),
    "writer-orphaned-on-cancel": (B, 'let writer = persist(fs, receiver);', 'let writer = async { tokio::spawn(persist(fs, receiver)).await.unwrap() };', "bash_cancel_drops_the_writer_future_and_no_write_outlives_the_call"),
    "always-drop-partial-line": (O, '.map_or(self.tail.as_str(), |(_, rest)| rest)', '.map_or("", |(_, rest)| rest)', "pinned_boundary_probe_all_ten_rolling_rows"),
    "never-drop-partial-line": (O, 'if self.starts_at_boundary {', 'if true {', "pinned_boundary_probe_all_ten_rolling_rows"),
    "timer-rounded-ms": (B, '(seconds * 1000.0).trunc().max(1.0)', '(seconds * 1000.0).round().max(1.0)', "bash_timer_is_whole_ms_minimum_one_and_abort_wins_during_grace"),
    "timer-no-one-ms-minimum": (B, '(seconds * 1000.0).trunc().max(1.0)', '(seconds * 1000.0).trunc()', "bash_timer_is_whole_ms_minimum_one_and_abort_wins_during_grace"),
    "lookup-unbounded": (S, 'total > 1048576', 'false && total > 1048576', "bash_lookup_combined_budget_keeps_exited_status_and_kills_only_unexited"),
    "lookup-per-stream-budget": (S, 'total > 1048576', 'stdout_total > 1048576 || stderr_total > 1048576', "bash_lookup_combined_budget_keeps_exited_status_and_kills_only_unexited"),
    "lookup-settles-at-exit": (S, 'if exited && stdout_ended && stderr_ended {', 'if exited {', "bash_lookup_waits_for_exit_and_both_eof_not_command_grace"),
    "lookup-command-idle-grace": (S, 'Duration::from_millis(5000)', 'Duration::from_millis(100)', "bash_lookup_waits_for_exit_and_both_eof_not_command_grace"),
    "lookup-fails-on-interruption": (S, 'if interrupted {', 'if interrupted { code = None;', "bash_lookup_combined_budget_keeps_exited_status_and_kills_only_unexited"),
    "lookup-keeps-pipes": (S, 'close(&process).await;', '', "bash_lookup_timeout_after_exit_keeps_zero_and_releases_pipes"),
    "existence-via-canonical": (S, '.probe_dir_entry(&FsPath::from(path), None)', '.canonical_path(&FsPath::from(path), None)', "bash_empty_override_falls_through_custom_missing_fails_and_windows_lookup_probes"),
    "windows-cwd-follows": (S, 'fs.file_info(&path, None).await.map(|_| ())', 'fs.probe_dir_entry(&path, None).await.map(|_| ())', "bash_windows_probe_selection_and_nonfollowing_cwd_are_distinct"),
    "inject-none-for-absent-file": (B, 'if let Some(file) = context.session_file() {', 'if let Some(file) = Some(context.session_file().unwrap_or("none")) {', "bash_context_env_is_explicit_clean_and_provider_snapshot_is_per_call"),
    "legacy-shell-uses-argv": (S, 'let stdin = bytes.len() > 2', 'let stdin = false && bytes.len() > 2', "bash_legacy_stdin_does_not_block_timeout_and_command_uses_usv_projection"),
    "spawn-error-raw-text": (B, 'format!("Failed to start the shell {}", shell.shell)', 'format!("spawn failure {}", shell.shell)', "bash_spawn_failure_has_own_text_and_zero_updates"),
    "late-pre-spawn-abort-is-spawn-error": (B, 'if error.code == SubprocessErrorCode::Aborted {', 'if false && error.code == SubprocessErrorCode::Aborted {', "bash_abort_arriving_during_discovery_is_not_a_spawn_failure"),
    "file-error-does-not-terminate": (B, 'process.terminate().await;', '', "bash_file_failure_kills_before_own_error_and_discards_output"),
    "unexpected-partial-update": (B, 'let formatted = output.snapshot(path.as_deref(), empty);', 'let formatted = output.snapshot(path.as_deref(), empty);\n if let Some(update) = &request.on_update { update(text_result("partial", serde_json::json!({}))); }', "bash_spawn_failure_has_own_text_and_zero_updates"),
    "command-lone-surrogate-dropped": (B, 'Some(PreparedValue::String(s)) => s.to_utf8_lossy(),', 'Some(PreparedValue::String(s)) => String::from_utf16(s.code_units()).unwrap_or_default(),', "bash_command_projection_argv_and_stdin_preserves_pairs_replaces_lone_units"),
    "raw-threshold-ignored": (O, 'self.raw_bytes > MAX_BYTES || self.decoded_bytes > MAX_BYTES || self.lines() > MAX_LINES', 'self.decoded_bytes > MAX_BYTES || self.lines() > MAX_LINES', "raw_threshold_opens_a_log_even_when_bom_stripping_avoids_truncation"),
    "timer-raw-fractional-ms": (B, 'Duration::from_millis((seconds * 1000.0).trunc().max(1.0) as u64)', 'Duration::from_secs_f64(seconds.max(0.001))', "bash_timer_is_whole_ms_minimum_one_and_abort_wins_during_grace"),
    "disabled-context-still-injected": (B, 'if options.expose_session_environment', 'if true', "bash_disabled_context_is_removed_signal_delegated_and_worlds_checked"),
    "signal-not-delegated": (B, 'self.0.is_cancelled()', 'false', "bash_disabled_context_is_removed_signal_delegated_and_worlds_checked"),
    "pipe-release-kills-descendants": (B, 'bash_shell::close(&process).await;', 'process.terminate().await; bash_shell::close(&process).await;', "bash_settlement_waits_for_output_resets_grace_and_releases_both_pipes"),
    "lookup-crossing-chunk-discarded": (S, 'bytes.extend(chunk);', 'if total <= 1048576 { bytes.extend(chunk); }', "bash_lookup_retains_the_budget_crossing_stdout_chunk"),
    "unsupported-prerequisite-fabricated": (S, 'Err(error) if error.code == FsErrorCode::NotSupported', 'Err(error) if false && error.code == FsErrorCode::NotSupported', "bash_append_failure_uses_own_error_and_prerequisites_are_not_fabricated"),
    "baseline-recaptured": (B, 'let snapshot = subprocess.base_env();', 'let _unused = subprocess.base_env(); let snapshot = subprocess.base_env();', "bash_precheck_order_timeout_before_abort_before_shell_and_cwd"),
    "timer-active-through-finalization": (B, 'let path = match writer.await {', 'let path = match tokio::select! { result = &mut writer => result, () = async { if let Some(deadline)=deadline {tokio::time::sleep_until(deadline).await} else {std::future::pending().await} } => Err(ToolCapabilityError::new("finalization timeout")) } {', "bash_persistence_join_freezes_timeout_and_abort_and_preserves_raw_bytes"),
    "abort-classified-after-finalization": (B, 'let aborted = request.signal.as_ref().is_some_and(|s| s.is_cancelled());', 'let aborted = false;', "bash_persistence_join_freezes_timeout_and_abort_and_preserves_raw_bytes"),
    "persistence-inline-in-intake": (B, 'let _ = sender.send(writes);', 'let _ = sender.send(writes); let _ = (&mut writer).await;', "bash_persistence_join_freezes_timeout_and_abort_and_preserves_raw_bytes"),
    "decoder-per-stream": (O, 'pending: Vec<u8>,', 'pub(super) pending: Vec<u8>,', "bash_stdout_stderr_share_one_decoder_in_read_completion_order"),
    "empty-nonzero-output": (B, 'if aborted || timed_out {', 'if aborted || timed_out || exit_code.is_some_and(|code| code != 0) {', "bash_nonzero_empty_output_and_notice_precede_the_status"),
    "status-before-truncation-notice": (B, 'format!("{}\\n\\n{suffix}", formatted.text)', 'format!("{suffix}\\n\\n{}", formatted.text)', "bash_nonzero_empty_output_and_notice_precede_the_status"),
    "case-insensitive-minion-removal": (E, 'env.remove(*name);', 'env.remove(&name.to_lowercase()); env.remove(*name);', "bash_context_env_is_explicit_clean_and_provider_snapshot_is_per_call"),
    "lookup-nonzero-is-success": (S, 'if code != Some(0) || bytes.is_empty() {', 'if bytes.is_empty() {', "bash_lookup_selection_uses_real_status_truthiness_and_buffer_utf8"),
}

def main():
    workspace = Path(__file__).resolve().parents[1]
    selected = sys.argv[1:] or list(CONTROLS)
    unknown = set(selected) - CONTROLS.keys()
    if unknown:
        raise SystemExit(f"Unknown controls: {unknown}")
    result = []
    with tempfile.TemporaryDirectory(prefix="minion-bash-controls-") as directory:
        root = Path(directory)
        scratch = root / "minion-agent-rust"
        shutil.copytree(workspace, scratch, ignore=shutil.ignore_patterns("target", ".git"))
        shutil.copytree(workspace.parent / "conformance", root / "conformance")
        oracle = Path("minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt")
        (root / oracle).parent.mkdir(parents=True)
        shutil.copyfile(workspace.parent / oracle, root / oracle)
        source = scratch / "crates/minion-agent/src/tools/builtin"
        originals = {name: (source / name).read_text(encoding="utf-8") for name in (B,S,O,E)}
        # Remove any executable from another worktree once. Every following
        # mutation rewrites tracked Rust source in this unique scratch tree;
        # Cargo rebuilds it, and its compiler-artifact event identifies the
        # exact executable to run. Keep incremental compilation between mutants.
        subprocess.run(["cargo","clean","-p","minion-agent"],cwd=scratch,check=True,stdout=subprocess.DEVNULL)
        for name in selected:
            for file, original in originals.items():
                (source / file).write_text(original,encoding="utf-8",newline="\n")
            file, old, new, witness = CONTROLS[name]
            original = originals[file]
            if old not in original:
                raise SystemExit(f"Missing source anchor: {name}: {old!r}")
            (source / file).write_text(original.replace(old,new),encoding="utf-8",newline="\n")
            if name == "lookup-per-stream-budget":
                content=(source/S).read_text(encoding="utf-8")
                content=content.replace('let mut total = 0usize;', 'let mut total = 0usize; let mut stdout_total = 0usize; let mut stderr_total = 0usize;')
                content=content.replace('total += chunk.len(); bytes.extend(chunk);', 'total += chunk.len(); stdout_total += chunk.len(); bytes.extend(chunk);')
                content=content.replace('total += chunk.len(); } else { stderr_ended', 'total += chunk.len(); stderr_total += chunk.len(); } else { stderr_ended')
                (source/S).write_text(content,encoding="utf-8",newline="\n")
            if name == "abort-classified-after-finalization":
                content=(source/B).read_text(encoding="utf-8").replace('let suffix = if aborted {','let suffix = if request.signal.as_ref().is_some_and(|s| s.is_cancelled()) {')
                (source/B).write_text(content,encoding="utf-8",newline="\n")
            if name == "decoder-per-stream":
                content=(source/B).read_text(encoding="utf-8")
                anchor='chunk = bash_shell::read(&stderr), if !stderr_done => {\n                if let Some(bytes) = chunk {\n                    let writes = output.append(bytes);'
                assert anchor in content
                (source/B).write_text(content.replace(anchor,anchor.replace('let writes = output.append(bytes);','output.pending.clear(); let writes = output.append(bytes);')),encoding="utf-8",newline="\n")
            build = subprocess.run(["cargo","test","--locked","--offline","-p","minion-agent","--all-features","--lib","--no-run","--message-format=json"],cwd=scratch,text=True,capture_output=True)
            artifacts = [json.loads(line) for line in build.stdout.splitlines() if line.startswith('{')]
            executables = [event["executable"] for event in artifacts if event.get("reason")=="compiler-artifact" and event.get("executable") and event.get("profile",{}).get("test")]
            if build.returncode or len(executables)!=1:
                print(build.stderr)
                raise SystemExit(f"BUILD_FAILURE_NOT_A_KILL: {name}")
            run = subprocess.run([executables[0],witness,"--nocapture"],cwd=scratch,text=True,capture_output=True,timeout=60)
            transcript = run.stdout + run.stderr
            # A test-level assertion at the intended witness is required.
            markers=("assertion", "not joined:", "settled success must", "settlement must", "total output requires truncation metadata", "provider probe success must", "released persistence must")
            killed = run.returncode!=0 and f"::{witness} ... FAILED" in transcript and any(marker in transcript for marker in markers)
            item={"control":name,"witness":witness,"killed":killed,"test_exit":run.returncode,"diagnostic":run.stderr.strip()[:3000]}
            result.append(item)
            print(json.dumps(item),flush=True)
            if not killed:
                print(transcript)
                raise SystemExit(f"SURVIVED_OR_INVALID_FAILURE: {name}")
    print(json.dumps({"controls":result,"killed":len(result)},indent=2))

if __name__=="__main__":
    main()
