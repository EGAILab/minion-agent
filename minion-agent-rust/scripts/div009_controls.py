"""One production-rule mutation per DIV009 witness; used by the fired-control runner."""
EXEC = 'crates/minion-agent/src/execution/'
TEST = 'execution::div009_tests::'


def controls():
    mapping = 'map_fs_error(e).with_path(raw)'
    fs_test = TEST + 'every_fs_consumer_preserves_original_error_and_raw_path'
    panic_test = TEST + 'failure_is_a_result_not_a_panic_and_unrelated_panics_stay_panics'
    nul = 'Err(map_path_error(error, rejected).with_path(&logical))'
    rows = [
        ('D013-K1', 'filesystem.rs', mapping,
         '{ FIRE panic!("LEAKED NATIVE CWD ERROR {e}") }',
         panic_test, 'DIV009 must return a typed result, not leak/panic'),
        ('D013-K2', 'filesystem.rs', mapping,
         '{ FIRE let mut error = map_fs_error(e).with_path(raw); error.code = FsErrorCode::Unknown; error }', fs_test, 'left: Unknown'),
        ('D013-K3', 'filesystem.rs', mapping,
         '{ FIRE map_fs_error(e) }', fs_test, 'left: None'),
        ('D013-K4', 'filesystem.rs', mapping,
         '{ FIRE map_fs_error(e).with_path(from_native(&self.cwd)) }', fs_test, 'absolute'),
        ('D013-K5', 'path.rs', 'let mut expanded = raw.clone();',
         'FIRE let _ = inputs.cwd_view()?; let mut expanded = raw.clone();',
         TEST + 'raw_logical_and_public_read_domains', 'cwd count'),
        ('D013-K6', 'path.rs', '|| inputs.cwd_view(),',
         '|| { FIRE std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| inputs.cwd_view())).unwrap_or_else(|_| Err(std::io::Error::other("caught unrelated panic"))) },',
         panic_test, 'unrelated reader panic must propagate'),
        ('D013-K7', 'path.rs', '|| inputs.cwd_view(),',
         '|| { FIRE Ok(inputs.cwd_view().unwrap_or_else(|_| if cfg!(windows) { "C:\\\\" } else { "/" }.into())) },',
         panic_test, 'native read failure may not fabricate a cwd'),
        ('D013-K8', 'subprocess.rs', 'self.resolved_cwd(options.cwd.as_deref())\n                .map_err(map_spawn_error)?,',
         'self.resolved_cwd(options.cwd.as_deref()).unwrap_or_else(|_| { FIRE self.cwd.clone() }),',
         TEST + 'shell_and_subprocess_read_domains', 'native spawn count'),
        ('D013-K9', 'filesystem.rs',
         "let logical = self.resolved(source)?;\n        // Pi's pre-call abort uses the destination; caught OS errors use source.\n        let logical_destination = self.resolved(destination)?;\n        Self::aborted(signal, &logical_destination)?;",
         'FIRE let logical_destination = self.resolved(destination)?; Self::aborted(signal, &logical_destination)?; let logical = self.resolved(source)?;',
         TEST + 'rename_precedence_and_nul_slots', 'assertion `left == right` failed'),
        ('D013-K10', 'filesystem.rs', 'let logical = self.resolved(path)?;\n        match self.info_for(&logical).await {',
         'let result = match self.resolved(path) { Ok(logical) => self.info_for(&logical).await, Err(e) => { FIRE Err(e) } }; match result {', fs_test, 'exists: ()'),
        ('D013-K11', 'filesystem.rs', 'let canonical_logical = self.resolved(path)?;\n        let key = match Self::canonical_resolved(&canonical_logical).await {',
         'let result = match self.resolved(path) { Ok(logical) => Self::canonical_resolved(&logical).await, Err(e) => { FIRE Err(e) } }; let key = match result {',
         TEST + 'resolve_second_read_failure_never_uses_the_saved_absolute_fallback', 'resolve must not fall back'),
        ('D013-K12', 'filesystem.rs', nul,
         'FIRE let _ = self.resolved(source)?; let _ = self.resolved(destination)?; ' + nul,
         TEST + 'rename_precedence_and_nul_slots', 'left: NotFound'),
        ('D013-K13', 'filesystem.rs', nul,
         'FIRE Err(map_path_error(error, rejected).with_path(if source == destination { &logical_destination } else { &logical }))',
         TEST + 'nul_slots_keep_first_resolution', 'source slot owns fallback'),
        ('D013-K13-raw', 'filesystem.rs', nul,
         'FIRE Err(map_path_error(error, rejected).with_path(if source == destination { &logical_destination } else { &logical }))',
         TEST + 'nul_slots_keep_raw_first_resolution', 'source slot owns fallback'),
    ]
    return [(name, EXEC + file, old, new, witness, reason, False)
            for name, file, old, new, witness, reason in rows]
