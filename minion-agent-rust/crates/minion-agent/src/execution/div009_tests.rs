//! Real-seam, instance-owned DIV-009 witnesses. No global cwd/env mutation.
use super::*;
use parking_lot::Mutex;
use serde_json::Value;
#[cfg(windows)]
use serde_json::json;
use std::{
    collections::VecDeque,
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Clone, Debug)]
enum Read {
    Value(FsPath),
    Error(i32),
    Panic,
}

#[derive(Debug)]
struct Inputs {
    steps: Mutex<VecDeque<Read>>,
    fallback: Read,
    entry: String,
    reads: AtomicUsize,
    entries: Mutex<Vec<String>>,
}

impl Inputs {
    fn new(entry: &str, fallback: Read) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(VecDeque::new()),
            fallback,
            entry: entry.into(),
            reads: AtomicUsize::new(0),
            entries: Mutex::new(Vec::new()),
        })
    }
    fn fail() -> Arc<Self> {
        Self::new("absent", Read::Error(2))
    }
    fn script(steps: Vec<Read>) -> Arc<Self> {
        let inputs = Self::new("absent", Read::Error(2));
        *inputs.steps.lock() = steps.into();
        inputs
    }
    fn count(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }
}

impl path::NativeInputs for Inputs {
    fn cwd_view(&self) -> io::Result<FsPath> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        let step = self
            .steps
            .lock()
            .pop_front()
            .unwrap_or_else(|| self.fallback.clone());
        match step {
            Read::Value(value) => Ok(value),
            Read::Error(code) => Err(io::Error::from_raw_os_error(code)),
            Read::Panic => panic!("UNRELATED READER PANIC"),
        }
    }
    fn drive_view(&self, device: &[u16]) -> Option<FsPath> {
        let device = String::from_utf16(device).unwrap().to_uppercase();
        self.entries.lock().push(format!("={device}"));
        match self.entry.as_str() {
            "absent" => None,
            "empty" => Some("".into()),
            "same" => Some(format!("{device}\\deep").into()),
            "other" => Some(r"X:\other".into()),
            _ => panic!("bad scripted entry state"),
        }
    }
}

fn base() -> &'static str {
    if cfg!(windows) { r"C:\base" } else { "base" }
}
fn reading() -> FsPath {
    if cfg!(windows) { r"\leaf" } else { "leaf" }.into()
}
fn absolute(name: &str) -> FsPath {
    format!("{}{name}", if cfg!(windows) { r"E:\" } else { "/" }).into()
}
fn filesystem(provider: &str, inputs: Arc<Inputs>) -> LocalFileSystem {
    let mut fs = LocalFileSystem::new(provider);
    fs.native_inputs = inputs;
    fs
}

#[derive(Debug, Default)]
struct SpawnSpy(AtomicUsize);
impl subprocess::NativeSpawner for SpawnSpy {
    fn spawn(&self, _command: &mut tokio::process::Command) -> io::Result<tokio::process::Child> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(io::Error::other("NATIVE SPAWN INTERCEPTED"))
    }
}

fn subprocess(provider: &str, inputs: Arc<Inputs>, spy: Arc<SpawnSpy>) -> LocalSubprocess {
    let mut provider = LocalSubprocess::new(provider);
    provider.native_inputs = inputs;
    provider.native_spawner = spy;
    provider
}

fn frozen() -> Vec<Value> {
    let rows =
        serde_json::from_str::<Value>(include_str!("../../tests/data/node_read_domain.json"))
            .unwrap()["rows"]
            .as_array()
            .unwrap()
            .clone();
    let keys: std::collections::BTreeSet<_> = rows
        .iter()
        .map(|r| {
            (
                r["platform"].as_str().unwrap(),
                r["provider"].as_str().unwrap(),
                r["input"].as_str().unwrap(),
                r["entries"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(keys.len(), 360);
    assert_eq!(rows.len(), 360);
    rows
}
fn platform() -> &'static str {
    if cfg!(windows) { "win32" } else { "posix" }
}
fn expected_public<'a>(row: &'a Value, rows: &'a [Value]) -> &'a Value {
    if row["provider"] == r"\\?\C:\base" {
        rows.iter()
            .find(|other| {
                other["platform"] == row["platform"]
                    && other["provider"] == r"C:\base"
                    && other["input"] == row["input"]
                    && other["entries"] == row["entries"]
            })
            .unwrap()
    } else {
        row
    }
}
fn counts(inputs: &Inputs, expected: &Value) {
    assert_eq!(
        inputs.count(),
        expected["cwdReads"].as_u64().unwrap() as usize,
        "cwd count {expected}"
    );
    let entries: Vec<_> = expected["envReads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_uppercase())
        .collect();
    assert_eq!(*inputs.entries.lock(), entries, "entry sequence {expected}");
}

#[tokio::test]
async fn raw_logical_and_public_read_domains() {
    let rows = frozen();
    let mut visited = 0;
    for row in rows.iter().filter(|r| r["platform"] == platform()) {
        let provider = row["provider"].as_str().unwrap();
        let raw: FsPath = row["input"].as_str().unwrap().into();
        let entry = row["entries"].as_str().unwrap();
        for failure in [
            None,
            Some(2),
            Some(if cfg!(windows) { 5 } else { 13 }),
            Some(if cfg!(windows) { 87 } else { 22 }),
        ] {
            let success = failure.is_none();
            let inputs = Inputs::new(
                entry,
                if success {
                    Read::Value(if cfg!(windows) { r"E:\proc" } else { "/proc" }.into())
                } else {
                    Read::Error(failure.unwrap())
                },
            );
            let logical =
                path::resolve_logical_with_inputs(&provider.into(), &raw, inputs.as_ref());
            counts(&inputs, row);
            if !success && row["cwdReads"] != 0 {
                assert_eq!(logical.unwrap_err().raw_os_error(), failure);
            } else {
                assert_eq!(logical.unwrap(), row["value"].as_str().unwrap(), "{row}");
            }

            let expected = expected_public(row, &rows);
            let inputs = Inputs::new(
                entry,
                if success {
                    Read::Value(if cfg!(windows) { r"E:\proc" } else { "/proc" }.into())
                } else {
                    Read::Error(failure.unwrap())
                },
            );
            let fs = filesystem(provider, inputs.clone());
            let result = FileSystem::absolute_path(&fs, &raw, None).await;
            counts(&inputs, expected);
            if !success && expected["cwdReads"] != 0 {
                let error = result.unwrap_err();
                assert_eq!(
                    error.code,
                    if failure == Some(2) {
                        FsErrorCode::NotFound
                    } else if failure == Some(if cfg!(windows) { 5 } else { 13 }) {
                        FsErrorCode::PermissionDenied
                    } else {
                        FsErrorCode::Invalid
                    }
                );
                assert_eq!(
                    error.message,
                    io::Error::from_raw_os_error(failure.unwrap()).to_string()
                );
                assert_eq!(error.path, Some(raw.clone()));
            } else {
                assert_eq!(
                    result.unwrap(),
                    expected["value"].as_str().unwrap(),
                    "{row}"
                );
            }
        }
        // Fail-if-called readers at every no-read cell, not merely zero counts
        // with a reader that would silently succeed if called incorrectly.
        for public in [false, true] {
            let expected = if public {
                expected_public(row, &rows)
            } else {
                row
            };
            if expected["cwdReads"] == 0 {
                let inputs = Inputs::new(entry, Read::Panic);
                if public {
                    FileSystem::absolute_path(&filesystem(provider, inputs.clone()), &raw, None)
                        .await
                        .unwrap();
                } else {
                    path::resolve_logical_with_inputs(&provider.into(), &raw, inputs.as_ref())
                        .unwrap();
                }
                counts(&inputs, expected);
            }
        }
        visited += 1;
    }
    assert_eq!(visited, if cfg!(windows) { 336 } else { 24 });
}

#[tokio::test]
async fn shell_and_subprocess_read_domains() {
    let rows = frozen();
    for row in rows.iter().filter(|r| r["platform"] == platform()) {
        let expected = expected_public(row, &rows);
        let reads = expected["cwdReads"].as_u64().unwrap();
        let provider = row["provider"].as_str().unwrap();
        let raw = PathBuf::from(row["input"].as_str().unwrap());
        for code in [
            2,
            if cfg!(windows) { 5 } else { 13 },
            if cfg!(windows) { 87 } else { 22 },
        ] {
            for shell in [false, true] {
                let inputs = Inputs::new(
                    row["entries"].as_str().unwrap(),
                    if reads == 0 {
                        Read::Panic
                    } else {
                        Read::Error(code)
                    },
                );
                let spy = Arc::new(SpawnSpy::default());
                let provider = subprocess(provider, inputs.clone(), spy.clone());
                if shell {
                    let mut provider =
                        LocalShell::new(Arc::new(provider)).with_configured_shell(PathBuf::new());
                    provider.native_inputs = inputs.clone();
                    let error = provider
                        .exec(
                            "never executed",
                            ShellExecOptions {
                                cwd: Some(raw.clone()),
                                ..Default::default()
                            },
                        )
                        .await
                        .unwrap_err();
                    assert_eq!(
                        error.code,
                        if reads == 0 {
                            ShellErrorCode::ShellUnavailable
                        } else {
                            ShellErrorCode::SpawnError
                        }
                    );
                    assert_eq!(spy.0.load(Ordering::SeqCst), 0);
                    if reads != 0 {
                        assert_eq!(
                            error.message,
                            io::Error::from_raw_os_error(code).to_string()
                        );
                    }
                } else {
                    let result = provider
                        .spawn(
                            &["never executed".into()],
                            SpawnOptions {
                                cwd: Some(raw.clone()),
                                ..Default::default()
                            },
                        )
                        .await;
                    assert!(result.is_err());
                    assert_eq!(
                        spy.0.load(Ordering::SeqCst),
                        usize::from(reads == 0),
                        "native spawn count"
                    );
                    let Err(error) = result else { unreachable!() };
                    assert_eq!(error.code, SubprocessErrorCode::SpawnError);
                    if reads != 0 {
                        assert_eq!(
                            error.message,
                            io::Error::from_raw_os_error(code).to_string()
                        );
                    }
                }
                counts(&inputs, expected);
            }
        }
    }
}

#[cfg(windows)]
#[tokio::test]
async fn extended_provider_48_observations_not_certification() {
    let rows = frozen();
    let mut observations = Vec::new();
    let mut differences = Vec::new();
    for row in rows.iter().filter(|r| r["provider"] == r"\\?\C:\base") {
        let inputs = Inputs::new(
            row["entries"].as_str().unwrap(),
            Read::Value(r"E:\proc".into()),
        );
        let fs = filesystem(r"\\?\C:\base", inputs.clone());
        let value = FileSystem::absolute_path(&fs, &row["input"].as_str().unwrap().into(), None)
            .await
            .unwrap();
        let expected = expected_public(row, &rows);
        assert_eq!(value, expected["value"].as_str().unwrap());
        counts(&inputs, expected);
        if value != row["value"].as_str().unwrap()
            || inputs.count() as u64 != row["cwdReads"]
            || json!(*inputs.entries.lock())
                != json!(
                    row["envReads"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_uppercase())
                        .collect::<Vec<_>>()
                )
        {
            differences.push((
                row["input"].as_str().unwrap().to_owned(),
                row["entries"].as_str().unwrap().to_owned(),
            ));
        }
        observations.push(json!({"input":row["input"], "entry":row["entries"], "value":value.code_units(), "cwdReads":inputs.count(), "envReads":*inputs.entries.lock()}));
    }
    assert_eq!(observations.len(), 48);
    assert_eq!(differences.len(), 12);
    assert!(
        differences
            .iter()
            .all(|(input, _)| ["c:leaf", "leaf", ""].contains(&input.as_str()))
    );
    println!("EXTENDED_PROVIDER_OBSERVATIONS {}", json!(observations));
}

async fn call(
    fs: &LocalFileSystem,
    op: &str,
    raw: &FsPath,
    signal: Option<&dyn AbortSignal>,
) -> Result<(), FsError> {
    match op {
        "absolute" => FileSystem::absolute_path(fs, raw, signal).await.map(|_| ()),
        "canonical" => FileSystem::canonical_path(fs, raw, signal)
            .await
            .map(|_| ()),
        "info" => FileSystem::file_info(fs, raw, signal).await.map(|_| ()),
        "text" => FileSystem::read_text_file(fs, raw, signal)
            .await
            .map(|_| ()),
        "lines" => FileSystem::read_text_lines(fs, raw, Some(0), signal)
            .await
            .map(|_| ()),
        "binary" => FileSystem::read_binary_file(fs, raw, signal)
            .await
            .map(|_| ()),
        "write" => FileSystem::write_file(fs, raw, b"x", signal).await,
        "append" => FileSystem::append_file(fs, raw, b"x", signal).await,
        "list" => FileSystem::list_dir(fs, raw, signal).await.map(|_| ()),
        "raw_list" => FileSystem::list_dir_raw(fs, raw, signal).await.map(|_| ()),
        "probe" => FileSystem::probe_dir_entry(fs, raw, signal)
            .await
            .map(|_| ()),
        "readable" => FileSystem::check_readable(fs, raw, signal).await,
        "read_write" => FileSystem::check_read_write(fs, raw, signal).await,
        "mkdir" => FileSystem::create_dir(fs, raw, true, signal).await,
        "remove" => FileSystem::remove(fs, raw, true, true, signal).await,
        "exists" => FileSystem::exists(fs, raw, signal).await.map(|_| ()),
        "resolve" => FileSystem::resolve(fs, raw, signal).await.map(|_| ()),
        _ => panic!("bad operation"),
    }
}

#[tokio::test]
async fn every_fs_consumer_preserves_original_error_and_raw_path() {
    let controller = CancellationController::default();
    controller.abort();
    let signal = controller.signal();
    for (code, expected) in [
        (2, FsErrorCode::NotFound),
        (
            if cfg!(windows) { 5 } else { 13 },
            FsErrorCode::PermissionDenied,
        ),
        (if cfg!(windows) { 87 } else { 22 }, FsErrorCode::Invalid),
    ] {
        for op in [
            "absolute",
            "canonical",
            "info",
            "text",
            "lines",
            "binary",
            "write",
            "append",
            "list",
            "raw_list",
            "probe",
            "readable",
            "read_write",
            "mkdir",
            "remove",
            "exists",
            "resolve",
        ] {
            for aborted in [false, true] {
                let inputs = Inputs::new("absent", Read::Error(code));
                let fs = filesystem(base(), inputs.clone());
                let mut units = reading().code_units().to_vec();
                units.extend_from_slice(&[0xd800, 92]);
                let raw = FsPath::from_code_units(units);
                let error = call(&fs, op, &raw, if aborted { Some(&signal) } else { None })
                    .await
                    .expect_err(op);
                assert_eq!(error.code, expected, "{op}");
                assert_eq!(
                    error.message,
                    io::Error::from_raw_os_error(code).to_string(),
                    "{op}"
                );
                assert_eq!(error.path, Some(raw.clone()), "{op}");
                assert_eq!(inputs.count(), 1, "{op}");
            }
        }
    }
}

#[tokio::test]
async fn failure_is_a_result_not_a_panic_and_unrelated_panics_stay_panics() {
    let fs = filesystem(base(), Inputs::fail());
    let result =
        tokio::spawn(async move { FileSystem::absolute_path(&fs, &reading(), None).await }).await;
    assert!(
        result.is_ok(),
        "DIV009 must return a typed result, not leak/panic"
    );
    assert!(
        result.unwrap().is_err(),
        "native read failure may not fabricate a cwd"
    );
    let fs = filesystem(base(), Inputs::new("absent", Read::Panic));
    let result =
        tokio::spawn(async move { FileSystem::absolute_path(&fs, &reading(), None).await }).await;
    assert!(
        result.is_err_and(|e| e.is_panic()),
        "unrelated reader panic must propagate"
    );
}

#[tokio::test]
async fn resolve_second_read_failure_never_uses_the_saved_absolute_fallback() {
    let inputs = Inputs::script(vec![Read::Value(absolute("proc")), Read::Error(2)]);
    let fs = filesystem(base(), inputs.clone());
    let error = FileSystem::resolve(&fs, &reading(), None)
        .await
        .expect_err("resolve must not fall back after cwd read failure");
    assert_eq!(error.path, Some(reading()));
    assert_eq!(inputs.count(), 2);
}

#[tokio::test]
async fn raw_file_url_and_spawn_precedence() {
    let inputs = Inputs::fail();
    let fs = filesystem(base(), inputs.clone());
    let raw: FsPath = "file://srv/".into();
    let error = FileSystem::absolute_path(&fs, &raw, None)
        .await
        .unwrap_err();
    assert_eq!(error.path, Some(raw));
    assert_eq!(inputs.count(), 1);
    let controller = CancellationController::default();
    controller.abort();
    for aborted in [false, true] {
        let inputs = Inputs::fail();
        let spy = Arc::new(SpawnSpy::default());
        let provider = subprocess(base(), inputs.clone(), spy.clone());
        let options = SpawnOptions {
            cwd: Some(path::native(&reading())),
            signal: if aborted {
                Some(Arc::new(controller.signal()))
            } else {
                None
            },
            ..Default::default()
        };
        let result = provider.spawn(&[], options).await;
        let Err(error) = result else {
            panic!("empty argv must fail")
        };
        assert_eq!(
            error.code,
            if aborted {
                SubprocessErrorCode::Aborted
            } else {
                SubprocessErrorCode::SpawnError
            }
        );
        assert_eq!(inputs.count(), 0);
        assert_eq!(spy.0.load(Ordering::SeqCst), 0);
    }
    for (abort, timeout, expected) in [
        (true, None, ShellErrorCode::Aborted),
        (false, Some(0.0), ShellErrorCode::Timeout),
        (false, None, ShellErrorCode::SpawnError),
    ] {
        let inputs = Inputs::fail();
        let spy = Arc::new(SpawnSpy::default());
        let mut shell = LocalShell::new(Arc::new(subprocess(base(), inputs.clone(), spy.clone())))
            .with_configured_shell(PathBuf::new());
        shell.native_inputs = inputs.clone();
        let error = shell
            .exec(
                "never executed",
                ShellExecOptions {
                    cwd: Some(path::native(&reading())),
                    signal: if abort {
                        Some(Arc::new(controller.signal()))
                    } else {
                        None
                    },
                    timeout_seconds: timeout,
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, expected);
        assert_eq!(inputs.count(), usize::from(!abort && timeout.is_none()));
        assert_eq!(spy.0.load(Ordering::SeqCst), 0);
    }
}

#[derive(Debug, Default)]
struct RenameSpy(Mutex<Vec<(PathBuf, PathBuf)>>);
#[async_trait::async_trait]
impl filesystem::HandleOperations for RenameSpy {
    async fn rename(&self, source: &Path, destination: &Path) -> io::Result<()> {
        self.0.lock().push((source.into(), destination.into()));
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "embedded NUL byte",
        ))
    }
}

#[tokio::test]
async fn rename_precedence_and_nul_slots() {
    let controller = CancellationController::default();
    controller.abort();
    let signal = controller.signal();
    for (source_reads, dest_reads, aborted, steps, want_reads, want_path) in [
        (true, false, true, vec![Read::Error(2)], 1, true),
        (true, false, false, vec![Read::Error(2)], 1, true),
        (false, true, true, vec![Read::Error(2)], 1, false),
        (false, true, false, vec![Read::Error(2)], 1, false),
        (true, true, false, vec![Read::Error(2)], 1, true),
        (
            true,
            true,
            false,
            vec![Read::Value(absolute("proc")), Read::Error(2)],
            2,
            false,
        ),
    ] {
        let source = if source_reads {
            reading()
        } else {
            absolute("src")
        };
        let dest = if dest_reads {
            reading()
        } else {
            absolute("dst")
        };
        let inputs = Inputs::script(steps);
        let mut fs = filesystem(base(), inputs.clone());
        let spy = Arc::new(RenameSpy::default());
        fs.handle_operations = spy.clone();
        let error = FileSystem::rename_file(
            &fs,
            &source,
            &dest,
            if aborted { Some(&signal) } else { None },
        )
        .await
        .unwrap_err();
        assert_eq!(error.path, Some(if want_path { source } else { dest }));
        assert_eq!(error.code, FsErrorCode::NotFound);
        assert_eq!(inputs.count(), want_reads);
        assert!(spy.0.lock().is_empty());
    }
    for reads in [false, true] {
        let source = if reads { reading() } else { absolute("src") };
        let dest = if reads { reading() } else { absolute("dst") };
        let inputs = if reads {
            Inputs::script(vec![
                Read::Value(absolute("proc")),
                Read::Value(absolute("next")),
            ])
        } else {
            Inputs::new("absent", Read::Panic)
        };
        let mut fs = filesystem(base(), inputs.clone());
        let spy = Arc::new(RenameSpy::default());
        fs.handle_operations = spy.clone();
        let error = FileSystem::rename_file(&fs, &source, &dest, Some(&signal))
            .await
            .unwrap_err();
        let expected = if reads {
            path::resolve_logical_with_inputs(
                &base().into(),
                &dest,
                Inputs::new("absent", Read::Value(absolute("next"))).as_ref(),
            )
            .unwrap()
        } else {
            dest
        };
        assert_eq!(error.code, FsErrorCode::Aborted);
        assert_eq!(error.path, Some(expected));
        assert_eq!(inputs.count(), if reads { 2 } else { 0 });
        assert!(spy.0.lock().is_empty());
    }
    let source = absolute("src\0x");
    let dest = reading();
    let inputs = Inputs::script(vec![Read::Value(absolute("proc"))]);
    let mut fs = filesystem(base(), inputs.clone());
    let spy = Arc::new(RenameSpy::default());
    fs.handle_operations = spy.clone();
    let error = FileSystem::rename_file(&fs, &source, &dest, None)
        .await
        .unwrap_err();
    assert_eq!(error.code, FsErrorCode::Unknown);
    assert_eq!(error.path, Some(source));
    assert_eq!(inputs.count(), 1);
}

async fn nul_slots(raw: bool, equal: bool, varying: bool) {
    let mut input = reading().code_units().to_vec();
    input.extend_from_slice(&[0, 120]);
    if raw {
        input.push(0xd800)
    }
    let input = FsPath::from_code_units(input);
    let destination = if equal {
        input.clone()
    } else {
        let mut units = input.code_units().to_vec();
        units.push(121);
        FsPath::from_code_units(units)
    };
    let first = absolute("proc");
    let second: FsPath = if !varying {
        first.clone()
    } else if cfg!(windows) {
        r"F:\proc".into()
    } else {
        "/other".into()
    };
    let expected = path::resolve_logical_with_inputs(
        &base().into(),
        &input,
        Inputs::new("absent", Read::Value(first.clone())).as_ref(),
    )
    .unwrap();
    let expected_dest = path::resolve_logical_with_inputs(
        &base().into(),
        &destination,
        Inputs::new("absent", Read::Value(second.clone())).as_ref(),
    )
    .unwrap();
    let inputs = Inputs::script(vec![Read::Value(first), Read::Value(second)]);
    let mut fs = filesystem(base(), inputs.clone());
    let spy = Arc::new(RenameSpy::default());
    fs.handle_operations = spy.clone();
    let error = FileSystem::rename_file(&fs, &input, &destination, None)
        .await
        .unwrap_err();
    assert_eq!(error.code, FsErrorCode::Unknown);
    assert_eq!(
        error.path,
        Some(expected.clone()),
        "source slot owns fallback"
    );
    assert_eq!(
        *spy.0.lock(),
        vec![(path::native(&expected), path::native(&expected_dest))]
    );
    assert_eq!(inputs.count(), 2, "NUL handler must not read again");
}

#[tokio::test]
async fn nul_slots_keep_first_resolution() {
    nul_slots(false, true, true).await;
}

#[tokio::test]
async fn nul_slots_keep_raw_first_resolution() {
    nul_slots(true, true, true).await;
}

#[tokio::test]
async fn nul_slots_paired_stable_and_distinct_positives() {
    for (equal, varying) in [(false, false), (false, true), (true, false)] {
        nul_slots(false, equal, varying).await;
    }
}

#[cfg(windows)]
#[tokio::test]
async fn frozen_rooted_native_operands_through_real_fs() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../tests/data/node_rooted_cwd.json")).unwrap();
    let mut observations = 0;
    for form in fixture["cwd_forms"].as_array().unwrap() {
        let cwd = FsPath::from_code_units(serde_json::from_value(form["cwd"].clone()).unwrap());
        for (input, expected) in fixture["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .zip(form["node"].as_array().unwrap())
        {
            let input = FsPath::from_code_units(serde_json::from_value(input.clone()).unwrap());
            let expected =
                FsPath::from_code_units(serde_json::from_value(expected.clone()).unwrap());
            for provider in [r"C:\base", r"D:\other", r"\\srv\share\base", r"\\?\C:\base"] {
                let inputs = Inputs::new("absent", Read::Value(cwd.clone()));
                let fs = filesystem(provider, inputs.clone());
                assert_eq!(
                    FileSystem::absolute_path(&fs, &input, None).await.unwrap(),
                    expected,
                    "{}",
                    form["label"]
                );
                assert!(
                    inputs.entries.lock().is_empty(),
                    "rooted branch must not read inherited entries"
                );
            }
            observations += 1;
        }
    }
    assert_eq!(observations, 100);
}
