//! Binding witnesses use controlled providers; all behaviour under test remains in the tool.
use super::*;
use crate::{
    execution::*,
    javascript::JsString,
    tools::{PreparedString, ToolExecutionContext},
};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::{
    collections::VecDeque,
    path::Path,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::{Notify, watch};

#[derive(Default)]
struct Gate {
    open: AtomicBool,
    notify: Notify,
}
impl Gate {
    fn release(&self) {
        self.open.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            if self.open.load(Ordering::SeqCst) {
                return;
            }
            notified.await;
        }
    }
}
enum Chunk {
    Bytes(Vec<u8>),
    Delay(Duration, Vec<u8>),
    Hold(Arc<Gate>),
}
struct Stream {
    chunks: tokio::sync::Mutex<VecDeque<Chunk>>,
    closed: AtomicBool,
}
impl Stream {
    fn new(chunks: Vec<Chunk>) -> Arc<Self> {
        Arc::new(Self {
            chunks: tokio::sync::Mutex::new(chunks.into()),
            closed: AtomicBool::new(false),
        })
    }
}
#[async_trait]
impl ReadableStream for Stream {
    async fn read_chunk(&self) -> Result<Option<Vec<u8>>, SubprocessError> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let mut chunks = self.chunks.lock().await;
        match chunks.front() {
            Some(Chunk::Delay(delay, _)) => tokio::time::sleep(*delay).await,
            Some(Chunk::Hold(gate)) => gate.wait().await,
            _ => {}
        }
        Ok(match chunks.pop_front() {
            Some(Chunk::Bytes(bytes) | Chunk::Delay(_, bytes)) => Some(bytes),
            _ => None,
        })
    }
    async fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}
struct Input {
    bytes: Mutex<Vec<u8>>,
    gate: Arc<Gate>,
    started: Arc<Gate>,
    closed: AtomicBool,
}
#[async_trait]
impl WritableStream for Input {
    async fn write(&self, bytes: &[u8]) -> Result<(), SubprocessError> {
        self.started.release();
        self.gate.wait().await;
        self.bytes.lock().extend_from_slice(bytes);
        Ok(())
    }
    async fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}
struct Proc {
    stdout: Arc<Stream>,
    stderr: Arc<Stream>,
    stdin: Option<Arc<Input>>,
    status: watch::Sender<Option<ExitStatus>>,
    kills: AtomicUsize,
}
impl Proc {
    fn new(stdout: Vec<Chunk>, stderr: Vec<Chunk>, code: Option<Option<i32>>) -> Arc<Self> {
        Arc::new(Self {
            stdout: Stream::new(stdout),
            stderr: Stream::new(stderr),
            stdin: None,
            status: watch::channel(code.map(|exit_code| ExitStatus { exit_code })).0,
            kills: AtomicUsize::new(0),
        })
    }
}
#[async_trait]
impl Process for Proc {
    fn pid(&self) -> u32 {
        123
    }
    fn stdin(&self) -> Option<Arc<dyn WritableStream>> {
        self.stdin.clone().map(|s| s as Arc<dyn WritableStream>)
    }
    fn stdout(&self) -> Option<Arc<dyn ReadableStream>> {
        Some(self.stdout.clone())
    }
    fn stderr(&self) -> Option<Arc<dyn ReadableStream>> {
        Some(self.stderr.clone())
    }
    async fn wait(&self) -> Result<ExitStatus, SubprocessError> {
        let mut rx = self.status.subscribe();
        loop {
            if let Some(status) = *rx.borrow_and_update() {
                return Ok(status);
            }
            rx.changed().await.unwrap();
        }
    }
    async fn terminate(&self) {
        self.kills.fetch_add(1, Ordering::SeqCst);
        self.status.send_if_modified(|status| {
            if status.is_none() {
                *status = Some(ExitStatus { exit_code: None });
                true
            } else {
                false
            }
        });
    }
}
struct Spawns {
    local: LocalSubprocess,
    platform: Platform,
    env: EnvSnapshot,
    queue: Mutex<VecDeque<Arc<Proc>>>,
    calls: Mutex<Vec<(Vec<String>, SpawnOptions)>>,
    captures: AtomicUsize,
}
impl Spawns {
    fn new(root: &Path, platform: Platform, queue: Vec<Arc<Proc>>) -> Arc<Self> {
        Arc::new(Self {
            local: LocalSubprocess::new(root),
            platform,
            env: EnvSnapshot::new(match platform {
                Platform::Posix => EnvEntries::Posix(vec![]),
                Platform::Windows => EnvEntries::Windows(vec![]),
            }),
            queue: Mutex::new(queue.into()),
            calls: Mutex::new(vec![]),
            captures: AtomicUsize::new(0),
        })
    }
}
#[async_trait]
impl Subprocess for Spawns {
    fn cwd(&self) -> &Path {
        self.local.cwd()
    }
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        self.local.execution_world()
    }
    fn platform(&self) -> Platform {
        self.platform
    }
    fn base_env(&self) -> EnvSnapshot {
        self.captures.fetch_add(1, Ordering::SeqCst);
        self.env.clone()
    }
    async fn spawn(
        &self,
        argv: &[String],
        options: SpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError> {
        self.calls.lock().push((argv.into(), options));
        self.queue
            .lock()
            .pop_front()
            .map(|p| p as Arc<dyn Process>)
            .ok_or_else(|| {
                SubprocessError::new(SubprocessErrorCode::SpawnError, "no scripted process")
            })
    }
}
struct Fs {
    local: LocalFileSystem,
    calls: Mutex<Vec<String>>,
    missing: Vec<String>,
    append_gate: Arc<Gate>,
    append_started: Arc<Gate>,
    data: Mutex<Vec<u8>>,
    file_error: Option<FsErrorCode>,
    append_error: Option<FsErrorCode>,
    unsupported: bool,
    live: AtomicUsize,
}
impl Fs {
    fn new(root: &Path) -> Self {
        let gate = Arc::new(Gate::default());
        gate.release();
        Self {
            local: LocalFileSystem::new(root),
            calls: Mutex::new(vec![]),
            missing: vec![],
            append_gate: gate,
            append_started: Arc::default(),
            data: Mutex::new(vec![]),
            file_error: None,
            append_error: None,
            unsupported: false,
            live: AtomicUsize::new(0),
        }
    }
}
struct Live<'a>(&'a AtomicUsize);
impl Drop for Live<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
#[async_trait]
impl FileSystem for Fs {
    fn cwd(&self) -> &Path {
        self.local.cwd()
    }
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        self.local.execution_world()
    }
    async fn absolute_path(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        self.local.absolute_path(path, signal).await
    }
    async fn join_path(
        &self,
        parts: &[&FsPath],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        self.local.join_path(parts, signal).await
    }
    async fn read_text_file(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        self.local.read_text_file(path, signal).await
    }
    async fn read_text_lines(
        &self,
        path: &FsPath,
        max_lines: Option<isize>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        self.local.read_text_lines(path, max_lines, signal).await
    }
    async fn read_binary_file(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError> {
        self.local.read_binary_file(path, signal).await
    }
    async fn write_file(
        &self,
        path: &FsPath,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        self.local.write_file(path, content, signal).await
    }
    async fn rename_file(
        &self,
        source: &FsPath,
        destination: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        self.local.rename_file(source, destination, signal).await
    }
    async fn list_dir(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError> {
        self.local.list_dir(path, signal).await
    }
    async fn canonical_path(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        self.local.canonical_path(path, signal).await
    }
    async fn exists(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<bool, FsError> {
        self.local.exists(path, signal).await
    }
    async fn create_dir(
        &self,
        path: &FsPath,
        recursive: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        self.local.create_dir(path, recursive, signal).await
    }
    async fn remove(
        &self,
        path: &FsPath,
        recursive: bool,
        force: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        self.local.remove(path, recursive, force, signal).await
    }
    async fn create_temp_dir(
        &self,
        prefix: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        self.local.create_temp_dir(prefix, signal).await
    }
    async fn resolve(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsTarget, FsError> {
        self.local.resolve(path, signal).await
    }
    async fn process_path(&self, target: &FsTarget) -> Result<FsPath, FsError> {
        self.local.process_path(target).await
    }
    async fn cleanup(&self) {
        self.local.cleanup().await;
    }
    async fn probe_dir_entry(
        &self,
        path: &FsPath,
        _: Option<&dyn AbortSignal>,
    ) -> Result<DirEntryProbe, FsError> {
        let name = String::from_utf16_lossy(path.code_units());
        if self.unsupported {
            return Err(FsError::new(FsErrorCode::NotSupported, "unsupported"));
        }
        self.calls.lock().push(format!("probe {name}"));
        if self.missing.contains(&name) {
            return Err(FsError::new(FsErrorCode::NotFound, "missing"));
        }
        Ok(DirEntryProbe {
            name: path.clone(),
            path: path.clone(),
            kind: DirEntryProbeKind::Directory,
        })
    }
    async fn file_info(
        &self,
        path: &FsPath,
        _: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError> {
        if self.unsupported {
            return Err(FsError::new(FsErrorCode::NotSupported, "unsupported"));
        }
        self.calls.lock().push(format!(
            "info {}",
            String::from_utf16_lossy(path.code_units())
        ));
        if self
            .missing
            .contains(&String::from_utf16_lossy(path.code_units()))
        {
            return Err(FsError::new(FsErrorCode::NotFound, "missing"));
        }
        Ok(FileInfo {
            name: path.clone(),
            path: path.clone(),
            kind: FileKind::Directory,
            size: 0,
            mtime_ms: 0,
        })
    }
    async fn create_temp_file(
        &self,
        prefix: &str,
        suffix: &str,
        _: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        assert_eq!((prefix, suffix), ("minion-bash-", ".log"));
        if let Some(code) = self.file_error {
            return Err(FsError::new(code, "native text never projected"));
        }
        Ok("full.log".into())
    }
    async fn append_file(
        &self,
        _: &FsPath,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        if let Some(code) = self.append_error {
            return Err(FsError::new(code, "native wording never exposed"));
        }
        assert!(signal.is_none());
        self.live.fetch_add(1, Ordering::SeqCst);
        let _live = Live(&self.live);
        self.append_started.release();
        self.append_gate.wait().await;
        self.data.lock().extend_from_slice(content);
        Ok(())
    }
}
fn request(seconds: Option<f64>) -> ToolExecutionRequest {
    let mut params: PreparedValue = serde_json::json!({"command":"true"}).into();
    if let Some(seconds) = seconds {
        params
            .as_object_mut()
            .unwrap()
            .insert("timeout".into(), PreparedValue::number(seconds));
    }
    ToolExecutionRequest {
        tool_call_id: "call".into(),
        params,
        signal: None,
        on_update: None,
        context: None,
    }
}
fn text(result: AgentToolResult) -> String {
    match &result.content[0] {
        crate::llm::ToolResultContentBlock::Text(block) => {
            String::from_utf16_lossy(block.text.code_units())
        }
        _ => panic!("text"),
    }
}

#[tokio::test(start_paused = true)]
async fn bash_settlement_waits_for_output_resets_grace_and_releases_both_pipes() {
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let process = Proc::new(
        vec![
            Chunk::Delay(Duration::from_millis(60), b"a".to_vec()),
            Chunk::Delay(Duration::from_millis(60), b"b".to_vec()),
            Chunk::Hold(Arc::default()),
        ],
        vec![Chunk::Hold(Arc::default())],
        Some(Some(0)),
    );
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process.clone()]);
    let began = Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        execute(fs, subprocess, BashToolOptions::default(), request(None)),
    )
    .await;
    assert!(
        result.is_ok(),
        "settlement must not wait indefinitely for inherited pipes"
    );
    let result = result.unwrap().unwrap();
    assert_eq!(text(result), "ab");
    assert_eq!(Instant::now() - began, Duration::from_millis(220));
    assert!(process.stdout.closed.load(Ordering::SeqCst));
    assert!(process.stderr.closed.load(Ordering::SeqCst));
    assert_eq!(process.kills.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn bash_persistence_join_freezes_timeout_and_abort_and_preserves_raw_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let mut fs = Fs::new(dir.path());
    fs.append_gate = Arc::default();
    let fs = Arc::new(fs);
    let raw = [vec![0xef, 0xbb, 0xbf], vec![b'x'; 51201], vec![0xff]].concat();
    let process = Proc::new(vec![Chunk::Bytes(raw.clone())], vec![], Some(Some(0)));
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process]);
    let mut req = request(Some(0.01));
    let signal = Arc::new(TestSignal(AtomicBool::new(false)));
    req.signal = Some(signal.clone());
    let work = execute(fs.clone(), subprocess, BashToolOptions::default(), req);
    tokio::pin!(work);
    tokio::select! {biased; result=&mut work=>panic!("not joined: {result:?}"),()=fs.append_started.wait()=>{}}
    // Allow execute to reach its persistence join before changing the signal.
    assert!(futures::poll!(&mut work).is_pending());
    tokio::time::advance(Duration::from_secs(1)).await;
    signal.0.store(true, Ordering::SeqCst);
    fs.append_gate.release();
    let result = tokio::time::timeout(Duration::from_secs(2), work).await;
    assert!(
        result.is_ok(),
        "released persistence must finish without an intake deadlock"
    );
    let result = result.unwrap();
    assert!(
        result.is_ok(),
        "settled success must not be reclassified by output finalization"
    );
    let result = result.unwrap();
    assert!(!text(result).contains("timed out"));
    assert_eq!(*fs.data.lock(), raw);
    assert_eq!(fs.live.load(Ordering::SeqCst), 0);
}
struct TestSignal(AtomicBool);
impl ToolExecutionSignal for TestSignal {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[tokio::test(start_paused = true)]
async fn bash_cancel_drops_the_writer_future_and_no_write_outlives_the_call() {
    let dir = tempfile::tempdir().unwrap();
    let mut fs = Fs::new(dir.path());
    fs.append_gate = Arc::default();
    let fs = Arc::new(fs);
    let process = Proc::new(vec![Chunk::Bytes(vec![b'x'; 51201])], vec![], Some(Some(0)));
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process]);
    let mut work = Box::pin(execute(
        fs.clone(),
        subprocess,
        BashToolOptions::default(),
        request(None),
    ));
    tokio::select! {biased;result=&mut work=>panic!("not joined: {result:?}"),()=fs.append_started.wait()=>{}}
    assert_eq!(fs.live.load(Ordering::SeqCst), 1);
    drop(work);
    assert_eq!(fs.live.load(Ordering::SeqCst), 0);
    fs.append_gate.release();
    assert!(fs.data.lock().is_empty());
}

#[tokio::test(start_paused = true)]
async fn bash_file_failure_kills_before_own_error_and_discards_output() {
    let dir = tempfile::tempdir().unwrap();
    let mut fs = Fs::new(dir.path());
    fs.file_error = Some(FsErrorCode::PermissionDenied);
    let process = Proc::new(
        vec![Chunk::Bytes(vec![b'x'; 51201]), Chunk::Hold(Arc::default())],
        vec![],
        None,
    );
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process.clone()]);
    let error = execute(
        Arc::new(fs),
        subprocess,
        BashToolOptions::default(),
        request(None),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.message().as_str(),
        Some("Cannot write the full-output file: permission denied")
    );
    assert_eq!(process.kills.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn bash_precheck_order_timeout_before_abort_before_shell_and_cwd() {
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![]);
    for seconds in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        -1.0,
        2147483.648,
    ] {
        let mut req = request(Some(seconds));
        req.signal = Some(Arc::new(TestSignal(AtomicBool::new(true))));
        let error = execute(
            fs.clone(),
            subprocess.clone(),
            BashToolOptions::default(),
            req,
        )
        .await
        .unwrap_err();
        assert!(
            error
                .message()
                .as_str()
                .unwrap()
                .starts_with("Invalid timeout:")
        );
    }
    let mut req = request(None);
    req.signal = Some(Arc::new(TestSignal(AtomicBool::new(true))));
    assert_eq!(
        execute(
            fs.clone(),
            subprocess.clone(),
            BashToolOptions::default(),
            req
        )
        .await
        .unwrap_err()
        .message()
        .as_str(),
        Some("Command aborted")
    );
    assert!(fs.calls.lock().is_empty());
    assert!(subprocess.calls.lock().is_empty());
    assert_eq!(subprocess.captures.load(Ordering::SeqCst), 8);
}

#[tokio::test(start_paused = true)]
async fn bash_legacy_stdin_does_not_block_timeout_and_command_uses_usv_projection() {
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let mut process = Proc::new(vec![Chunk::Hold(Arc::default())], vec![], None);
    let input = Arc::new(Input {
        bytes: Mutex::default(),
        gate: Arc::default(),
        started: Arc::default(),
        closed: AtomicBool::new(false),
    });
    Arc::get_mut(&mut process).unwrap().stdin = Some(input.clone());
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process.clone()]);
    let began = Instant::now();
    let mut req = request(Some(0.0019));
    req.params.as_object_mut().unwrap().insert(
        "command".into(),
        PreparedValue::String(PreparedString::from_code_units(vec![0xd800])),
    );
    let error = execute(
        fs,
        subprocess.clone(),
        BashToolOptions {
            shell_path: Some("C:/Windows/System32/bash.exe".into()),
            ..BashToolOptions::default()
        },
        req,
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.message().as_str(),
        Some("Command timed out after 0.0019 seconds")
    );
    assert_eq!(Instant::now() - began, Duration::from_millis(101));
    assert_eq!(process.kills.load(Ordering::SeqCst), 1);
    let calls = subprocess.calls.lock();
    assert_eq!(calls[0].0, vec!["C:/Windows/System32/bash.exe", "-s"]);
    assert_eq!(calls[0].1.stdin, StdioMode::Piped);
    assert!(input.started.open.load(Ordering::SeqCst));
    assert!(input.bytes.lock().is_empty());
}

#[tokio::test]
async fn bash_context_env_is_explicit_clean_and_provider_snapshot_is_per_call() {
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let mut subprocess = Spawns::new(
        dir.path(),
        Platform::Posix,
        vec![
            Proc::new(vec![], vec![], Some(Some(0))),
            Proc::new(vec![], vec![], Some(Some(0))),
        ],
    );
    Arc::get_mut(&mut subprocess).unwrap().env = EnvSnapshot::new(EnvEntries::Posix(vec![
        (b"MINION_MODEL".to_vec(), b"stale".to_vec()),
        (b"minion_model".to_vec(), b"keep".to_vec()),
        (b"UNRELATED".to_vec(), b"value".to_vec()),
    ]));
    let mut req = request(None);
    req.context = Some(ToolExecutionContext::new(
        "id".into(),
        None,
        Some("p".into()),
        Some("m".into()),
        Some("off".into()),
    ));
    execute(
        fs.clone(),
        subprocess.clone(),
        BashToolOptions::default(),
        req,
    )
    .await
    .unwrap();
    execute(
        fs,
        subprocess.clone(),
        BashToolOptions::default(),
        request(None),
    )
    .await
    .unwrap();
    let calls = subprocess.calls.lock();
    let env = &calls[0].1.env;
    assert!(!calls[0].1.inherit_env);
    assert_eq!(env["MINION_SESSION_ID"], "id");
    assert_eq!(env["MINION_PROVIDER"], "p");
    assert_eq!(env["MINION_MODEL"], "m");
    assert_eq!(env["MINION_REASONING_LEVEL"], "off");
    assert!(!env.contains_key("MINION_SESSION_FILE"));
    assert_eq!(env["minion_model"], "keep");
    assert_eq!(env["UNRELATED"], "value");
    assert!(!calls[1].1.env.contains_key("MINION_MODEL"));
    assert_eq!(subprocess.captures.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn bash_lookup_waits_for_exit_and_both_eof_not_command_grace() {
    let dir = tempfile::tempdir().unwrap();
    let mut fs = Fs::new(dir.path());
    fs.missing.push("/bin/bash".into());
    let fs = Arc::new(fs);
    let lookup = Proc::new(
        vec![Chunk::Delay(
            Duration::from_millis(250),
            b"\xef\xbb\xbf/found\r\nignored".to_vec(),
        )],
        vec![Chunk::Delay(Duration::from_millis(300), vec![b'e'])],
        Some(Some(0)),
    );
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![lookup.clone()]);
    let began = Instant::now();
    let selected = bash_shell::select(
        &(fs as Arc<dyn FileSystem>),
        &(subprocess.clone() as Arc<dyn Subprocess>),
        &subprocess.base_env(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(selected.shell, "/found");
    assert!(Instant::now() - began >= Duration::from_millis(300));
    assert_eq!(lookup.kills.load(Ordering::SeqCst), 0);
    assert!(lookup.stdout.closed.load(Ordering::SeqCst));
    assert!(lookup.stderr.closed.load(Ordering::SeqCst));
    assert!(subprocess.calls.lock()[0].1.inherit_env);
}

#[tokio::test(start_paused = true)]
async fn bash_lookup_combined_budget_keeps_exited_status_and_kills_only_unexited() {
    for exited in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut fs = Fs::new(dir.path());
        fs.missing.push("/bin/bash".into());
        let fs = Arc::new(fs);
        let lookup = Proc::new(
            vec![
                Chunk::Bytes([b"/found\n".to_vec(), vec![b'a'; 600000]].concat()),
                Chunk::Hold(Arc::default()),
            ],
            vec![
                Chunk::Delay(Duration::from_millis(10), vec![b'e'; 600000]),
                Chunk::Hold(Arc::default()),
            ],
            if exited { Some(Some(0)) } else { None },
        );
        let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![lookup.clone()]);
        let began = Instant::now();
        let selected = bash_shell::select(
            &(fs as Arc<dyn FileSystem>),
            &(subprocess.clone() as Arc<dyn Subprocess>),
            &subprocess.base_env(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(selected.shell, if exited { "/found" } else { "sh" });
        assert_eq!(lookup.kills.load(Ordering::SeqCst), usize::from(!exited));
        assert!(Instant::now() - began < Duration::from_secs(1));
    }
}

#[tokio::test]
async fn bash_windows_probe_selection_and_nonfollowing_cwd_are_distinct() {
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let mut subprocess = Spawns::new(
        dir.path(),
        Platform::Windows,
        vec![Proc::new(vec![], vec![], Some(Some(0)))],
    );
    Arc::get_mut(&mut subprocess).unwrap().env = EnvSnapshot::new(EnvEntries::Windows(vec![(
        JsString::from("ProgramFiles"),
        JsString::from("P"),
    )]));
    execute(
        fs.clone(),
        subprocess.clone(),
        BashToolOptions::default(),
        request(None),
    )
    .await
    .unwrap();
    assert_eq!(
        subprocess.calls.lock()[0].0,
        vec!["P\\Git\\bin\\bash.exe", "-c", "true"]
    );
    let calls = fs.calls.lock();
    assert_eq!(calls[0], "probe P\\Git\\bin\\bash.exe");
    assert!(calls[1].starts_with("info "));
}

#[tokio::test(start_paused = true)]
async fn bash_timer_is_whole_ms_minimum_one_and_abort_wins_during_grace() {
    for (seconds, milliseconds) in [(0.0005, 1), (0.001, 1), (0.0019, 1), (0.0025, 2)] {
        let dir = tempfile::tempdir().unwrap();
        let fs = Arc::new(Fs::new(dir.path()));
        let process = Proc::new(vec![Chunk::Hold(Arc::default())], vec![], None);
        let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process.clone()]);
        let started = Instant::now();
        let signal = Arc::new(TestSignal(AtomicBool::new(false)));
        let mut req = request(Some(seconds));
        req.signal = Some(signal.clone());
        let work = execute(fs, subprocess, BashToolOptions::default(), req);
        tokio::pin!(work);
        tokio::select! { biased; result=&mut work=>panic!("premature: {result:?}"), ()=tokio::time::sleep(Duration::from_millis(milliseconds+10))=>{} }
        assert_eq!(process.kills.load(Ordering::SeqCst), 1);
        signal.0.store(true, Ordering::SeqCst);
        assert_eq!(
            work.await.unwrap_err().message().as_str(),
            Some("Command aborted")
        );
        assert_eq!(
            Instant::now() - started,
            Duration::from_millis(milliseconds + 100)
        );
    }
}

#[tokio::test]
async fn bash_command_projection_argv_and_stdin_preserves_pairs_replaces_lone_units() {
    for (units, expected) in [
        (vec![0xd800], "\u{fffd}"),
        (vec![0xdc00], "\u{fffd}"),
        (vec![0xdc00, 0xd800], "\u{fffd}\u{fffd}"),
        (vec![0xd83d, 0xde00], "\u{1f600}"),
    ] {
        for stdin in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let fs = Arc::new(Fs::new(dir.path()));
            let mut process = Proc::new(vec![], vec![], Some(Some(0)));
            let input = Arc::new(Input {
                bytes: Mutex::default(),
                gate: Arc::default(),
                started: Arc::default(),
                closed: AtomicBool::new(false),
            });
            input.gate.release();
            Arc::get_mut(&mut process).unwrap().stdin = Some(input.clone());
            let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process]);
            let mut req = request(None);
            req.params.as_object_mut().unwrap().insert(
                "command".into(),
                PreparedValue::String(PreparedString::from_code_units(units.clone())),
            );
            let shell = if stdin {
                "C:\\Windows\\Sysnative\\bash.exe"
            } else {
                "custom"
            };
            if stdin {
                let proc = subprocess.queue.lock()[0].clone();
                proc.status.send_replace(None);
                let work = execute(
                    fs,
                    subprocess.clone(),
                    BashToolOptions {
                        shell_path: Some(shell.into()),
                        ..BashToolOptions::default()
                    },
                    req,
                );
                tokio::pin!(work);
                tokio::select! { biased; result=&mut work=>panic!("premature: {result:?}"), ()=input.started.wait()=>{} }
                assert_eq!(*input.bytes.lock(), expected.as_bytes());
                assert!(input.closed.load(Ordering::SeqCst));
                proc.status
                    .send_replace(Some(ExitStatus { exit_code: Some(0) }));
                work.await.unwrap();
            } else {
                execute(
                    fs,
                    subprocess.clone(),
                    BashToolOptions {
                        shell_path: Some(shell.into()),
                        ..BashToolOptions::default()
                    },
                    req,
                )
                .await
                .unwrap();
                assert_eq!(subprocess.calls.lock()[0].0, vec![shell, "-c", expected]);
                assert_eq!(subprocess.calls.lock()[0].1.stdin, StdioMode::Null);
            }
        }
    }
}

#[tokio::test(start_paused = true)]
async fn bash_lookup_timeout_after_exit_keeps_zero_and_releases_pipes() {
    let dir = tempfile::tempdir().unwrap();
    let mut fs = Fs::new(dir.path());
    fs.missing.push("/bin/bash".into());
    let lookup = Proc::new(
        vec![
            Chunk::Bytes(b"/found".to_vec()),
            Chunk::Hold(Arc::default()),
        ],
        vec![Chunk::Hold(Arc::default())],
        Some(Some(0)),
    );
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![lookup.clone()]);
    let began = Instant::now();
    let shell = bash_shell::select(
        &(Arc::new(fs) as Arc<dyn FileSystem>),
        &(subprocess.clone() as Arc<dyn Subprocess>),
        &subprocess.base_env(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(shell.shell, "/found");
    assert_eq!(Instant::now() - began, Duration::from_secs(5));
    assert_eq!(lookup.kills.load(Ordering::SeqCst), 0);
    assert!(lookup.stdout.closed.load(Ordering::SeqCst));
    assert!(lookup.stderr.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn bash_empty_override_falls_through_custom_missing_fails_and_windows_lookup_probes() {
    let dir = tempfile::tempdir().unwrap();
    let mut fs = Fs::new(dir.path());
    fs.missing.extend(["missing".into(), "/found".into()]);
    let fs = Arc::new(fs) as Arc<dyn FileSystem>;
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![]);
    let s = subprocess.clone() as Arc<dyn Subprocess>;
    assert_eq!(
        bash_shell::select(&fs, &s, &subprocess.base_env(), Some(""))
            .await
            .unwrap()
            .shell,
        "/bin/bash"
    );
    assert_eq!(
        bash_shell::select(&fs, &s, &subprocess.base_env(), Some("missing"))
            .await
            .unwrap_err()
            .message()
            .as_str(),
        Some("Custom shell path not found: missing")
    );
    let subprocess = Spawns::new(
        dir.path(),
        Platform::Windows,
        vec![Proc::new(
            vec![Chunk::Bytes(b"/found\n".to_vec())],
            vec![],
            Some(Some(0)),
        )],
    );
    let error = bash_shell::select(
        &fs,
        &(subprocess.clone() as Arc<dyn Subprocess>),
        &subprocess.base_env(),
        None,
    )
    .await
    .unwrap_err();
    assert!(
        error
            .message()
            .as_str()
            .unwrap()
            .starts_with("No bash shell found. Options:")
    );
    assert_eq!(subprocess.calls.lock()[0].0, vec!["where", "bash.exe"]);
}

#[tokio::test]
async fn bash_spawn_failure_has_own_text_and_zero_updates() {
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![]);
    assert_eq!(
        execute(
            fs.clone(),
            subprocess,
            BashToolOptions::default(),
            request(None)
        )
        .await
        .unwrap_err()
        .message()
        .as_str(),
        Some("Failed to start the shell /bin/bash")
    );
    let subprocess = Spawns::new(
        dir.path(),
        Platform::Posix,
        vec![Proc::new(
            vec![Chunk::Bytes(b"one".to_vec()), Chunk::Bytes(b"two".to_vec())],
            vec![],
            Some(Some(0)),
        )],
    );
    let updates = Arc::new(AtomicUsize::new(0));
    let counter = updates.clone();
    let mut req = request(None);
    req.on_update = Some(Arc::new(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
    }));
    assert_eq!(
        text(
            execute(fs, subprocess, BashToolOptions::default(), req)
                .await
                .unwrap()
        ),
        "onetwo"
    );
    assert_eq!(updates.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn bash_disabled_context_is_removed_signal_delegated_and_worlds_checked() {
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let mut subprocess = Spawns::new(
        dir.path(),
        Platform::Posix,
        vec![Proc::new(vec![], vec![], Some(Some(0)))],
    );
    Arc::get_mut(&mut subprocess).unwrap().env = EnvSnapshot::new(EnvEntries::Posix(vec![(
        b"MINION_MODEL".to_vec(),
        b"stale".to_vec(),
    )]));
    let mut req = request(None);
    req.context = Some(ToolExecutionContext::new(
        "id".into(),
        Some("file".into()),
        None,
        None,
        None,
    ));
    let signal = Arc::new(TestSignal(AtomicBool::new(false)));
    req.signal = Some(signal.clone());
    execute(
        fs.clone(),
        subprocess.clone(),
        BashToolOptions {
            expose_session_environment: false,
            ..BashToolOptions::default()
        },
        req,
    )
    .await
    .unwrap();
    let calls = subprocess.calls.lock();
    assert!(!calls[0].1.env.keys().any(|k| k.starts_with("MINION_")));
    let delegated = calls[0].1.signal.as_ref().expect("tool signal delegated");
    assert!(!delegated.aborted());
    signal.0.store(true, Ordering::SeqCst);
    assert!(delegated.aborted());
    let incompatible = Arc::new(LocalSubprocess::with_world(
        dir.path(),
        ExecutionWorldIdentity::fresh(),
    ));
    assert!(create_bash_tool(fs, incompatible, BashToolOptions::default()).is_err());
}

#[tokio::test(start_paused = true)]
async fn bash_append_failure_uses_own_error_and_prerequisites_are_not_fabricated() {
    let dir = tempfile::tempdir().unwrap();
    let mut fs = Fs::new(dir.path());
    fs.append_error = Some(FsErrorCode::Unknown);
    let process = Proc::new(
        vec![Chunk::Bytes(vec![b'x'; 51201]), Chunk::Hold(Arc::default())],
        vec![],
        None,
    );
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process.clone()]);
    let error = execute(
        Arc::new(fs),
        subprocess,
        BashToolOptions::default(),
        request(None),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.message().as_str(),
        Some("Cannot write the full-output file: unknown filesystem error")
    );
    assert_eq!(process.kills.load(Ordering::SeqCst), 1);
    for platform in [Platform::Posix, Platform::Windows] {
        let mut fs = Fs::new(dir.path());
        fs.unsupported = true;
        let fs = Arc::new(fs) as Arc<dyn FileSystem>;
        let subprocess = Spawns::new(dir.path(), platform, vec![]) as Arc<dyn Subprocess>;
        let error = bash_shell::check_cwd(&fs, &subprocess).await.unwrap_err();
        assert_eq!(
            error.message().as_str(),
            Some(if platform == Platform::Posix {
                "bash requires a filesystem provider that supports probe_dir_entry"
            } else {
                "bash requires a filesystem provider that supports file_info"
            })
        );
    }
}

#[tokio::test(start_paused = true)]
async fn bash_lookup_retains_the_budget_crossing_stdout_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let mut fs = Fs::new(dir.path());
    fs.missing.push("/bin/bash".into());
    let process = Proc::new(
        vec![Chunk::Bytes(
            [b"/crossing\n".to_vec(), vec![b'x'; 1048576]].concat(),
        )],
        vec![],
        Some(Some(0)),
    );
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process.clone()]);
    let selected = bash_shell::select(
        &(Arc::new(fs) as Arc<dyn FileSystem>),
        &(subprocess.clone() as Arc<dyn Subprocess>),
        &subprocess.base_env(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(selected.shell, "/crossing");
    assert_eq!(process.kills.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn bash_stdout_stderr_share_one_decoder_in_read_completion_order() {
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let process = Proc::new(
        vec![
            Chunk::Bytes(vec![0xe2]),
            Chunk::Delay(Duration::from_millis(20), vec![0xac]),
        ],
        vec![Chunk::Delay(Duration::from_millis(10), vec![0x82])],
        Some(Some(0)),
    );
    let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![process]);
    assert_eq!(
        text(
            execute(fs, subprocess, BashToolOptions::default(), request(None))
                .await
                .unwrap()
        ),
        "\u{20ac}"
    );
}

#[tokio::test]
async fn bash_nonzero_empty_output_and_notice_precede_the_status() {
    for code in [1, 255] {
        let dir = tempfile::tempdir().unwrap();
        let fs = Arc::new(Fs::new(dir.path()));
        let subprocess = Spawns::new(
            dir.path(),
            Platform::Posix,
            vec![Proc::new(vec![], vec![], Some(Some(code)))],
        );
        assert_eq!(
            execute(fs, subprocess, BashToolOptions::default(), request(None))
                .await
                .unwrap_err()
                .message()
                .as_str(),
            Some(format!("(no output)\n\nCommand exited with code {code}").as_str())
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let fs = Arc::new(Fs::new(dir.path()));
    let subprocess = Spawns::new(
        dir.path(),
        Platform::Posix,
        vec![Proc::new(
            vec![Chunk::Bytes(vec![b'a'; 51201])],
            vec![],
            Some(Some(1)),
        )],
    );
    let error = execute(fs, subprocess, BashToolOptions::default(), request(None))
        .await
        .unwrap_err();
    assert_eq!(
        error.message().as_str().unwrap(),
        format!(
            "{}\n\n[Showing last 50.0KB of line 1 (line is 50.0KB). Full output: full.log]\n\nCommand exited with code 1",
            "a".repeat(51200)
        )
    );
}

#[tokio::test(start_paused = true)]
async fn bash_lookup_selection_uses_real_status_truthiness_and_buffer_utf8() {
    for (code, bytes, expected) in [
        (Some(0), b"".to_vec(), "sh"),
        (Some(0), b" \r\n ".to_vec(), "sh"),
        (Some(1), b"/found".to_vec(), "sh"),
        (None, b"/found".to_vec(), "sh"),
        (Some(0), vec![0xff], "\u{fffd}"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut fs = Fs::new(dir.path());
        fs.missing.push("/bin/bash".into());
        let proc = Proc::new(vec![Chunk::Bytes(bytes)], vec![], Some(code));
        let subprocess = Spawns::new(dir.path(), Platform::Posix, vec![proc]);
        let selected = bash_shell::select(
            &(Arc::new(fs) as Arc<dyn FileSystem>),
            &(subprocess.clone() as Arc<dyn Subprocess>),
            &subprocess.base_env(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(selected.shell, expected);
    }
}
