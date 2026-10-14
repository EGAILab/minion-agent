use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    pin::Pin,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use tokio::{
    io::{AsyncRead, AsyncWriteExt, ReadBuf},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command},
    sync::{Mutex, Notify, watch},
};

use super::{
    AbortSignal, EnvEntries, EnvSnapshot, ExecutionWorldIdentity, Platform, SubprocessError,
    SubprocessErrorCode,
};
use crate::llm::ResultString;
#[cfg(test)]
#[path = "spawn_nul_tests.rs"]
mod spawn_nul_tests;

/// Lossless, ordered additive EXEC-013 carrier. Repeated exact overlay names
/// overwrite their value without moving their first insertion position.
#[derive(Clone)]
pub struct LosslessSpawnOptions {
    pub cwd: Option<ResultString>,
    pub env: Vec<(ResultString, ResultString)>,
    pub inherit_env: bool,
    pub stdin: StdioMode,
    pub stdout: StdioMode,
    pub stderr: StdioMode,
    pub signal: Option<Arc<dyn AbortSignal>>,
}

impl Default for LosslessSpawnOptions {
    fn default() -> Self {
        SpawnOptions::default().into()
    }
}

impl From<SpawnOptions> for LosslessSpawnOptions {
    fn from(options: SpawnOptions) -> Self {
        Self {
            cwd: options.cwd.as_deref().map(super::path::from_native),
            env: options
                .env
                .into_iter()
                .map(|(n, v)| (n.into(), v.into()))
                .collect(),
            inherit_env: options.inherit_env,
            stdin: options.stdin,
            stdout: options.stdout,
            stderr: options.stderr,
            signal: options.signal,
        }
    }
}

fn snapshot_entries(snapshot: EnvSnapshot) -> Vec<(ResultString, ResultString)> {
    match snapshot.copy() {
        EnvEntries::Windows(entries) => entries
            .into_iter()
            .map(|(n, v)| (n.into(), v.into()))
            .collect(),
        EnvEntries::Posix(entries) => entries
            .into_iter()
            .filter_map(|(n, v)| {
                String::from_utf8(n)
                    .ok()
                    .map(|n| (n.into(), String::from_utf8_lossy(&v).into_owned().into()))
            })
            .collect(),
    }
}

fn prepare_spawn<S: Subprocess + ?Sized>(
    provider: &S,
    argv: &[ResultString],
    options: &LosslessSpawnOptions,
) -> Result<(ResultString, Vec<(ResultString, ResultString)>), SubprocessError> {
    if options.signal.as_ref().is_some_and(|s| s.aborted()) {
        return Err(SubprocessError::new(
            SubprocessErrorCode::Aborted,
            "operation aborted",
        ));
    }
    if argv.is_empty() {
        return Err(SubprocessError::new(
            SubprocessErrorCode::SpawnError,
            "argv must contain a program",
        ));
    }
    let logical_base = provider.logical_cwd();
    let cwd = match options.cwd.as_ref() {
        Some(raw) => super::path::resolve_logical(&logical_base, raw).map_err(map_spawn_error)?,
        None => logical_base,
    };
    let baseline = if options.inherit_env {
        provider.spawn_environment()
    } else {
        Vec::new()
    };
    let env = super::spawn_nul::effective_environment(
        &baseline,
        &options.env,
        options.inherit_env,
        provider.platform(),
    );
    super::spawn_nul::validate(argv, &cwd, &env)?;
    Ok((cwd, env))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StdioMode {
    Inherit,
    Piped,
    #[default]
    Null,
}

#[derive(Clone)]
pub struct SpawnOptions {
    pub cwd: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    pub inherit_env: bool,
    pub stdin: StdioMode,
    pub stdout: StdioMode,
    pub stderr: StdioMode,
    pub signal: Option<Arc<dyn AbortSignal>>,
}

impl Default for SpawnOptions {
    fn default() -> Self {
        Self {
            cwd: None,
            env: BTreeMap::new(),
            inherit_env: true,
            stdin: StdioMode::Null,
            stdout: StdioMode::Piped,
            stderr: StdioMode::Piped,
            signal: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExitStatus {
    pub exit_code: Option<i32>,
}

#[async_trait]
pub trait WritableStream: Send + Sync {
    async fn write(&self, data: &[u8]) -> Result<(), SubprocessError>;
    async fn close(&self);
}

#[async_trait]
pub trait ReadableStream: Send + Sync {
    async fn read_chunk(&self) -> Result<Option<Vec<u8>>, SubprocessError>;
    /// Release only this local read end, abandoning unread output. Pending and
    /// later reads become EOF; this never signals a process or closes a sibling.
    async fn close(&self);
}

#[async_trait]
pub trait Process: Send + Sync {
    fn pid(&self) -> u32;
    fn stdin(&self) -> Option<Arc<dyn WritableStream>>;
    fn stdout(&self) -> Option<Arc<dyn ReadableStream>>;
    fn stderr(&self) -> Option<Arc<dyn ReadableStream>>;
    async fn wait(&self) -> Result<ExitStatus, SubprocessError>;
    async fn terminate(&self);
}

#[async_trait]
pub trait Subprocess: Send + Sync {
    fn cwd(&self) -> &Path;
    fn execution_world(&self) -> &ExecutionWorldIdentity;
    /// Read-only declaration, constant for this provider's lifetime (EXEC-010).
    fn platform(&self) -> Platform;
    /// Isolated snapshot of this provider's configured inheritance baseline.
    fn base_env(&self) -> EnvSnapshot;
    fn logical_cwd(&self) -> ResultString {
        super::path::from_native(self.cwd())
    }
    /// Logical configured inheritance entries before strings-to-OS projection.
    fn spawn_environment(&self) -> Vec<(ResultString, ResultString)> {
        snapshot_entries(self.base_env())
    }
    /// Providers retaining the legacy seam get validation before its projection.
    async fn spawn_lossless(
        &self,
        argv: &[ResultString],
        options: LosslessSpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError> {
        prepare_spawn(self, argv, &options)?;
        let argv: Vec<_> = argv
            .iter()
            .map(|a| String::from_utf16_lossy(a.code_units()))
            .collect();
        self.spawn(
            &argv,
            SpawnOptions {
                cwd: options.cwd.as_ref().map(super::path::native),
                env: options
                    .env
                    .into_iter()
                    .map(|(n, v)| {
                        (
                            String::from_utf16_lossy(n.code_units()),
                            String::from_utf16_lossy(v.code_units()),
                        )
                    })
                    .collect(),
                inherit_env: options.inherit_env,
                stdin: options.stdin,
                stdout: options.stdout,
                stderr: options.stderr,
                signal: options.signal,
            },
        )
        .await
    }
    async fn spawn(
        &self,
        argv: &[String],
        options: SpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError>;
}

#[derive(Clone, Debug)]
/// The local platform declaration cannot be retagged through the public API.
/// ```compile_fail
/// use minion_agent::execution::{LocalSubprocess, Platform};
/// let mut provider = LocalSubprocess::new(".");
/// provider.platform = Platform::Posix;
/// assert_eq!(provider.platform, Platform::Posix);
/// ```
/// ```
/// use minion_agent::execution::{LocalSubprocess, Platform, Subprocess};
/// let provider = LocalSubprocess::new(".");
/// assert_eq!(provider.platform(), Platform::local());
/// ```
pub struct LocalSubprocess {
    cwd: PathBuf,
    base_env: BTreeMap<String, String>,
    logical_base_env: Option<Vec<(ResultString, ResultString)>>,
    logical_cwd: Option<ResultString>,
    world: ExecutionWorldIdentity,
    #[cfg(test)]
    native_spawn_calls: Option<Arc<std::sync::atomic::AtomicUsize>>,
}

impl LocalSubprocess {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self::with_world(cwd, ExecutionWorldIdentity::local())
    }

    pub fn with_world(cwd: impl Into<PathBuf>, world: ExecutionWorldIdentity) -> Self {
        Self {
            cwd: cwd.into(),
            base_env: std::env::vars().collect(),
            logical_base_env: None,
            logical_cwd: None,
            world,
            #[cfg(test)]
            native_spawn_calls: None,
        }
    }

    pub fn with_base_env(mut self, base_env: BTreeMap<String, String>) -> Self {
        self.base_env = base_env;
        self.logical_base_env = None;
        self
    }

    pub fn with_lossless_base_env(mut self, entries: Vec<(ResultString, ResultString)>) -> Self {
        self.base_env = entries
            .iter()
            .map(|(n, v)| {
                (
                    String::from_utf16_lossy(n.code_units()),
                    String::from_utf16_lossy(v.code_units()),
                )
            })
            .collect();
        self.logical_base_env = Some(entries);
        self
    }

    pub fn with_logical_cwd(mut self, cwd: ResultString) -> Self {
        self.cwd = super::path::native(&cwd);
        self.logical_cwd = Some(cwd);
        self
    }
}

#[async_trait]
impl Subprocess for LocalSubprocess {
    fn logical_cwd(&self) -> ResultString {
        self.logical_cwd
            .clone()
            .unwrap_or_else(|| super::path::from_native(&self.cwd))
    }

    fn spawn_environment(&self) -> Vec<(ResultString, ResultString)> {
        self.logical_base_env.clone().unwrap_or_else(|| {
            self.base_env
                .iter()
                .map(|(n, v)| (n.as_str().into(), v.as_str().into()))
                .collect()
        })
    }
    fn platform(&self) -> Platform {
        Platform::local()
    }

    fn base_env(&self) -> EnvSnapshot {
        EnvSnapshot::configured(self.platform(), &self.base_env)
    }

    fn cwd(&self) -> &Path {
        &self.cwd
    }

    fn execution_world(&self) -> &ExecutionWorldIdentity {
        &self.world
    }

    async fn spawn(
        &self,
        argv: &[String],
        options: SpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError> {
        let argv: Vec<_> = argv.iter().map(|a| a.as_str().into()).collect();
        self.spawn_lossless(&argv, options.into()).await
    }

    async fn spawn_lossless(
        &self,
        argv: &[ResultString],
        options: LosslessSpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError> {
        let (cwd, env) = prepare_spawn(self, argv, &options)?;
        let mut command = Command::new(String::from_utf16_lossy(argv[0].code_units()));
        command.args(
            argv[1..]
                .iter()
                .map(|a| String::from_utf16_lossy(a.code_units())),
        );
        command.current_dir(super::path::native(&cwd));
        command.env_clear();
        command.envs(env.into_iter().map(|(n, v)| {
            (
                String::from_utf16_lossy(n.code_units()),
                String::from_utf16_lossy(v.code_units()),
            )
        }));
        command.stdin(to_stdio(options.stdin));
        command.stdout(to_stdio(options.stdout));
        command.stderr(to_stdio(options.stderr));
        command.kill_on_drop(false);
        #[cfg(unix)]
        command.process_group(0);

        #[cfg(test)]
        if let Some(calls) = &self.native_spawn_calls {
            calls.fetch_add(1, Ordering::SeqCst);
        }
        let mut child = command.spawn().map_err(map_spawn_error)?;
        let pid = child.id().ok_or_else(|| {
            SubprocessError::new(
                SubprocessErrorCode::SpawnError,
                "spawn returned no process id",
            )
        })?;
        let stdin = child.stdin.take().map(LocalWritableStream::new);
        let stdout = child.stdout.take().map(LocalReadableStream::stdout);
        let stderr = child.stderr.take().map(LocalReadableStream::stderr);
        Ok(Arc::new(LocalProcess::new(
            pid,
            child,
            stdin,
            stdout,
            stderr,
            options.signal,
        )))
    }
}

const CAUSE_NONE: u8 = 0;
const CAUSE_SIGNAL: u8 = 1;
const CAUSE_EXPLICIT: u8 = 2;

pub struct LocalProcess {
    pid: u32,
    stdin: Option<Arc<LocalWritableStream>>,
    stdout: Option<Arc<LocalReadableStream>>,
    stderr: Option<Arc<LocalReadableStream>>,
    outcome: watch::Receiver<Option<Result<ExitStatus, SubprocessError>>>,
    cause: Arc<AtomicU8>,
    terminate: Arc<Notify>,
}

impl LocalProcess {
    fn new(
        pid: u32,
        child: Child,
        stdin: Option<LocalWritableStream>,
        stdout: Option<LocalReadableStream>,
        stderr: Option<LocalReadableStream>,
        signal: Option<Arc<dyn AbortSignal>>,
    ) -> Self {
        let (sender, outcome) = watch::channel(None);
        let cause = Arc::new(AtomicU8::new(CAUSE_NONE));
        let terminate = Arc::new(Notify::new());
        tokio::spawn(monitor_child(
            child,
            signal,
            Arc::clone(&cause),
            Arc::clone(&terminate),
            sender,
        ));
        Self {
            pid,
            stdin: stdin.map(Arc::new),
            stdout: stdout.map(Arc::new),
            stderr: stderr.map(Arc::new),
            outcome,
            cause,
            terminate,
        }
    }
}

#[async_trait]
impl Process for LocalProcess {
    fn pid(&self) -> u32 {
        self.pid
    }

    fn stdin(&self) -> Option<Arc<dyn WritableStream>> {
        self.stdin
            .as_ref()
            .map(|stream| Arc::clone(stream) as Arc<dyn WritableStream>)
    }

    fn stdout(&self) -> Option<Arc<dyn ReadableStream>> {
        self.stdout
            .as_ref()
            .map(|stream| Arc::clone(stream) as Arc<dyn ReadableStream>)
    }

    fn stderr(&self) -> Option<Arc<dyn ReadableStream>> {
        self.stderr
            .as_ref()
            .map(|stream| Arc::clone(stream) as Arc<dyn ReadableStream>)
    }

    async fn wait(&self) -> Result<ExitStatus, SubprocessError> {
        let mut receiver = self.outcome.clone();
        loop {
            if let Some(outcome) = receiver.borrow().clone() {
                return outcome;
            }
            receiver.changed().await.map_err(|_| {
                SubprocessError::new(
                    SubprocessErrorCode::Unknown,
                    "process monitor ended without an outcome",
                )
            })?;
        }
    }

    async fn terminate(&self) {
        // EXEC-012: after settlement this operation also releases handles that
        // surviving descendants may still hold open. Never hold the outcome
        // channel's borrow across a stream operation.
        let settled = self.outcome.borrow().is_some();
        if self
            .cause
            .compare_exchange(
                CAUSE_NONE,
                CAUSE_EXPLICIT,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.terminate.notify_one();
        }
        if settled {
            if let Some(stdout) = &self.stdout {
                stdout.close().await;
            }
            if let Some(stderr) = &self.stderr {
                stderr.close().await;
            }
            if let Some(stdin) = &self.stdin {
                stdin.close().await;
            }
        }
    }
}

struct LocalWritableStream {
    inner: Mutex<Option<ChildStdin>>,
}

impl LocalWritableStream {
    fn new(stdin: ChildStdin) -> Self {
        Self {
            inner: Mutex::new(Some(stdin)),
        }
    }
}

#[async_trait]
impl WritableStream for LocalWritableStream {
    async fn write(&self, data: &[u8]) -> Result<(), SubprocessError> {
        let mut guard = self.inner.lock().await;
        let Some(stdin) = guard.as_mut() else {
            return Err(SubprocessError::new(
                SubprocessErrorCode::PipeError,
                "stdin is closed",
            ));
        };
        stdin.write_all(data).await.map_err(map_pipe_error)
    }

    async fn close(&self) {
        let mut guard = self.inner.lock().await;
        if let Some(mut stdin) = guard.take() {
            let _ = stdin.shutdown().await;
        }
    }
}

enum ChildReader {
    Stdout(ChildStdout),
    Stderr(ChildStderr),
}

struct LocalReadableStream {
    inner: std::sync::Mutex<Option<ChildReader>>,
    read_lock: Mutex<()>,
    closed: watch::Sender<bool>,
}

impl LocalReadableStream {
    fn stdout(stdout: ChildStdout) -> Self {
        Self::new(ChildReader::Stdout(stdout))
    }

    fn stderr(stderr: ChildStderr) -> Self {
        Self::new(ChildReader::Stderr(stderr))
    }

    fn new(reader: ChildReader) -> Self {
        let (closed, _) = watch::channel(false);
        Self {
            inner: std::sync::Mutex::new(Some(reader)),
            read_lock: Mutex::new(()),
            closed,
        }
    }
}

#[async_trait]
impl ReadableStream for LocalReadableStream {
    async fn read_chunk(&self) -> Result<Option<Vec<u8>>, SubprocessError> {
        let mut closed = self.closed.subscribe();
        if *closed.borrow() {
            return Ok(None);
        }
        // Serialize consumers, but never require this permit to close the pipe:
        // a caller may park a Pending read future without polling it again.
        let _permit = tokio::select! {
            biased;
            _ = closed.changed() => return Ok(None),
            guard = self.read_lock.lock() => guard,
        };
        let mut buffer = vec![0; 8192];
        let read = std::future::poll_fn(|cx| {
            // Only a single nonblocking I/O poll holds the reader lock. No lock
            // guarding the OS handle survives a Pending return.
            let mut guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
            if *self.closed.borrow() {
                guard.take();
                return std::task::Poll::Ready(Ok(0));
            }
            let Some(reader) = guard.as_mut() else {
                return std::task::Poll::Ready(Ok(0));
            };
            let mut destination = ReadBuf::new(&mut buffer);
            let result = match reader {
                ChildReader::Stdout(stream) => Pin::new(stream).poll_read(cx, &mut destination),
                ChildReader::Stderr(stream) => Pin::new(stream).poll_read(cx, &mut destination),
            };
            match result {
                std::task::Poll::Ready(Ok(())) => {
                    let count = destination.filled().len();
                    if count == 0 {
                        guard.take(); // natural EOF releases this OS handle
                    }
                    std::task::Poll::Ready(Ok(count))
                }
                std::task::Poll::Ready(Err(error)) => std::task::Poll::Ready(Err(error)),
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        });
        let result = tokio::select! {
            biased;
            _ = closed.changed() => None,
            result = read => Some(result),
        };
        if *closed.borrow() || result.is_none() {
            return Ok(None);
        }
        let count = result
            .expect("read branch returned a result")
            .map_err(map_pipe_error)?;
        if count == 0 {
            Ok(None)
        } else {
            buffer.truncate(count);
            Ok(Some(buffer))
        }
    }

    async fn close(&self) {
        self.closed.send_replace(true);
        // This never waits for a read future's serialization permit. Taking the
        // reader drops exactly this OS handle, even with retained public Arcs.
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
    }
}

async fn monitor_child(
    mut child: Child,
    signal: Option<Arc<dyn AbortSignal>>,
    cause: Arc<AtomicU8>,
    terminate: Arc<Notify>,
    outcome: watch::Sender<Option<Result<ExitStatus, SubprocessError>>>,
) {
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(error) => break Err(error),
        }
        if signal.as_ref().is_some_and(|signal| signal.aborted()) {
            let _ = cause.compare_exchange(
                CAUSE_NONE,
                CAUSE_SIGNAL,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
        if cause.load(Ordering::Acquire) != CAUSE_NONE {
            let helper = child.id().map(|pid| tokio::spawn(kill_process_tree(pid)));
            break wait_for_exit_or_kill_helper(&mut child, helper).await;
        }
        tokio::select! {
            result = child.wait() => break result,
            () = terminate.notified() => {},
            () = tokio::time::sleep(Duration::from_millis(5)) => {},
        }
    };
    let result = match status {
        Ok(status) => classify_exit(cause.load(Ordering::Acquire), status.code()),
        Err(error) => Err(SubprocessError::new(
            SubprocessErrorCode::Unknown,
            error.to_string(),
        )),
    };
    outcome.send_replace(Some(result));
}

fn classify_exit(cause: u8, exit_code: Option<i32>) -> Result<ExitStatus, SubprocessError> {
    if cause == CAUSE_SIGNAL {
        Err(SubprocessError::new(
            SubprocessErrorCode::Aborted,
            "operation aborted",
        ))
    } else {
        Ok(ExitStatus { exit_code })
    }
}

async fn wait_for_exit_or_kill_helper(
    child: &mut Child,
    helper: Option<tokio::task::JoinHandle<bool>>,
) -> io::Result<std::process::ExitStatus> {
    let Some(mut helper) = helper else {
        return child.wait().await;
    };
    tokio::select! {
        status = child.wait() => status,
        helper_result = &mut helper => {
            if !helper_result.unwrap_or(false) {
                let _ = child.start_kill();
            }
            child.wait().await
        }
    }
}

async fn kill_process_tree(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let group = format!("-{pid}");
        if Command::new("kill")
            // A negative process-group operand must not be parsed as an option
            // by procps-ng kill. Without `--`, it can signal the caller's group.
            .args(["-KILL", "--", &group])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_ok_and(|status| status.success())
        {
            return true;
        }
    }
    #[cfg(windows)]
    {
        if Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_ok_and(|status| status.success())
        {
            return true;
        }
    }
    false
}

fn to_stdio(mode: StdioMode) -> Stdio {
    match mode {
        StdioMode::Inherit => Stdio::inherit(),
        StdioMode::Piped => Stdio::piped(),
        StdioMode::Null => Stdio::null(),
    }
}

fn map_spawn_error(error: io::Error) -> SubprocessError {
    SubprocessError::new(SubprocessErrorCode::SpawnError, error.to_string())
}

fn map_pipe_error(error: io::Error) -> SubprocessError {
    SubprocessError::new(SubprocessErrorCode::PipeError, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn group_kill_terminates_ready_parent_and_pipe_holding_descendant() {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

        // Readiness is emitted only after the descendant has been created in this
        // isolated process group. No spawn/termination race or sleep-based assertion.
        let mut child = Command::new("sh")
            .args(["-c", "sleep 60 & printf '%s\\n' \"$!\"; wait"])
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let pid = child.id().unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut ready = String::new();
        tokio::time::timeout(Duration::from_secs(5), output.read_line(&mut ready))
            .await
            .expect("descendant readiness")
            .unwrap();
        let descendant: u32 = ready.trim().parse().unwrap();
        assert_ne!(descendant, pid);

        let helper_succeeded = kill_process_tree(pid).await;
        let settled = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
        let mut remaining = Vec::new();
        let eof =
            tokio::time::timeout(Duration::from_secs(3), output.read_to_end(&mut remaining)).await;
        // Also clean up the known-bad run before asserting: never leak the live
        // group merely because the helper lied about success or addressed no PID.
        if settled.is_err() || eof.is_err() {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{pid}")])
                .status()
                .await;
            let _ = child.wait().await;
        }
        assert!(
            helper_succeeded,
            "the actual group helper must report success"
        );
        assert_eq!(
            settled
                .expect("the ready parent must be killed")
                .unwrap()
                .code(),
            None
        );
        assert_eq!(
            eof.expect("the descendant must release its inherited pipe")
                .unwrap(),
            0
        );
    }

    fn short_lived_child() -> Child {
        let mut command = if cfg!(windows) {
            let mut command = Command::new("cmd.exe");
            command.args(["/C", "exit 0"]);
            command
        } else {
            let mut command = Command::new("sh");
            command.args(["-c", "exit 0"]);
            command
        };
        command.spawn().unwrap()
    }

    fn long_lived_child() -> Child {
        let mut command = if cfg!(windows) {
            let mut command = Command::new("cmd.exe");
            command.args(["/C", "ping 127.0.0.1 -n 10 >NUL"]);
            command
        } else {
            let mut command = Command::new("sh");
            command.args(["-c", "sleep 10"]);
            command
        };
        command.spawn().unwrap()
    }

    #[test]
    fn explicit_termination_preserves_the_os_reported_exit_code() {
        assert_eq!(
            classify_exit(CAUSE_EXPLICIT, Some(23)).unwrap(),
            ExitStatus {
                exit_code: Some(23)
            }
        );
        assert_eq!(
            classify_exit(CAUSE_EXPLICIT, None).unwrap(),
            ExitStatus { exit_code: None }
        );
    }

    #[test]
    fn signal_termination_remains_an_aborted_error() {
        assert_eq!(
            classify_exit(CAUSE_SIGNAL, Some(23)).unwrap_err().code,
            SubprocessErrorCode::Aborted
        );
    }

    #[test]
    fn first_successful_cause_claim_wins_a_signal_natural_exit_race() {
        let cause = AtomicU8::new(CAUSE_NONE);
        assert!(
            cause
                .compare_exchange(
                    CAUSE_NONE,
                    CAUSE_SIGNAL,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        );
        assert!(
            cause
                .compare_exchange(
                    CAUSE_NONE,
                    CAUSE_EXPLICIT,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
        );
        assert_eq!(
            classify_exit(cause.load(Ordering::Acquire), Some(0))
                .unwrap_err()
                .code,
            SubprocessErrorCode::Aborted
        );
    }

    #[tokio::test]
    async fn failed_kill_helper_does_not_reclassify_an_already_claimed_signal() {
        let mut child = long_lived_child();
        let helper = tokio::spawn(async { false });
        let status = tokio::time::timeout(
            Duration::from_secs(5),
            wait_for_exit_or_kill_helper(&mut child, Some(helper)),
        )
        .await
        .expect("fallback kill must settle")
        .unwrap();
        assert_eq!(
            classify_exit(CAUSE_SIGNAL, status.code()).unwrap_err().code,
            SubprocessErrorCode::Aborted
        );
    }

    #[tokio::test]
    async fn target_exit_settles_without_waiting_for_a_hung_kill_helper() {
        let mut child = short_lived_child();
        let helper = tokio::spawn(async {
            std::future::pending::<()>().await;
            true
        });
        let status = tokio::time::timeout(
            Duration::from_secs(5),
            wait_for_exit_or_kill_helper(&mut child, Some(helper)),
        )
        .await
        .expect("target exit must be the only settlement dependency")
        .unwrap();
        assert_eq!(status.code(), Some(0));
    }
}
