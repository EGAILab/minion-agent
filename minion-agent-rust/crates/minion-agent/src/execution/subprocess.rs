use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command},
    sync::{Mutex, Notify, watch},
};

use super::{
    AbortSignal, ExecutionWorldIdentity, SubprocessError, SubprocessErrorCode,
    filesystem::resolve_local_path,
};

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
    async fn spawn(
        &self,
        argv: &[String],
        options: SpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError>;
}

#[derive(Clone, Debug)]
pub struct LocalSubprocess {
    cwd: PathBuf,
    base_env: BTreeMap<String, String>,
    world: ExecutionWorldIdentity,
}

impl LocalSubprocess {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self::with_world(cwd, ExecutionWorldIdentity::local())
    }

    pub fn with_world(cwd: impl Into<PathBuf>, world: ExecutionWorldIdentity) -> Self {
        Self {
            cwd: cwd.into(),
            base_env: std::env::vars().collect(),
            world,
        }
    }

    pub fn with_base_env(mut self, base_env: BTreeMap<String, String>) -> Self {
        self.base_env = base_env;
        self
    }

    fn resolved_cwd(&self, cwd: Option<&Path>) -> PathBuf {
        match cwd {
            Some(cwd) => resolve_local_path(&self.cwd, &cwd.to_string_lossy()),
            None => self.cwd.clone(),
        }
    }
}

#[async_trait]
impl Subprocess for LocalSubprocess {
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
        if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted())
        {
            return Err(SubprocessError::new(
                SubprocessErrorCode::Aborted,
                "operation aborted",
            ));
        }
        let Some(program) = argv.first() else {
            return Err(SubprocessError::new(
                SubprocessErrorCode::SpawnError,
                "argv must contain a program",
            ));
        };
        let mut command = Command::new(program);
        command.args(&argv[1..]);
        command.current_dir(self.resolved_cwd(options.cwd.as_deref()));
        if !options.inherit_env {
            command.env_clear();
        } else {
            command.env_clear();
            command.envs(&self.base_env);
        }
        command.envs(&options.env);
        command.stdin(to_stdio(options.stdin));
        command.stdout(to_stdio(options.stdout));
        command.stderr(to_stdio(options.stderr));
        command.kill_on_drop(false);
        #[cfg(unix)]
        command.process_group(0);

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
    inner: Mutex<ChildReader>,
}

impl LocalReadableStream {
    fn stdout(stdout: ChildStdout) -> Self {
        Self {
            inner: Mutex::new(ChildReader::Stdout(stdout)),
        }
    }

    fn stderr(stderr: ChildStderr) -> Self {
        Self {
            inner: Mutex::new(ChildReader::Stderr(stderr)),
        }
    }
}

#[async_trait]
impl ReadableStream for LocalReadableStream {
    async fn read_chunk(&self) -> Result<Option<Vec<u8>>, SubprocessError> {
        let mut guard = self.inner.lock().await;
        let mut buffer = vec![0; 8192];
        let count = match &mut *guard {
            ChildReader::Stdout(stream) => stream.read(&mut buffer).await,
            ChildReader::Stderr(stream) => stream.read(&mut buffer).await,
        }
        .map_err(map_pipe_error)?;
        if count == 0 {
            Ok(None)
        } else {
            buffer.truncate(count);
            Ok(Some(buffer))
        }
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
            .args(["-KILL", &group])
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
