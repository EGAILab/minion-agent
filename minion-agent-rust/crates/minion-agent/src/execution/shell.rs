use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Weak},
    time::Duration,
};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::{
    AbortSignal, ExecutionWorldIdentity, Process, ShellError, ShellErrorCode, SpawnOptions,
    StdioMode, Subprocess, SubprocessErrorCode, filesystem::resolve_local_path,
};

pub type StreamCallback = Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>;

#[derive(Clone)]
pub struct ShellExecOptions {
    pub cwd: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    pub inherit_env: bool,
    pub timeout_seconds: Option<f64>,
    pub signal: Option<Arc<dyn AbortSignal>>,
    pub on_stdout: Option<StreamCallback>,
    pub on_stderr: Option<StreamCallback>,
}

impl Default for ShellExecOptions {
    fn default() -> Self {
        Self {
            cwd: None,
            env: BTreeMap::new(),
            inherit_env: true,
            timeout_seconds: None,
            signal: None,
            on_stdout: None,
            on_stderr: None,
        }
    }
}

impl ShellExecOptions {
    pub fn inherited() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShellOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

#[async_trait]
pub trait Shell: Send + Sync {
    fn execution_world(&self) -> &ExecutionWorldIdentity;
    async fn exec(
        &self,
        command: &str,
        options: ShellExecOptions,
    ) -> Result<ShellOutput, ShellError>;
    async fn cleanup(&self);
}

pub struct LocalShell {
    subprocess: Arc<dyn Subprocess>,
    configured_shell: Option<PathBuf>,
    active: Mutex<BTreeMap<u32, Weak<dyn Process>>>,
}

impl LocalShell {
    pub fn new(subprocess: Arc<dyn Subprocess>) -> Self {
        Self {
            subprocess,
            configured_shell: None,
            active: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn with_configured_shell(mut self, shell: impl Into<PathBuf>) -> Self {
        self.configured_shell = Some(shell.into());
        self
    }

    fn resolved_cwd(&self, requested: Option<&Path>) -> PathBuf {
        match requested {
            Some(path) => resolve_local_path(self.subprocess.cwd(), &path.to_string_lossy()),
            None => self.subprocess.cwd().to_owned(),
        }
    }

    fn resolve_shell(&self) -> Result<PathBuf, ShellError> {
        if let Some(configured) = &self.configured_shell {
            if configured.is_file() {
                return Ok(configured.clone());
            }
            return Err(ShellError::new(
                ShellErrorCode::ShellUnavailable,
                format!("configured shell does not exist: {}", configured.display()),
            ));
        }
        #[cfg(windows)]
        {
            for variable in ["ProgramFiles", "ProgramFiles(x86)"] {
                if let Some(root) = std::env::var_os(variable) {
                    let candidate = PathBuf::from(root).join(r"Git\bin\bash.exe");
                    if candidate.is_file() {
                        return Ok(candidate);
                    }
                }
            }
        }
        #[cfg(not(windows))]
        {
            let candidate = PathBuf::from("/bin/bash");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        if let Some(path) = find_on_path(if cfg!(windows) { "bash.exe" } else { "bash" }) {
            return Ok(path);
        }
        #[cfg(not(windows))]
        if let Some(path) = find_on_path("sh") {
            return Ok(path);
        }
        Err(ShellError::new(
            ShellErrorCode::ShellUnavailable,
            "no usable bash installation was found",
        ))
    }
}

#[async_trait]
impl Shell for LocalShell {
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        self.subprocess.execution_world()
    }

    async fn exec(
        &self,
        command: &str,
        options: ShellExecOptions,
    ) -> Result<ShellOutput, ShellError> {
        if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted())
        {
            return Err(ShellError::new(
                ShellErrorCode::Aborted,
                "operation aborted",
            ));
        }
        if let Some(timeout) = options.timeout_seconds
            && (!timeout.is_finite() || timeout <= 0.0 || timeout * 1000.0 > 2_147_483_647.0)
        {
            return Err(ShellError::new(
                ShellErrorCode::Timeout,
                "invalid shell timeout",
            ));
        }
        let cwd = self.resolved_cwd(options.cwd.as_deref());
        let shell = self.resolve_shell()?;
        if !cwd.exists() {
            return Err(ShellError::new(
                ShellErrorCode::SpawnError,
                format!("working directory does not exist: {}", cwd.display()),
            ));
        }

        let argv = vec![
            shell.to_string_lossy().into_owned(),
            "-c".to_owned(),
            command.to_owned(),
        ];
        let process = self
            .subprocess
            .spawn(
                &argv,
                SpawnOptions {
                    cwd: Some(cwd),
                    env: options.env.clone(),
                    inherit_env: options.inherit_env,
                    stdin: StdioMode::Null,
                    stdout: StdioMode::Piped,
                    stderr: StdioMode::Piped,
                    signal: options.signal.clone(),
                },
            )
            .await
            .map_err(|error| {
                ShellError::new(
                    if error.code == SubprocessErrorCode::Aborted {
                        ShellErrorCode::Aborted
                    } else {
                        ShellErrorCode::SpawnError
                    },
                    error.message,
                )
            })?;
        let pid = process.pid();
        self.active.lock().insert(pid, Arc::downgrade(&process));

        let (sender, mut receiver) = mpsc::unbounded_channel();
        if let Some(stdout) = process.stdout() {
            spawn_reader(stdout, OutputKind::Stdout, sender.clone());
        } else {
            let _ = sender.send(OutputEvent::Eof(OutputKind::Stdout));
        }
        if let Some(stderr) = process.stderr() {
            spawn_reader(stderr, OutputKind::Stderr, sender.clone());
        } else {
            let _ = sender.send(OutputEvent::Eof(OutputKind::Stderr));
        }
        drop(sender);

        let started = tokio::time::Instant::now();
        let deadline = options
            .timeout_seconds
            .map(|seconds| started + Duration::from_secs_f64(seconds));
        let mut wait = Box::pin(process.wait());
        let mut process_result = None;
        let mut stdout_done = false;
        let mut stderr_done = false;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut stdout_decoder = Utf8Decoder::default();
        let mut stderr_decoder = Utf8Decoder::default();
        let mut callback_error = None;
        let mut timeout_fired = false;
        let mut grace_deadline = None;
        let mut events_open = true;

        loop {
            if process_result.is_some()
                && (stdout_done && stderr_done
                    || grace_deadline.is_some_and(|grace| tokio::time::Instant::now() >= grace))
            {
                break;
            }
            if !timeout_fired
                && process_result.is_none()
                && deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline)
            {
                timeout_fired = true;
                process.terminate().await;
            }
            tokio::select! {
                biased;
                event = receiver.recv(), if events_open => {
                    match event {
                        Some(OutputEvent::Data(kind, bytes)) => {
                            let (destination, decoder, callback) = match kind {
                                OutputKind::Stdout => {
                                    (&mut stdout, &mut stdout_decoder, options.on_stdout.as_ref())
                                }
                                OutputKind::Stderr => {
                                    (&mut stderr, &mut stderr_decoder, options.on_stderr.as_ref())
                                }
                            };
                            destination.extend_from_slice(&bytes);
                            let chunk = decoder.push(&bytes);
                            if callback_error.is_none()
                                && !chunk.is_empty()
                                && let Some(callback) = callback
                                && let Err(error) = callback(&chunk)
                            {
                                callback_error = Some(error);
                                process.terminate().await;
                            }
                            if process_result.is_some() {
                                grace_deadline = Some(tokio::time::Instant::now() + Duration::from_millis(100));
                            }
                        }
                        Some(OutputEvent::Eof(kind)) => {
                            let (decoder, callback) = match kind {
                                OutputKind::Stdout => (&mut stdout_decoder, options.on_stdout.as_ref()),
                                OutputKind::Stderr => (&mut stderr_decoder, options.on_stderr.as_ref()),
                            };
                            let tail = decoder.finish();
                            if callback_error.is_none()
                                && !tail.is_empty()
                                && let Some(callback) = callback
                                && let Err(error) = callback(&tail)
                            {
                                callback_error = Some(error);
                                process.terminate().await;
                            }
                            match kind {
                                OutputKind::Stdout => stdout_done = true,
                                OutputKind::Stderr => stderr_done = true,
                            }
                        }
                        Some(OutputEvent::Error(kind, error)) => {
                            let diagnostic = format!("[stream error: {error}]");
                            match kind {
                                OutputKind::Stdout => stdout.extend_from_slice(diagnostic.as_bytes()),
                                OutputKind::Stderr => stderr.extend_from_slice(diagnostic.as_bytes()),
                            }
                            match kind {
                                OutputKind::Stdout => stdout_done = true,
                                OutputKind::Stderr => stderr_done = true,
                            }
                        }
                        None => {
                            stdout_done = true;
                            stderr_done = true;
                            events_open = false;
                        }
                    }
                }
                result = &mut wait, if process_result.is_none() => {
                    process_result = Some(result);
                    grace_deadline = Some(tokio::time::Instant::now() + Duration::from_millis(100));
                }
                () = tokio::time::sleep(Duration::from_millis(5)) => {}
            }
        }
        self.active.lock().remove(&pid);

        if let Some(error) = callback_error {
            return Err(ShellError::new(ShellErrorCode::CallbackError, error));
        }
        if timeout_fired {
            return Err(ShellError::new(
                ShellErrorCode::Timeout,
                "shell command timed out",
            ));
        }
        let status = process_result.expect("process result is set before completion");
        match status {
            Ok(status) => Ok(ShellOutput {
                stdout: String::from_utf8_lossy(&stdout).into_owned(),
                stderr: String::from_utf8_lossy(&stderr).into_owned(),
                exit_code: status.exit_code.unwrap_or(0),
            }),
            Err(error) if error.code == SubprocessErrorCode::Aborted => {
                Err(ShellError::new(ShellErrorCode::Aborted, error.message))
            }
            Err(error) => Err(ShellError::new(ShellErrorCode::Unknown, error.message)),
        }
    }

    async fn cleanup(&self) {
        let active: Vec<_> = self
            .active
            .lock()
            .values()
            .filter_map(Weak::upgrade)
            .collect();
        for process in active {
            process.terminate().await;
        }
        self.active.lock().clear();
    }
}

#[derive(Clone, Copy)]
enum OutputKind {
    Stdout,
    Stderr,
}

enum OutputEvent {
    Data(OutputKind, Vec<u8>),
    Eof(OutputKind),
    Error(OutputKind, String),
}

fn spawn_reader(
    stream: Arc<dyn super::ReadableStream>,
    kind: OutputKind,
    sender: mpsc::UnboundedSender<OutputEvent>,
) {
    tokio::spawn(async move {
        loop {
            match stream.read_chunk().await {
                Ok(Some(bytes)) => {
                    if sender.send(OutputEvent::Data(kind, bytes)).is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = sender.send(OutputEvent::Eof(kind));
                    break;
                }
                Err(error) => {
                    let _ = sender.send(OutputEvent::Error(kind, error.message));
                    break;
                }
            }
        }
    });
}

fn find_on_path(program: &str) -> Option<PathBuf> {
    let candidate = Path::new(program);
    if candidate.components().count() > 1 && candidate.is_file() {
        return Some(candidate.to_owned());
    }
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|directory| directory.join(program))
            .find(|candidate| candidate.is_file())
    })
}

#[derive(Default)]
struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    output.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    output.push_str(
                        std::str::from_utf8(&self.pending[..valid_up_to])
                            .expect("valid_up_to identifies valid UTF-8"),
                    );
                    match error.error_len() {
                        Some(length) => {
                            output.push('\u{fffd}');
                            self.pending.drain(..valid_up_to + length);
                        }
                        None => {
                            self.pending.drain(..valid_up_to);
                            break;
                        }
                    }
                }
            }
        }
        output
    }

    fn finish(&mut self) -> String {
        let result = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        result
    }
}
