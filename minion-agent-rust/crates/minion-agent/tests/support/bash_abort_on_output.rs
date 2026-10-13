//! Test-only output tap. No tool updates or product cancellation semantics.
use super::capabilities::{
    EnvSnapshot, ExecutionWorldIdentity, ExitStatus, Platform, Process, ReadableStream,
    SpawnOptions, Subprocess, SubprocessError, WritableStream,
};
use async_trait::async_trait;
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

struct Trigger {
    marker: String,
    bytes: Mutex<Vec<u8>>,
    fired: AtomicBool,
    cancel: Arc<dyn Fn() + Send + Sync>,
}

impl Trigger {
    fn observe(&self, chunk: &[u8]) {
        if self.fired.load(Ordering::SeqCst) {
            return;
        }
        let mut bytes = self.bytes.lock().unwrap();
        bytes.extend_from_slice(chunk);
        // Do not expose an incomplete UTF-8 suffix as a premature replacement
        // character. Complete malformed subsequences use ordinary lossy decode.
        let mut end = 0;
        while end < bytes.len() {
            match std::str::from_utf8(&bytes[end..]) {
                Ok(_) => {
                    end = bytes.len();
                    break;
                }
                Err(error) => {
                    end += error.valid_up_to();
                    if let Some(length) = error.error_len() {
                        end += length;
                    } else {
                        break;
                    }
                }
            }
        }
        if String::from_utf8_lossy(&bytes[..end]).contains(&self.marker)
            && !self.fired.swap(true, Ordering::SeqCst)
        {
            let cancel = self.cancel.clone();
            // The read completes in this poll, so the tool receives/appends its
            // chunk before the spawned task can cancel on a later loop turn.
            tokio::spawn(async move {
                tokio::task::yield_now().await;
                cancel();
            });
        }
    }
}

struct Stream {
    inner: Arc<dyn ReadableStream>,
    trigger: Arc<Trigger>,
}
#[async_trait]
impl ReadableStream for Stream {
    async fn read_chunk(&self) -> Result<Option<Vec<u8>>, SubprocessError> {
        let chunk = self.inner.read_chunk().await?;
        if let Some(bytes) = &chunk {
            self.trigger.observe(bytes);
        }
        Ok(chunk)
    }
    async fn close(&self) {
        self.inner.close().await;
    }
}

struct TappedProcess {
    inner: Arc<dyn Process>,
    stdout: Option<Arc<dyn ReadableStream>>,
    stderr: Option<Arc<dyn ReadableStream>>,
}
impl TappedProcess {
    fn new(inner: Arc<dyn Process>, trigger: Arc<Trigger>) -> Self {
        let wrap = |stream: Option<Arc<dyn ReadableStream>>| {
            stream.map(|inner| {
                Arc::new(Stream {
                    inner,
                    trigger: trigger.clone(),
                }) as Arc<dyn ReadableStream>
            })
        };
        Self {
            stdout: wrap(inner.stdout()),
            stderr: wrap(inner.stderr()),
            inner,
        }
    }
}
#[async_trait]
impl Process for TappedProcess {
    fn pid(&self) -> u32 {
        self.inner.pid()
    }
    fn stdin(&self) -> Option<Arc<dyn WritableStream>> {
        self.inner.stdin()
    }
    fn stdout(&self) -> Option<Arc<dyn ReadableStream>> {
        self.stdout.clone()
    }
    fn stderr(&self) -> Option<Arc<dyn ReadableStream>> {
        self.stderr.clone()
    }
    async fn wait(&self) -> Result<ExitStatus, SubprocessError> {
        self.inner.wait().await
    }
    async fn terminate(&self) {
        self.inner.terminate().await;
    }
}

pub struct AbortOnOutput {
    inner: Arc<dyn Subprocess>,
    trigger: Arc<Trigger>,
}
impl AbortOnOutput {
    pub fn new(
        inner: Arc<dyn Subprocess>,
        marker: String,
        cancel: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        assert!(!marker.is_empty(), "abort_on_output must be nonempty");
        Self {
            inner,
            trigger: Arc::new(Trigger {
                marker,
                bytes: Mutex::default(),
                fired: AtomicBool::new(false),
                cancel,
            }),
        }
    }
}
#[async_trait]
impl Subprocess for AbortOnOutput {
    fn cwd(&self) -> &Path {
        self.inner.cwd()
    }
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        self.inner.execution_world()
    }
    fn platform(&self) -> Platform {
        self.inner.platform()
    }
    fn base_env(&self) -> EnvSnapshot {
        self.inner.base_env()
    }
    async fn spawn(
        &self,
        argv: &[String],
        options: SpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError> {
        // This wrapper is bound to one tool call. Its certified bash lookup
        // passes None; only that call's command carries a signal adapter.
        let command = options.signal.is_some();
        let process = self.inner.spawn(argv, options).await?;
        if command {
            Ok(Arc::new(TappedProcess::new(process, self.trigger.clone())))
        } else {
            Ok(process)
        }
    }
}

/// Only the command start is delayed; lookup is unchanged. Fixture, not timeout policy.
pub struct SlowCommand(pub Arc<dyn Subprocess>, pub Duration);
#[async_trait]
impl Subprocess for SlowCommand {
    fn cwd(&self) -> &Path {
        self.0.cwd()
    }
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        self.0.execution_world()
    }
    fn platform(&self) -> Platform {
        self.0.platform()
    }
    fn base_env(&self) -> EnvSnapshot {
        self.0.base_env()
    }
    async fn spawn(
        &self,
        argv: &[String],
        options: SpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError> {
        if options.signal.is_some() {
            tokio::time::sleep(self.1).await;
        }
        self.0.spawn(argv, options).await
    }
}
