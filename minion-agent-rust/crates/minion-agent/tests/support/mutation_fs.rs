//! Canonical-boundary provider fixture. No matching, queue, or tool semantics live here.
use async_trait::async_trait;
use minion_agent::{
    execution::{
        AbortSignal, ExecutionWorldIdentity, FileInfo, FileSystem, FsError, FsErrorCode, FsTarget,
        LocalFileSystem,
    },
    tools::ToolExecutionSignal,
};
use parking_lot::Mutex;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::Notify;

#[derive(Clone, Default)]
pub struct Signal(pub Arc<AtomicBool>);
impl Signal {
    pub fn abort(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
impl ToolExecutionSignal for Signal {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

pub struct Gate {
    pub spec: Value,
    answer: Mutex<Option<Option<FsErrorCode>>>,
    notify: Notify,
}
impl Gate {
    pub fn new(spec: Value) -> Self {
        Self {
            spec,
            answer: Mutex::new(None),
            notify: Notify::new(),
        }
    }
    pub fn release(&self, error: Option<FsErrorCode>) {
        *self.answer.lock() = Some(error);
        self.notify.notify_waiters();
    }
    async fn wait(&self) -> Option<FsErrorCode> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(answer) = *self.answer.lock() {
                return answer;
            }
            notified.await;
        }
    }
}

pub struct FixtureFs {
    pub local: LocalFileSystem,
    root: PathBuf,
    pub calls: Mutex<Vec<String>>,
    pub log: Arc<Mutex<Vec<String>>>,
    pub gates: Vec<Arc<Gate>>,
    pub id: String,
    pub provider: Value,
    counts: Mutex<BTreeMap<(String, String), usize>>,
    pub signal: Signal,
    pub abort_after: Option<String>,
    aborted_once: AtomicBool,
}
impl FixtureFs {
    pub fn new(
        root: &Path,
        id: &str,
        provider: Value,
        log: Arc<Mutex<Vec<String>>>,
        gates: Vec<Arc<Gate>>,
        signal: Signal,
        abort_after: Option<String>,
    ) -> Self {
        Self {
            local: LocalFileSystem::new(root),
            root: root.to_owned(),
            calls: Mutex::new(Vec::new()),
            log,
            gates,
            id: id.into(),
            provider,
            counts: Mutex::new(BTreeMap::new()),
            signal,
            abort_after,
            aborted_once: AtomicBool::new(false),
        }
    }
    fn relative(&self, path: &str) -> String {
        if Path::new(path).is_absolute() {
            let relative = Path::new(path)
                .strip_prefix(&self.root)
                .map_or_else(|_| path.to_owned(), |p| p.to_string_lossy().into_owned());
            if relative.is_empty() {
                ".".into()
            } else {
                relative.replace('\\', "/")
            }
        } else {
            path.to_owned()
        }
    }
    async fn begin(
        &self,
        op: &str,
        path: &minion_agent::execution::FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> (String, Option<FsErrorCode>) {
        assert!(
            signal.is_none(),
            "mutation tools must not forward cancellation to ctx.fs"
        );
        let path = self.relative(path.as_str().expect("scalar fixture path"));
        self.calls.lock().push(format!("{op} {path}"));
        let count = {
            let mut counts = self.counts.lock();
            let count = counts.entry((op.into(), path.clone())).or_default();
            *count += 1;
            *count
        };
        let prefix = format!("{} {op} {path} #{count}", self.id);
        self.log.lock().push(format!("{prefix} start"));
        let gate = self.gates.iter().find(|g| {
            g.spec["operation"] == op
                && g.spec["path"] == path
                && g.spec["provider"].as_str().unwrap_or("p") == self.id
                && g.spec["occurrence"].as_u64().unwrap_or(1) == count as u64
        });
        let gate_error = if let Some(gate) = gate {
            gate.wait().await
        } else {
            None
        };
        let script_error = self.provider[op]
            .as_array()
            .and_then(|a| a.iter().find(|s| s["path"] == path))
            .and_then(|s| s.get("error"))
            .map(|s| serde_json::from_value(s.clone()).unwrap());
        let unsupported = match op {
            "check_read_write" => self.provider["without_exec_009"] == true,
            "check_readable" => self.provider["without_exec_008"] == true,
            _ => false,
        };
        (
            prefix,
            gate_error
                .or(script_error)
                .or(unsupported.then_some(FsErrorCode::NotSupported)),
        )
    }
    fn finish<T>(
        &self,
        op: &str,
        prefix: String,
        answer: Result<T, FsError>,
    ) -> Result<T, FsError> {
        let suffix = answer.as_ref().map_or_else(
            |e| {
                serde_json::to_value(e.code)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned()
            },
            |_| "ok".into(),
        );
        self.log.lock().push(format!("{prefix} {suffix}"));
        if self.abort_after.as_deref() == Some(op)
            && !self.aborted_once.swap(true, Ordering::SeqCst)
        {
            self.signal.abort();
        }
        answer
    }
    fn error(code: FsErrorCode) -> FsError {
        FsError::new(code, "scripted provider answer")
    }
}

#[async_trait]
impl FileSystem for FixtureFs {
    fn cwd(&self) -> &Path {
        self.local.cwd()
    }
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        self.local.execution_world()
    }
    async fn absolute_path(
        &self,
        path: &minion_agent::execution::FsPath,
        s: Option<&dyn AbortSignal>,
    ) -> Result<minion_agent::execution::FsPath, FsError> {
        let (p, e) = self.begin("absolute_path", path, s).await;
        let r = if let Some(e) = e {
            Err(Self::error(e))
        } else {
            self.local.absolute_path(path, None).await
        };
        self.finish("absolute_path", p, r)
    }
    async fn canonical_path(
        &self,
        path: &minion_agent::execution::FsPath,
        s: Option<&dyn AbortSignal>,
    ) -> Result<minion_agent::execution::FsPath, FsError> {
        let (p, e) = self.begin("canonical_path", path, s).await;
        let r = if let Some(e) = e {
            Err(Self::error(e))
        } else {
            self.local.canonical_path(path, None).await
        };
        self.finish("canonical_path", p, r)
    }
    async fn check_read_write(
        &self,
        path: &minion_agent::execution::FsPath,
        s: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let (p, e) = self.begin("check_read_write", path, s).await;
        let r = if let Some(e) = e {
            Err(Self::error(e))
        } else {
            self.local.check_read_write(path, None).await
        };
        self.finish("check_read_write", p, r)
    }
    async fn check_readable(
        &self,
        path: &minion_agent::execution::FsPath,
        s: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let (p, e) = self.begin("check_readable", path, s).await;
        let r = if let Some(e) = e {
            Err(Self::error(e))
        } else {
            self.local.check_readable(path, None).await
        };
        self.finish("check_readable", p, r)
    }
    async fn read_binary_file(
        &self,
        path: &minion_agent::execution::FsPath,
        s: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError> {
        let (p, e) = self.begin("read_binary_file", path, s).await;
        let r = if let Some(e) = e {
            Err(Self::error(e))
        } else {
            self.local.read_binary_file(path, None).await
        };
        self.finish("read_binary_file", p, r)
    }
    async fn write_file(
        &self,
        path: &minion_agent::execution::FsPath,
        data: &[u8],
        s: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let (p, e) = self.begin("write_file", path, s).await;
        let r = if let Some(e) = e {
            Err(Self::error(e))
        } else {
            self.local.write_file(path, data, None).await
        };
        self.finish("write_file", p, r)
    }
    async fn create_dir(
        &self,
        path: &minion_agent::execution::FsPath,
        recursive: bool,
        s: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let (p, e) = self.begin("create_dir", path, s).await;
        let r = if let Some(e) = e {
            Err(Self::error(e))
        } else {
            self.local.create_dir(path, recursive, None).await
        };
        self.finish("create_dir", p, r)
    }
    async fn join_path(
        &self,
        _: &[&minion_agent::execution::FsPath],
        _: Option<&dyn AbortSignal>,
    ) -> Result<minion_agent::execution::FsPath, FsError> {
        panic!("unexpected join_path")
    }
    async fn read_text_file(
        &self,
        _: &minion_agent::execution::FsPath,
        _: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        panic!("unexpected read_text_file")
    }
    async fn read_text_lines(
        &self,
        _: &minion_agent::execution::FsPath,
        _: Option<isize>,
        _: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        panic!("unexpected read_text_lines")
    }
    async fn append_file(
        &self,
        _: &minion_agent::execution::FsPath,
        _: &[u8],
        _: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        panic!("unexpected append_file")
    }
    async fn rename_file(
        &self,
        _: &minion_agent::execution::FsPath,
        _: &minion_agent::execution::FsPath,
        _: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        panic!("unexpected rename_file")
    }
    async fn file_info(
        &self,
        _: &minion_agent::execution::FsPath,
        _: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError> {
        panic!("unexpected file_info")
    }
    async fn list_dir(
        &self,
        _: &minion_agent::execution::FsPath,
        _: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError> {
        panic!("unexpected list_dir")
    }
    async fn exists(
        &self,
        _: &minion_agent::execution::FsPath,
        _: Option<&dyn AbortSignal>,
    ) -> Result<bool, FsError> {
        panic!("unexpected exists")
    }
    async fn remove(
        &self,
        _: &minion_agent::execution::FsPath,
        _: bool,
        _: bool,
        _: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        panic!("unexpected remove")
    }
    async fn create_temp_dir(
        &self,
        _: &str,
        _: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        panic!("unexpected create_temp_dir")
    }
    async fn create_temp_file(
        &self,
        _: &str,
        _: &str,
        _: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        panic!("unexpected create_temp_file")
    }
    async fn resolve(
        &self,
        _: &minion_agent::execution::FsPath,
        _: Option<&dyn AbortSignal>,
    ) -> Result<FsTarget, FsError> {
        panic!("unexpected resolve")
    }
    async fn process_path(&self, _: &FsTarget) -> Result<minion_agent::execution::FsPath, FsError> {
        panic!("unexpected process_path")
    }
    async fn cleanup(&self) {}
}
