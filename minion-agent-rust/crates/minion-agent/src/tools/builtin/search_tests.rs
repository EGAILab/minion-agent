//! Binding-level controlled providers supplement the real-engine canonical adapter.
use super::{create_find_tool, create_grep_tool, search_engines::*};
use crate::{execution::*, tools::*};
use async_trait::async_trait;
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use tokio::sync::Mutex;
struct Pipe(Mutex<Option<Vec<u8>>>);
#[async_trait]
impl ReadableStream for Pipe {
    async fn read_chunk(&self) -> Result<Option<Vec<u8>>, SubprocessError> {
        Ok(self.0.lock().await.take())
    }
    async fn close(&self) {}
}
struct Child {
    out: Arc<Pipe>,
    err: Arc<Pipe>,
    changed: Arc<AtomicBool>,
    change: bool,
    code: i32,
}
#[async_trait]
impl Process for Child {
    fn pid(&self) -> u32 {
        1
    }
    fn stdin(&self) -> Option<Arc<dyn WritableStream>> {
        None
    }
    fn stdout(&self) -> Option<Arc<dyn ReadableStream>> {
        Some(self.out.clone())
    }
    fn stderr(&self) -> Option<Arc<dyn ReadableStream>> {
        Some(self.err.clone())
    }
    async fn wait(&self) -> Result<ExitStatus, SubprocessError> {
        if self.change {
            self.changed.store(true, Ordering::SeqCst);
        }
        Ok(ExitStatus {
            exit_code: Some(self.code),
        })
    }
    async fn terminate(&self) {}
}
struct Engines {
    changed: Arc<AtomicBool>,
    calls: AtomicUsize,
}
#[async_trait]
impl SearchEngines for Engines {
    async fn resolve(
        &self,
        _: SearchEngine,
        _: &ExecutionWorldIdentity,
    ) -> Result<PathBuf, ToolCapabilityError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.changed.load(Ordering::SeqCst) {
            Err(ToolCapabilityError::new("binary changed"))
        } else {
            Ok("scripted-engine".into())
        }
    }
}
struct Spawn {
    local: LocalSubprocess,
    argv: Mutex<Vec<Vec<String>>>,
    out: Vec<u8>,
    err: Vec<u8>,
    code: i32,
    change: bool,
    changed: Arc<AtomicBool>,
}
#[async_trait]
impl Subprocess for Spawn {
    fn cwd(&self) -> &Path {
        self.local.cwd()
    }
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        self.local.execution_world()
    }
    fn platform(&self) -> Platform {
        Platform::Windows
    }
    fn base_env(&self) -> EnvSnapshot {
        self.local.base_env()
    }
    async fn spawn(
        &self,
        argv: &[String],
        options: SpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError> {
        assert!(options.inherit_env);
        assert!(options.signal.is_none());
        assert_eq!(options.stdin, StdioMode::Null);
        assert_eq!(options.stdout, StdioMode::Piped);
        assert_eq!(options.stderr, StdioMode::Piped);
        if argv.contains(&"--json".into()) {
            assert!(!argv.contains(&"--no-require-git".into()));
        }
        self.argv.lock().await.push(argv.to_vec());
        Ok(Arc::new(Child {
            out: Arc::new(Pipe(Mutex::new(Some(self.out.clone())))),
            err: Arc::new(Pipe(Mutex::new(Some(self.err.clone())))),
            changed: self.changed.clone(),
            change: self.change,
            code: self.code,
        }))
    }
}
fn request(params: serde_json::Value) -> ToolExecutionRequest {
    ToolExecutionRequest {
        tool_call_id: "case".into(),
        params: params.into(),
        signal: None,
        on_update: None,
        context: None,
    }
}
struct PartitionSpawn {
    local: LocalSubprocess,
    process: Arc<dyn Process>,
    signal: Arc<super::search_run::tests::Signal>,
    at: &'static str,
}
#[async_trait]
impl Subprocess for PartitionSpawn {
    fn cwd(&self) -> &Path {
        self.local.cwd()
    }
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        self.local.execution_world()
    }
    fn platform(&self) -> Platform {
        self.local.platform()
    }
    fn base_env(&self) -> EnvSnapshot {
        self.local.base_env()
    }
    async fn spawn(
        &self,
        _: &[String],
        _: SpawnOptions,
    ) -> Result<Arc<dyn Process>, SubprocessError> {
        if self.at == "spawn" {
            self.signal.abort();
        }
        Ok(self.process.clone())
    }
}
#[tokio::test]
async fn real_factories_reproduce_the_fourteen_pi_abort_partition_cells() {
    for find in [true, false] {
        for point in [
            "spawn",
            "stdout_data",
            "stdout_eof",
            "stderr_eof",
            "wait",
            "stdout_close",
            "stderr_close",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let output = if find {
                b"a.ts\n".to_vec()
            } else {
                format!("{}\n",serde_json::json!({"type":"match","data":{"path":{"text":dir.path().join("a.ts").to_string_lossy()},"line_number":1,"lines":{"text":"x\n"}}})).into_bytes()
            };
            let (process, signal) = super::search_run::tests::partition_fixture(point, output);
            let spawn = Arc::new(PartitionSpawn {
                local: LocalSubprocess::new(dir.path()),
                process,
                signal: signal.clone(),
                at: point,
            });
            let fs = Arc::new(LocalFileSystem::new(dir.path()));
            let engines = Arc::new(Engines {
                changed: Arc::new(AtomicBool::new(false)),
                calls: AtomicUsize::new(0),
            });
            let tool = if find {
                create_find_tool(fs, spawn, engines)
            } else {
                create_grep_tool(fs, spawn, engines)
            }
            .unwrap();
            let mut req = request(serde_json::json!({"pattern":"x"}));
            req.signal = Some(signal);
            let result = (tool.execute())(req).await;
            let abort = !point.ends_with("close") && (find || point != "spawn");
            assert_eq!(
                result.is_err(),
                abort,
                "{} {point}: {result:?}",
                if find { "find" } else { "grep" }
            );
            if abort {
                assert_eq!(
                    String::from_utf16_lossy(result.unwrap_err().message().code_units()),
                    "Operation aborted"
                );
            } else {
                assert_eq!(
                    text(result.unwrap()),
                    if find { "a.ts" } else { "a.ts:1: x" }
                );
            }
        }
    }
}
fn text(result: AgentToolResult) -> String {
    match &result.content[0] {
        crate::llm::ToolResultContentBlock::Text(t) => {
            String::from_utf16_lossy(t.text.code_units())
        }
        _ => panic!(),
    }
}
fn setup(
    root: &Path,
    out: Vec<u8>,
    err: Vec<u8>,
    code: i32,
    change: bool,
) -> (Arc<Spawn>, Arc<Engines>) {
    let changed = Arc::new(AtomicBool::new(false));
    (
        Arc::new(Spawn {
            local: LocalSubprocess::new(root),
            argv: Mutex::new(Vec::new()),
            out,
            err,
            code,
            change,
            changed: changed.clone(),
        }),
        Arc::new(Engines {
            changed,
            calls: AtomicUsize::new(0),
        }),
    )
}
#[tokio::test]
async fn diagnostic_rerun_is_verified_again_and_preserves_pi_text() {
    for change in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (spawn, engines) = setup(
            dir.path(),
            Vec::new(),
            b"error parsing glob: Pi diagnostic".to_vec(),
            2,
            change,
        );
        let tool = create_find_tool(
            Arc::new(LocalFileSystem::new(dir.path())),
            spawn.clone(),
            engines.clone(),
        )
        .unwrap();
        let result = (tool.execute())(request(serde_json::json!({"pattern":"src/**/["})))
            .await
            .unwrap_err();
        assert_eq!(
            String::from_utf16_lossy(result.message().code_units()),
            if change {
                "binary changed"
            } else {
                "error parsing glob: Pi diagnostic"
            }
        );
        assert_eq!(engines.calls.load(Ordering::SeqCst), 2);
        let calls = spawn.argv.lock().await;
        assert_eq!(calls.len(), if change { 1 } else { 2 });
        if !change {
            assert_eq!(
                calls[1][calls[1].len() - 2],
                super::search_glob::pi_windows("**/src/**/[")
            );
        }
    }
}
#[tokio::test]
async fn find_stream_order_duplicates_and_untrimmed_empty_decision() {
    let dir = tempfile::tempdir().unwrap();
    let (spawn, engines) = setup(
        dir.path(),
        b"z.ts\na.ts \na.ts\n".to_vec(),
        Vec::new(),
        2,
        false,
    );
    let tool = create_find_tool(
        Arc::new(LocalFileSystem::new(dir.path())),
        spawn.clone(),
        engines,
    )
    .unwrap();
    assert_eq!(
        text(
            (tool.execute())(request(serde_json::json!({"pattern":"*.ts","limit":3})))
                .await
                .unwrap()
        ),
        "z.ts\na.ts\na.ts\n\n[3 results limit reached. Use limit=6 for more, or refine pattern]"
    );
    assert!(spawn.argv.lock().await[0].contains(&"--no-require-git".into()));
    let (spawn, engines) = setup(dir.path(), b" \n".to_vec(), Vec::new(), 2, false);
    let tool =
        create_find_tool(Arc::new(LocalFileSystem::new(dir.path())), spawn, engines).unwrap();
    let result = (tool.execute())(request(serde_json::json!({"pattern":"x"}))).await;
    assert!(
        result.is_ok(),
        "whitespace-only nonempty stdout must be success, not status error: {result:?}"
    );
    assert_eq!(text(result.unwrap()), "");
}
#[tokio::test]
async fn grep_counts_uncollected_matches_without_reordering() {
    let dir = tempfile::tempdir().unwrap();
    let lines=[serde_json::json!({"type":"match","data":{"path":{"bytes":"YQ=="},"line_number":1,"lines":{"text":"x\n"}}}),serde_json::json!({"type":"match","data":{"path":{"text":"b.ts"},"line_number":2,"lines":{"text":"second\n"}}}),serde_json::json!({"type":"match","data":{"path":{"text":"a.ts"},"line_number":1,"lines":{"text":"first\n"}}})].iter().map(|x|format!("{x}\n")).collect::<String>();
    let (spawn, engines) = setup(dir.path(), lines.into_bytes(), Vec::new(), 0, false);
    let tool =
        create_grep_tool(Arc::new(LocalFileSystem::new(dir.path())), spawn, engines).unwrap();
    assert_eq!(
        text(
            (tool.execute())(request(serde_json::json!({"pattern":"x","limit":2})))
                .await
                .unwrap()
        ),
        "b.ts:2: second\n\n[2 matches limit reached. Use limit=4 for more, or refine pattern]"
    );
    assert_eq!(
        text(
            (tool.execute())(request(serde_json::json!({"pattern":"x","limit":4})))
                .await
                .unwrap()
        ),
        "b.ts:2: second\na.ts:1: first"
    );
}
#[tokio::test]
async fn managed_store_verifies_every_use_and_provisioning_is_idempotent() {
    let source = PathBuf::from(
        std::env::var_os("MINION_SEARCH_ENGINE_ARTIFACTS").expect("official artifacts required"),
    );
    let dir = tempfile::tempdir().unwrap();
    let store = SearchEngineStore::new(dir.path());
    assert!(
        store
            .resolve(SearchEngine::Fd, &ExecutionWorldIdentity::local())
            .await
            .is_err()
    );
    provision_search_engines(&store, Some(&source))
        .await
        .unwrap();
    for engine in [SearchEngine::Fd, SearchEngine::Ripgrep] {
        let path = store
            .resolve(engine, &ExecutionWorldIdentity::local())
            .await
            .unwrap();
        let before = std::fs::metadata(&path).unwrap().modified().unwrap();
        provision_search_engines(&store, Some(&source))
            .await
            .unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            before
        );
        std::fs::write(&path, b"replaced").unwrap();
        assert!(
            store
                .resolve(engine, &ExecutionWorldIdentity::local())
                .await
                .is_err()
        );
        provision_search_engines(&store, Some(&source))
            .await
            .unwrap();
        store
            .resolve(engine, &ExecutionWorldIdentity::local())
            .await
            .unwrap();
    }
    let bad = tempfile::tempdir().unwrap();
    let empty = tempfile::tempdir().unwrap();
    std::fs::write(
        bad.path().join(SearchEngine::Fd.pin().unwrap().artifact),
        b"wrong",
    )
    .unwrap();
    let error = provision_search_engines(&SearchEngineStore::new(empty.path()), Some(bad.path()))
        .await
        .unwrap_err();
    assert_eq!(
        String::from_utf16_lossy(error.message().code_units()),
        "search engine artifact failed verification"
    );
    assert_eq!(std::fs::read_dir(empty.path()).unwrap().count(), 0);
}
