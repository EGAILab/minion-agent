#![cfg(feature = "conformance")]

//! Thin canonical adapter: fixture filesystem + scripted provider responses, real tools and
//! real Layer-06 execution. All ordering, truncation, collation and image semantics stay in lib.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use minion_agent::{
    Runtime,
    execution::{
        AbortSignal, DirEntryProbe, DirEntryProbeKind, ExecutionWorldIdentity, FileInfo,
        FileSystem, FsError, FsErrorCode, FsTarget, LocalFileSystem,
    },
    llm::{ImageSource, StopReason, TextBlock, ToolCall, ToolResultContentBlock},
    tools::{
        ToolExecutionOptions, ToolExecutionSignal,
        builtin::{ReadToolOptions, create_ls_tool, create_read_tool},
        execute_tool_calls,
    },
};
use parking_lot::Mutex;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn root() -> PathBuf {
    std::env::var_os("MINION_AGENT_CONFORMANCE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

#[derive(Clone, Default)]
struct ToggleSignal(Arc<AtomicBool>);
impl ToggleSignal {
    fn abort(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
impl ToolExecutionSignal for ToggleSignal {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

struct FixtureFs {
    root: PathBuf,
    local: LocalFileSystem,
    provider: Value,
    calls: Arc<Mutex<Vec<String>>>,
    abort_after: Option<String>,
    controller: ToggleSignal,
}

impl FixtureFs {
    fn new(
        root: &Path,
        provider: Value,
        abort_after: Option<String>,
        controller: ToggleSignal,
    ) -> Self {
        Self {
            root: root.to_owned(),
            local: LocalFileSystem::new(root),
            provider,
            calls: Arc::new(Mutex::new(Vec::new())),
            abort_after,
            controller,
        }
    }

    fn relative(&self, path: &str) -> String {
        let absolute = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            self.root.join(path)
        };
        let path = absolute.strip_prefix(&self.root).map_or_else(
            |_| path.to_owned(),
            |relative| relative.to_string_lossy().into_owned(),
        );
        if path.is_empty() {
            ".".into()
        } else {
            path.replace('\\', "/")
        }
    }

    fn log(&self, operation: &str, path: &str) {
        self.calls
            .lock()
            .push(format!("{operation} {}", self.relative(path)));
    }

    fn scripted(&self, operation: &str, path: &str) -> Option<&Value> {
        let relative = self.relative(path);
        self.provider
            .get(operation)?
            .as_array()?
            .iter()
            .find(|entry| entry.get("path").and_then(Value::as_str) == Some(relative.as_str()))
    }

    fn code(value: &Value) -> FsErrorCode {
        serde_json::from_value(value.clone()).expect("schema validates filesystem error code")
    }

    fn scripted_error(&self, operation: &str, path: &str) -> Option<FsError> {
        self.scripted(operation, path)
            .and_then(|entry| entry.get("error"))
            .map(|code| FsError::new(Self::code(code), "scripted"))
    }

    fn not_supported(operation: &str) -> FsError {
        FsError::new(
            FsErrorCode::NotSupported,
            format!("{operation} not supported"),
        )
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
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        self.log("absolute_path", path);
        self.local.absolute_path(path, signal).await
    }
    async fn join_path(
        &self,
        parts: &[&str],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        self.calls.lock().push("join_path *".into());
        self.local.join_path(parts, signal).await
    }
    async fn read_binary_file(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError> {
        self.log("read_binary_file", path);
        if let Some(error) = self.scripted_error("read_binary_file", path) {
            return Err(error);
        }
        let answer = self.local.read_binary_file(path, signal).await;
        if self.abort_after.as_deref() == Some("read_binary_file") {
            self.controller.abort();
        }
        answer
    }
    async fn check_readable(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        self.log("check_readable", path);
        let answer = if self.provider.get("without_exec_008") == Some(&Value::Bool(true)) {
            Err(Self::not_supported("check_readable"))
        } else if let Some(error) = self.scripted_error("check_readable", path) {
            Err(error)
        } else {
            self.local.check_readable(path, signal).await
        };
        if self.abort_after.as_deref() == Some("check_readable") {
            self.controller.abort();
        }
        answer
    }
    async fn list_dir_raw(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        self.log("list_dir_raw", path);
        if self.provider.get("without_exec_007") == Some(&Value::Bool(true)) {
            return Err(Self::not_supported("list_dir_raw"));
        }
        if let Some(entry) = self.scripted("list_dir_raw", path) {
            if let Some(code) = entry.get("error") {
                return Err(FsError::new(Self::code(code), "scripted"));
            }
            return Ok(entry["names"]
                .as_array()
                .expect("names array")
                .iter()
                .map(|name| name.as_str().expect("name string").to_owned())
                .collect());
        }
        self.local.list_dir_raw(path, signal).await
    }
    async fn probe_dir_entry(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<DirEntryProbe, FsError> {
        self.log("probe_dir_entry", path);
        if self.provider.get("without_exec_007") == Some(&Value::Bool(true)) {
            return Err(Self::not_supported("probe_dir_entry"));
        }
        if let Some(entry) = self.scripted("probe_dir_entry", path) {
            if let Some(code) = entry.get("error") {
                return Err(FsError::new(Self::code(code), "scripted"));
            }
            let kind: DirEntryProbeKind =
                serde_json::from_value(entry["kind"].clone()).expect("schema kind");
            return Ok(DirEntryProbe {
                name: Path::new(path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                path: path.to_owned(),
                kind,
            });
        }
        self.local.probe_dir_entry(path, signal).await
    }
    async fn canonical_path(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        self.log("canonical_path", path);
        if let Some(error) = self.scripted_error("canonical_path", path) {
            return Err(error);
        }
        self.local.canonical_path(path, signal).await
    }
    async fn file_info(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError> {
        self.log("file_info", path);
        if let Some(error) = self.scripted_error("file_info", path) {
            return Err(error);
        }
        self.local.file_info(path, signal).await
    }

    async fn read_text_file(
        &self,
        _: &str,
        _: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        Err(Self::not_supported("unexpected read_text_file"))
    }
    async fn read_text_lines(
        &self,
        _: &str,
        _: Option<isize>,
        _: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        Err(Self::not_supported("unexpected read_text_lines"))
    }
    async fn write_file(
        &self,
        _: &str,
        _: &[u8],
        _: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Err(Self::not_supported("unexpected write_file"))
    }
    async fn append_file(
        &self,
        _: &str,
        _: &[u8],
        _: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Err(Self::not_supported("unexpected append_file"))
    }
    async fn rename_file(
        &self,
        _: &str,
        _: &str,
        _: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Err(Self::not_supported("unexpected rename_file"))
    }
    async fn list_dir(
        &self,
        _: &str,
        _: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError> {
        Err(Self::not_supported("unexpected list_dir"))
    }
    async fn exists(&self, _: &str, _: Option<&dyn AbortSignal>) -> Result<bool, FsError> {
        Err(Self::not_supported("unexpected exists"))
    }
    async fn create_dir(
        &self,
        _: &str,
        _: bool,
        _: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Err(Self::not_supported("unexpected create_dir"))
    }
    async fn remove(
        &self,
        _: &str,
        _: bool,
        _: bool,
        _: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Err(Self::not_supported("unexpected remove"))
    }
    async fn create_temp_dir(
        &self,
        _: &str,
        _: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        Err(Self::not_supported("unexpected create_temp_dir"))
    }
    async fn create_temp_file(
        &self,
        _: &str,
        _: &str,
        _: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        Err(Self::not_supported("unexpected create_temp_file"))
    }
    async fn resolve(&self, _: &str, _: Option<&dyn AbortSignal>) -> Result<FsTarget, FsError> {
        Err(Self::not_supported("unexpected resolve"))
    }
    async fn process_path(&self, _: &FsTarget) -> Result<String, FsError> {
        Err(Self::not_supported("unexpected process_path"))
    }
    async fn cleanup(&self) {}
}

#[tokio::test]
async fn abort_during_read_binary_file_wins_at_worker_settlement() {
    let document = json!({
        "name": "worker settle abort witness",
        "builtin_tool": {
            "tool": "read",
            "fixture": [{"path": "note.txt", "file": {"text": "hello"}}]
        }
    });
    let case = json!({
        "id": "abort_after_read_binary_file",
        "arguments": {"path": "note.txt"},
        "abort_after": "read_binary_file",
        "expect": {"is_error": true, "text": "Operation aborted", "details": {}}
    });
    run_case(&document, &case).await;
}

fn content_bytes(root: &Path, content: &Value) -> Vec<u8> {
    if let Some(text) = content.get("text").and_then(Value::as_str) {
        return text.as_bytes().to_vec();
    }
    if let Some(base64) = content.get("base64").and_then(Value::as_str) {
        return STANDARD.decode(base64).unwrap();
    }
    if let Some(lines) = content.get("lines") {
        return (1..=lines["count"].as_u64().unwrap())
            .map(|n| {
                lines["template"]
                    .as_str()
                    .unwrap()
                    .replace("{n}", &n.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes();
    }
    if let Some(repeat) = content.get("repeat") {
        return repeat["unit"]
            .as_str()
            .unwrap()
            .repeat(repeat["times"].as_u64().unwrap() as usize)
            .into_bytes();
    }
    std::fs::read(
        root.join("conformance/agent/fixtures")
            .join(content["fixture_file"].as_str().unwrap()),
    )
    .unwrap()
}

fn build_fixture(directory: &Path, entries: &[Value]) {
    for entry in entries {
        let target = directory.join(entry["path"].as_str().unwrap());
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        if entry.get("dir").is_some() {
            std::fs::create_dir_all(target).unwrap();
        } else if let Some(link) = entry.get("symlink").and_then(Value::as_str) {
            let target_path = directory.join(link);
            #[cfg(unix)]
            std::os::unix::fs::symlink(target_path, target).unwrap();
            #[cfg(windows)]
            {
                if target_path.is_dir() {
                    std::os::windows::fs::symlink_dir(target_path, target).unwrap();
                } else {
                    std::os::windows::fs::symlink_file(target_path, target).unwrap();
                }
            }
        } else {
            std::fs::write(target, content_bytes(&root(), &entry["file"])).unwrap();
        }
    }
}

async fn expand_abs(mut text: String, local: &LocalFileSystem) -> String {
    while let Some(start) = text.find("{abs:") {
        let tail = &text[start + 5..];
        let end = tail.find('}').expect("absolute token closes");
        let path = &tail[..end];
        let absolute = local.absolute_path(path, None).await.unwrap();
        text.replace_range(start..start + 5 + end + 1, &absolute);
    }
    text
}

async fn run_case(document: &Value, case: &Value) {
    let tmp = tempfile::tempdir().unwrap();
    let directory = tmp.path().canonicalize().unwrap();
    let spec = &document["builtin_tool"];
    build_fixture(
        &directory,
        spec.get("fixture")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]),
    );
    let signal = ToggleSignal::default();
    if case.get("signal").and_then(Value::as_str) == Some("pre_aborted") {
        signal.abort();
    }
    let fs = Arc::new(FixtureFs::new(
        &directory,
        spec.get("provider").cloned().unwrap_or_else(|| json!({})),
        case.get("abort_after")
            .and_then(Value::as_str)
            .map(str::to_owned),
        signal.clone(),
    ));
    let options = spec.get("options").cloned().unwrap_or_else(|| json!({}));
    let supports = options
        .get("model_supports_images")
        .and_then(Value::as_bool);
    let read_options = ReadToolOptions {
        auto_resize_images: options
            .get("auto_resize_images")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        model_supports_images: supports.map(|value| Arc::new(move || Some(value)) as _),
    };
    let tool = if spec["tool"] == "read" {
        create_read_tool(fs.clone(), read_options)
    } else {
        create_ls_tool(fs.clone())
    };
    let runtime = Runtime::new();
    runtime.tools().register_for_scope(None, tool).unwrap();
    let context = runtime.context();
    let arguments: BTreeMap<String, Value> =
        serde_json::from_value(case["arguments"].clone()).unwrap();
    let call = ToolCall::new("call-1", spec["tool"].as_str().unwrap(), arguments);
    let batch = execute_tool_calls(
        &context,
        &[call],
        ToolExecutionOptions::new(StopReason::ToolUse, 0.0).with_signal(Arc::new(signal)),
    )
    .await
    .unwrap();
    assert_eq!(batch.messages.len(), 1);
    let result = &batch.messages[0];
    let label = format!(
        "{} / {}",
        document["name"],
        case.get("id").and_then(Value::as_str).unwrap_or("case")
    );
    let expected = &case["expect"];
    assert_eq!(
        result.is_error,
        expected["is_error"].as_bool().unwrap(),
        "{label}: is_error"
    );
    let ToolResultContentBlock::Text(TextBlock { text, .. }) = &result.content[0] else {
        panic!("{label}: first block must be text");
    };
    if let Some(expected_text) = expected.get("text").and_then(Value::as_str) {
        assert_eq!(
            text,
            &expand_abs(expected_text.to_owned(), &fs.local).await,
            "{label}: text"
        );
    } else {
        assert_eq!(
            format!("{:x}", Sha256::digest(text.as_bytes())),
            expected["text_sha256"].as_str().unwrap(),
            "{label}: text hash"
        );
        assert!(
            text.ends_with(
                &expand_abs(
                    expected["text_tail"].as_str().unwrap().to_owned(),
                    &fs.local
                )
                .await
            ),
            "{label}: text tail"
        );
    }
    match (result.content.get(1), expected.get("image")) {
        (Some(ToolResultContentBlock::Image(image)), Some(Value::Object(expected_image))) => {
            assert_eq!(
                image.mime_type,
                expected_image["mime_type"].as_str().unwrap(),
                "{label}: image MIME"
            );
            let ImageSource::Data { data } = &image.source else {
                panic!("{label}: image must be inline data");
            };
            let bytes = STANDARD.decode(data).unwrap();
            assert_eq!(STANDARD.encode(&bytes), *data, "{label}: canonical base64");
            if let Some(length) = expected_image.get("base64_len") {
                assert_eq!(
                    data.len(),
                    length.as_u64().unwrap() as usize,
                    "{label}: base64 length"
                );
            }
            assert_eq!(
                bytes.len(),
                expected_image["bytes"].as_u64().unwrap() as usize,
                "{label}: image bytes"
            );
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                expected_image["sha256"].as_str().unwrap(),
                "{label}: image hash"
            );
        }
        (None, Some(Value::Null) | None) => {}
        (actual, expected) => panic!("{label}: image mismatch: {actual:?} vs {expected:?}"),
    }
    assert_eq!(
        result.details.clone().unwrap_or_else(|| json!({})),
        expected
            .get("details")
            .cloned()
            .unwrap_or_else(|| json!({})),
        "{label}: details"
    );
    let calls = fs.calls.lock().clone();
    if let Some(expected_calls) = expected.get("fs_calls") {
        assert_eq!(json!(calls), *expected_calls, "{label}: filesystem calls");
    }
    if let Some(expected_probes) = expected.get("probed_entries") {
        let probes: Vec<_> = calls
            .iter()
            .filter_map(|call| call.strip_prefix("probe_dir_entry "))
            .collect();
        assert_eq!(json!(probes), *expected_probes, "{label}: probes");
    }
}

#[tokio::test]
async fn every_builtin_scenario_uses_real_rust_tools_and_execution() {
    let directory = root().join("conformance/agent");
    let mut paths = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yaml")
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut count = 0;
    let mut documents = 0;
    for path in paths {
        let document: Value = serde_yaml::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        if document.get("builtin_tool").is_none() {
            continue;
        }
        documents += 1;
        for case in document["builtin_tool"]["cases"].as_array().unwrap() {
            run_case(&document, case).await;
            count += 1;
        }
    }
    assert_eq!(
        documents, 45,
        "all WP-13.1 canonical documents must be discovered"
    );
    assert!(count >= documents, "every document has an executable case");
    eprintln!("WP-13.1 canonical: {documents} documents, {count} cases passed");
}
