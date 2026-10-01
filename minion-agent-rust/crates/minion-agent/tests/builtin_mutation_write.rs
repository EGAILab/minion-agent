use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use minion_agent::{
    execution::LocalFileSystem,
    tools::{PreparedValue, ToolExecutionRequest, ToolExecutionSignal, builtin::create_write_tool},
};
use serde_json::json;

struct Signal(AtomicBool);
impl ToolExecutionSignal for Signal {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[tokio::test]
async fn write_uses_the_filesystem_and_reports_utf16_units_not_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let tool = create_write_tool(Arc::new(LocalFileSystem::new(directory.path())));
    let result = (tool.execute())(ToolExecutionRequest {
        tool_call_id: "write".into(),
        params: PreparedValue::from(json!({"path":"@nested/file.txt", "content":"a😀é"})),
        signal: None,
        on_update: None,
    })
    .await
    .unwrap();
    assert_eq!(
        tokio::fs::read(directory.path().join("nested/file.txt"))
            .await
            .unwrap(),
        "a😀é".as_bytes()
    );
    let text = serde_json::to_value(&result.content).unwrap();
    assert_eq!(
        text[0]["text"],
        "Successfully wrote 4 bytes to @nested/file.txt"
    );
    assert_eq!(result.details, json!({}));
}

#[tokio::test]
async fn direct_aborted_execution_registers_but_does_not_create_parents() {
    let directory = tempfile::tempdir().unwrap();
    let tool = create_write_tool(Arc::new(LocalFileSystem::new(directory.path())));
    let error = (tool.execute())(ToolExecutionRequest {
        tool_call_id: "write".into(),
        params: PreparedValue::from(json!({"path":"new/file.txt", "content":"x"})),
        signal: Some(Arc::new(Signal(AtomicBool::new(true)))),
        on_update: None,
    })
    .await
    .unwrap_err();
    assert_eq!(error.message(), "Operation aborted");
    assert!(!directory.path().join("new").exists());
}

#[tokio::test]
async fn parent_creation_error_keeps_the_argument_and_deterministic_cause() {
    let directory = tempfile::tempdir().unwrap();
    tokio::fs::write(directory.path().join("parent"), b"not a directory")
        .await
        .unwrap();
    let tool = create_write_tool(Arc::new(LocalFileSystem::new(directory.path())));
    let error = (tool.execute())(ToolExecutionRequest {
        tool_call_id: "write".into(),
        params: PreparedValue::from(json!({"path":"parent/file", "content":"x"})),
        signal: None,
        on_update: None,
    })
    .await
    .unwrap_err();
    // LocalFileSystem maps create_dir_all's AlreadyExists error to the certified
    // closed Unknown vocabulary; the builtin must not reinterpret the raw OS error.
    assert_eq!(
        error.message(),
        "Cannot create parent directory of parent/file: unknown filesystem error"
    );
}
