//! TOOL-029: filesystem writes with Pi's queue and cooperative checkpoints.

use super::{
    mutation_queue::Registration,
    paths::{OPERATION_ABORTED, cause, preprocess_path},
};
use crate::{
    execution::FileSystem,
    llm::{TextBlock, ToolResultContentBlock},
    tools::{
        AgentToolResult, PreparedValue, ToolCapabilityError, ToolDefinition, ToolExecutionRequest,
        ToolExecutionSignal,
    },
};
use serde_json::json;
use std::{path::Path, sync::Arc};

pub(super) fn check_abort(
    signal: Option<&Arc<dyn ToolExecutionSignal>>,
) -> Result<(), ToolCapabilityError> {
    if signal.is_some_and(|s| s.is_cancelled()) {
        Err(ToolCapabilityError::new(OPERATION_ABORTED))
    } else {
        Ok(())
    }
}

pub(super) fn text_result(text: String, details: serde_json::Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new(text))],
        details,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

async fn write(
    fs: Arc<dyn FileSystem>,
    registration: Registration,
    working: String,
    path: String,
    content: String,
    signal: Option<Arc<dyn ToolExecutionSignal>>,
) -> Result<AgentToolResult, ToolCapabilityError> {
    let _entry = registration.acquire().await.map_err(|e| {
        ToolCapabilityError::new(format!("Cannot resolve {path}: {}", cause(e.code)))
    })?;
    check_abort(signal.as_ref())?;
    let absolute = fs.absolute_path(&working, None).await.map_err(|e| {
        ToolCapabilityError::new(format!("Cannot resolve {path}: {}", cause(e.code)))
    })?;
    let parent = Path::new(&absolute)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_string_lossy();
    fs.create_dir(&parent, true, None).await.map_err(|e| {
        ToolCapabilityError::new(format!(
            "Cannot create parent directory of {path}: {}",
            cause(e.code)
        ))
    })?;
    check_abort(signal.as_ref())?;
    fs.write_file(&working, content.as_bytes(), None)
        .await
        .map_err(|e| ToolCapabilityError::new(format!("Cannot write {path}: {}", cause(e.code))))?;
    check_abort(signal.as_ref())?;
    Ok(text_result(
        format!(
            "Successfully wrote {} bytes to {path}",
            content.encode_utf16().count()
        ),
        json!({}),
    ))
}

pub fn create_write_tool(fs: Arc<dyn FileSystem>) -> ToolDefinition {
    ToolDefinition::new("write", "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.", serde_json::from_value(json!({"type":"object","properties":{"path":{"type":"string","description":"Path to the file to write (relative or absolute)"},"content":{"type":"string","description":"Content to write to the file"}},"required":["path","content"]})).expect("static write schema"), "write", move |request: ToolExecutionRequest| {
        let fs = fs.clone();
        Box::pin(async move {
            let path = request.params.get("path").and_then(PreparedValue::as_str).ok_or_else(|| ToolCapabilityError::new("path is required"))?.to_owned();
            let content = request.params.get("content").and_then(PreparedValue::as_str).ok_or_else(|| ToolCapabilityError::new("content is required"))?.to_owned();
            let working = preprocess_path(&path)?;
            let registration = Registration::new(fs.clone(), working.clone());
            // Do not race an abort against the worker. It retains the queue lock through
            // every in-flight filesystem await, even when the caller drops its future.
            tokio::spawn(write(fs, registration, working, path, content, request.signal)).await.map_err(|e| ToolCapabilityError::new(e.to_string()))?
        })
    })
}
