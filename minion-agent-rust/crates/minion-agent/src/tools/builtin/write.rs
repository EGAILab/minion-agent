//! TOOL-029: filesystem writes with Pi's queue and cooperative checkpoints.

use super::{
    mutation_queue::Registration,
    paths::{OPERATION_ABORTED, argument_path, cause, path_message, preprocess_path},
};
use crate::{
    execution::{FileSystem, FsPath},
    llm::{ResultTextBlock, ResultValue, ToolResultContentBlock},
    tools::{
        AgentToolResult, PreparedString, PreparedValue, ToolCapabilityError, ToolDefinition,
        ToolExecutionRequest, ToolExecutionSignal,
    },
};
use serde_json::json;
use std::sync::Arc;

pub(super) fn check_abort(
    signal: Option<&Arc<dyn ToolExecutionSignal>>,
) -> Result<(), ToolCapabilityError> {
    if signal.is_some_and(|s| s.is_cancelled()) {
        Err(ToolCapabilityError::new(OPERATION_ABORTED))
    } else {
        Ok(())
    }
}

pub(super) fn text_result(
    text: impl Into<crate::llm::ResultString>,
    details: impl Into<ResultValue>,
) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text(ResultTextBlock::new(text))],
        details: details.into(),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

async fn write(
    fs: Arc<dyn FileSystem>,
    registration: Registration,
    working: FsPath,
    path: FsPath,
    content: PreparedString,
    signal: Option<Arc<dyn ToolExecutionSignal>>,
) -> Result<AgentToolResult, ToolCapabilityError> {
    let _entry = registration.acquire().await.map_err(|e| {
        ToolCapabilityError::new(path_message(
            &format!("Cannot resolve {{path}}: {}", cause(e.code)),
            &path,
        ))
    })?;
    check_abort(signal.as_ref())?;
    let absolute = fs.absolute_path(&working, None).await.map_err(|e| {
        ToolCapabilityError::new(path_message(
            &format!("Cannot resolve {{path}}: {}", cause(e.code)),
            &path,
        ))
    })?;
    let parent = crate::execution::path::parent(&absolute);
    fs.create_dir(&parent, true, None).await.map_err(|e| {
        ToolCapabilityError::new(path_message(
            &format!(
                "Cannot create parent directory of {{path}}: {}",
                cause(e.code)
            ),
            &path,
        ))
    })?;
    check_abort(signal.as_ref())?;
    fs.write_file(&working, content.to_utf8_lossy().as_bytes(), None)
        .await
        .map_err(|e| {
            ToolCapabilityError::new(path_message(
                &format!("Cannot write {{path}}: {}", cause(e.code)),
                &path,
            ))
        })?;
    check_abort(signal.as_ref())?;
    Ok(text_result(
        path_message(
            &format!(
                "Successfully wrote {} bytes to {{path}}",
                content.code_units().len()
            ),
            &path,
        ),
        json!({}),
    ))
}

pub fn create_write_tool(fs: Arc<dyn FileSystem>) -> ToolDefinition {
    ToolDefinition::new("write", "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.", serde_json::from_value(json!({"type":"object","properties":{"path":{"type":"string","description":"Path to the file to write (relative or absolute)"},"content":{"type":"string","description":"Content to write to the file"}},"required":["path","content"]})).expect("static write schema"), "write", move |request: ToolExecutionRequest| {
        let fs = fs.clone();
        Box::pin(async move {
            let path = request.params.get("path").and_then(argument_path).ok_or_else(|| ToolCapabilityError::new("path is required"))?;
            let content = match request.params.get("content") {
                Some(PreparedValue::String(content)) => content.clone(),
                _ => return Err(ToolCapabilityError::new("content is required")),
            };
            let working = preprocess_path(&path)?;
            let registration = Registration::new(fs.clone(), working.clone());
            // Do not race an abort against the worker. It retains the queue lock through
            // every in-flight filesystem await, even when the caller drops its future.
            tokio::spawn(write(fs, registration, working, path, content, request.signal)).await.map_err(|e| ToolCapabilityError::new(e.to_string()))?
        })
    })
}
