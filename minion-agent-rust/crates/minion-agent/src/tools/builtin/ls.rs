//! `ls`: raw enumerate, pinned stable sort, lazy per-entry probe and cap.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::{
    execution::{DirEntryProbeKind, FileSystem, FsErrorCode},
    llm::{TextBlock, ToolResultContentBlock},
    tools::{
        AgentToolResult, PreparedValue, ToolCapabilityError, ToolDefinition, ToolExecutionRequest,
        ToolExecutionSignal,
    },
};

use super::{
    collation::sort_names,
    numeric::number_to_string,
    paths::{OPERATION_ABORTED, cause, preprocess_path},
    truncate::{DEFAULT_MAX_BYTES, format_size, truncate_head},
};

fn directory(kind: DirEntryProbeKind) -> bool {
    matches!(
        kind,
        DirEntryProbeKind::Directory | DirEntryProbeKind::SymlinkToDirectory
    )
}

fn aborted(signal: Option<&Arc<dyn ToolExecutionSignal>>) -> bool {
    signal.is_some_and(|signal| signal.is_cancelled())
}

async fn list(
    fs: Arc<dyn FileSystem>,
    path: String,
    limit: f64,
    limit_value: PreparedValue,
) -> Result<AgentToolResult, ToolCapabilityError> {
    let working = preprocess_path(&path)?;
    let absolute = fs
        .absolute_path(&working, None)
        .await
        .unwrap_or_else(|_| working.clone());
    match fs.probe_dir_entry(&working, None).await {
        Err(error) if error.code == FsErrorCode::NotSupported => {
            return Err(ToolCapabilityError::new(format!(
                "Cannot access {absolute}: {}",
                cause(error.code)
            )));
        }
        Err(_) => {
            return Err(ToolCapabilityError::new(format!(
                "Path not found: {absolute}"
            )));
        }
        Ok(entry) if !directory(entry.kind) => {
            return Err(ToolCapabilityError::new(format!(
                "Not a directory: {absolute}"
            )));
        }
        Ok(_) => {}
    }
    let names = fs.list_dir_raw(&working, None).await.map_err(|error| {
        if error.code == FsErrorCode::Aborted {
            ToolCapabilityError::new(OPERATION_ABORTED)
        } else {
            ToolCapabilityError::new(format!("Cannot read directory: {}", cause(error.code)))
        }
    })?;
    let names = sort_names(names)?;
    let mut results = Vec::new();
    let mut entry_limit_reached = false;
    for name in names {
        if results.len() as f64 >= limit {
            entry_limit_reached = true;
            break;
        }
        let Ok(joined) = fs.join_path(&[&absolute, &name], None).await else {
            continue;
        };
        let Ok(entry) = fs.probe_dir_entry(&joined, None).await else {
            continue;
        };
        if directory(entry.kind) {
            results.push(format!("{name}/"));
        } else {
            results.push(name);
        }
    }
    let (text, details) = if results.is_empty() {
        ("(empty directory)".to_owned(), json!({}))
    } else {
        let truncated = truncate_head(
            &results.join("\n"),
            9_007_199_254_740_991,
            DEFAULT_MAX_BYTES,
        );
        let mut text = truncated.content.clone();
        let mut details = serde_json::Map::new();
        let mut notices = Vec::new();
        if entry_limit_reached {
            notices.push(format!(
                "{} entries limit reached. Use limit={} for more",
                number_to_string(limit),
                number_to_string(limit * 2.0)
            ));
            details.insert(
                "entry_limit_reached".into(),
                limit_value
                    .try_to_json()
                    .map_err(|error| ToolCapabilityError::new(error.to_string()))?,
            );
        }
        if truncated.truncated {
            notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
            details.insert("truncation".into(), truncated.details());
        }
        if !notices.is_empty() {
            text.push_str(&format!("\n\n[{}]", notices.join(". ")));
        }
        (text, Value::Object(details))
    };
    Ok(AgentToolResult {
        content: vec![ToolResultContentBlock::Text(TextBlock::new(text))],
        details,
        usage: None,
        added_tool_names: None,
        terminate: None,
    })
}

pub fn create_ls_tool(fs: Arc<dyn FileSystem>) -> ToolDefinition {
    let schema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: current directory)"},
            "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"}
        }
    })).expect("static ls schema is valid");
    ToolDefinition::new(
        "ls",
        "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first).",
        schema,
        "ls",
        move |request: ToolExecutionRequest| {
            let fs = fs.clone();
            Box::pin(async move {
                if aborted(request.signal.as_ref()) {
                    return Err(ToolCapabilityError::new(OPERATION_ABORTED));
                }
                let path = request
                    .params
                    .get("path")
                    .and_then(PreparedValue::as_str)
                    .filter(|path| !path.is_empty())
                    .unwrap_or(".")
                    .to_owned();
                let limit_value = request
                    .params
                    .get("limit")
                    .filter(|value| !value.is_null())
                    .cloned()
                    .unwrap_or_else(|| json!(500).into());
                let limit = limit_value.as_f64().unwrap_or(500.0);
                let signal = request.signal;
                let worker_signal = signal.clone();
                let worker = tokio::spawn(async move {
                    let outcome = list(fs, path, limit, limit_value).await;
                    if aborted(worker_signal.as_ref()) {
                        Err(ToolCapabilityError::new(OPERATION_ABORTED))
                    } else {
                        outcome
                    }
                });
                super::race_abort(worker, signal).await
            })
        },
    )
}
