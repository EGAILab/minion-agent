//! `read` composes the certified filesystem capability and typed tool-result boundary.

use std::sync::Arc;

use crate::{
    execution::{FileSystem, FsError, FsErrorCode, FsPath},
    llm::{ResultTextBlock, ToolResultContentBlock},
    tools::{
        AgentToolResult, PreparedValue, ToolCapabilityError, ToolDefinition, ToolExecutionRequest,
        ToolExecutionSignal,
    },
};
use serde_json::{Value, json};

use super::{
    mime::detect_supported_image_mime_type,
    paths::{OPERATION_ABORTED, argument_path, cause, path_message, preprocess_path},
    text::read_text,
};

pub type ModelSupportsImages = Arc<dyn Fn() -> Option<bool> + Send + Sync>;

#[derive(Clone)]
pub struct ReadToolOptions {
    pub auto_resize_images: bool,
    pub model_supports_images: Option<ModelSupportsImages>,
}

impl Default for ReadToolOptions {
    fn default() -> Self {
        Self {
            auto_resize_images: true,
            model_supports_images: None,
        }
    }
}

fn aborted() -> ToolCapabilityError {
    ToolCapabilityError::new(OPERATION_ABORTED)
}

fn is_aborted(signal: Option<&Arc<dyn ToolExecutionSignal>>) -> bool {
    signal.is_some_and(|signal| signal.is_cancelled())
}

fn result(content: Vec<ToolResultContentBlock>, details: Value) -> AgentToolResult {
    AgentToolResult {
        content,
        details: details.into(),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

async fn absolute_or_working(fs: &dyn FileSystem, working: &FsPath) -> FsPath {
    fs.absolute_path(working, None)
        .await
        .unwrap_or_else(|_| working.to_owned())
}

async fn fs_failure(
    fs: &dyn FileSystem,
    working: &FsPath,
    site: &str,
    error: FsError,
) -> ToolCapabilityError {
    if error.code == FsErrorCode::Aborted {
        return aborted();
    }
    let absolute = absolute_or_working(fs, working).await;
    ToolCapabilityError::new(path_message(
        &format!("{site} {{path}}: {}", cause(error.code)),
        &absolute,
    ))
}

async fn read(
    fs: Arc<dyn FileSystem>,
    options: ReadToolOptions,
    path: FsPath,
    offset: Option<f64>,
    limit: Option<f64>,
    signal: Option<Arc<dyn ToolExecutionSignal>>,
) -> Result<AgentToolResult, ToolCapabilityError> {
    let working = preprocess_path(&path)?;
    if is_aborted(signal.as_ref()) {
        return Err(aborted());
    }
    let access = fs.check_readable(&working, None).await;
    let fallback = match access {
        Ok(()) => false,
        Err(error) if error.code == FsErrorCode::NotSupported => true,
        Err(error) => return Err(fs_failure(fs.as_ref(), &working, "Cannot access", error).await),
    };
    if is_aborted(signal.as_ref()) {
        return Err(aborted());
    }
    let bytes = match fs.read_binary_file(&working, None).await {
        Ok(data) => data,
        Err(error) => {
            let site = if fallback
                && !matches!(
                    error.code,
                    FsErrorCode::IsDirectory | FsErrorCode::NotSupported
                ) {
                "Cannot access"
            } else {
                "Cannot read"
            };
            return Err(fs_failure(fs.as_ref(), &working, site, error).await);
        }
    };
    if let Some(mime) = detect_supported_image_mime_type(&bytes) {
        return super::image::read_image(&bytes, mime, &options);
    }
    let (text, details) = read_text(&bytes, &path, offset, limit)?;
    Ok(result(
        vec![ToolResultContentBlock::Text(ResultTextBlock::new(text))],
        details,
    ))
}

fn optional_number(params: &PreparedValue, name: &str) -> Option<f64> {
    params.get(name).and_then(|v| v.as_f64())
}

pub fn create_read_tool(fs: Arc<dyn FileSystem>, options: ReadToolOptions) -> ToolDefinition {
    let schema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"}
        },
        "required": ["path"]
    })).expect("static read schema is valid");
    ToolDefinition::new(
        "read",
        "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
        schema,
        "read",
        move |request: ToolExecutionRequest| {
            let fs = fs.clone();
            let options = options.clone();
            Box::pin(async move {
                if is_aborted(request.signal.as_ref()) {
                    return Err(aborted());
                }
                let path = request
                    .params
                    .get("path")
                    .and_then(argument_path)
                    .ok_or_else(|| ToolCapabilityError::new("path is required"))?;
                let offset = optional_number(&request.params, "offset");
                let limit = optional_number(&request.params, "limit");
                let signal = request.signal;
                let worker_signal = signal.clone();
                let worker = tokio::spawn(async move {
                    let outcome =
                        read(fs, options, path, offset, limit, worker_signal.clone()).await;
                    if is_aborted(worker_signal.as_ref()) {
                        Err(aborted())
                    } else {
                        outcome
                    }
                });
                super::race_abort(worker, signal).await
            })
        },
    )
}
