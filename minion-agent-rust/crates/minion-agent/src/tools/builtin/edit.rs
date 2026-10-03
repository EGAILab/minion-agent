//! TOOL-030: queued edit execution over certified EXEC-009 capabilities.

use super::{
    edit_apply::{Edit, apply_edits_with_base, normalize_lf_units},
    edit_diff::generate_edit_details_units,
    edit_prepare::prepare_edit_arguments,
    mutation_queue::Registration,
    paths::{argument_path, cause, path_message, preprocess_path},
    write::{check_abort, text_result},
};
use crate::{
    execution::{FileSystem, FsErrorCode, FsPath},
    tools::{
        AgentToolResult, PreparedValue, ToolCapabilityError, ToolDefinition, ToolExecutionRequest,
        ToolExecutionSignal,
    },
};
use serde_json::json;
use std::sync::Arc;

async fn edit(
    fs: Arc<dyn FileSystem>,
    registration: Registration,
    working: FsPath,
    path: FsPath,
    edits: Vec<Edit>,
    signal: Option<Arc<dyn ToolExecutionSignal>>,
) -> Result<AgentToolResult, ToolCapabilityError> {
    let _entry = registration.acquire().await.map_err(|e| {
        ToolCapabilityError::new(path_message(
            &format!("Cannot resolve {{path}}: {}", cause(e.code)),
            &path,
        ))
    })?;
    check_abort(signal.as_ref())?;
    let mut access = fs.check_read_write(&working, None).await;
    if access
        .as_ref()
        .is_err_and(|e| e.code == FsErrorCode::NotSupported)
    {
        access = fs.check_readable(&working, None).await;
        if access
            .as_ref()
            .is_err_and(|e| e.code == FsErrorCode::NotSupported)
        {
            access = Ok(());
        }
    }
    if let Err(e) = access {
        check_abort(signal.as_ref())?;
        return Err(ToolCapabilityError::new(path_message(
            &format!("Could not edit file: {{path}}. {}.", cause(e.code)),
            &path,
        )));
    }
    check_abort(signal.as_ref())?;
    let bytes = fs.read_binary_file(&working, None).await.map_err(|e| {
        ToolCapabilityError::new(path_message(
            &format!("Cannot read {{path}}: {}", cause(e.code)),
            &path,
        ))
    })?;
    let raw = String::from_utf8_lossy(&bytes);
    check_abort(signal.as_ref())?;
    let (bom, content) = raw
        .strip_prefix('\u{feff}')
        .map_or(("", raw.as_ref()), |text| ("\u{feff}", text));
    let crlf = content
        .find("\r\n")
        .is_some_and(|cr| content.find('\n').is_some_and(|lf| cr < lf));
    let normalized = normalize_lf_units(&content.encode_utf16().collect::<Vec<_>>());
    let (base, new) = apply_edits_with_base(&normalized, &edits, &path)?;
    check_abort(signal.as_ref())?;
    let mut final_units: Vec<u16> = bom.encode_utf16().collect();
    for unit in &new {
        if crlf && *unit == 10 {
            final_units.push(13);
        }
        final_units.push(*unit);
    }
    // Only file encoding replaces unpaired units. Runtime details below use `new`.
    let final_content = String::from_utf16_lossy(&final_units);
    fs.write_file(&working, final_content.as_bytes(), None)
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
                "Successfully replaced {} block(s) in {{path}}.",
                edits.len()
            ),
            &path,
        ),
        generate_edit_details_units(&path, &base, &new),
    ))
}

pub fn create_edit_tool(fs: Arc<dyn FileSystem>) -> ToolDefinition {
    let parameters = serde_json::from_value(json!({"type":"object","properties":{
        "path":{"type":"string","description":"Path to the file to edit (relative or absolute)"},
        "edits":{"type":"array","description":"One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.","items":{"type":"object","properties":{
            "oldText":{"type":"string","description":"Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."},
            "newText":{"type":"string","description":"Replacement text for this targeted edit."}},"required":["oldText","newText"]}}},"required":["path","edits"]})).expect("static edit schema");
    ToolDefinition::new("edit", "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.", parameters, "edit", move |request: ToolExecutionRequest| {
        let fs = fs.clone();
        Box::pin(async move {
            let values = request.params.get("edits").and_then(PreparedValue::as_array).filter(|v| !v.is_empty()).ok_or_else(|| ToolCapabilityError::new("Edit tool input is invalid. edits must contain at least one replacement."))?;
            let edits = values.iter().map(|v| match (v.get("oldText"), v.get("newText")) {
                (Some(PreparedValue::String(old)), Some(PreparedValue::String(new))) => Ok(Edit::new(old.clone(), new.clone())),
                _ => Err(ToolCapabilityError::new("oldText and newText are required")),
            }).collect::<Result<Vec<_>, ToolCapabilityError>>()?;
            let path = request.params.get("path").and_then(argument_path).ok_or_else(|| ToolCapabilityError::new("path is required"))?;
            let working = preprocess_path(&path)?;
            let registration = Registration::new(fs.clone(), working.clone());
            tokio::spawn(edit(fs, registration, working, path, edits, request.signal)).await.map_err(|e| ToolCapabilityError::new(e.to_string()))?
        })
    }).with_prepare_raw_arguments(prepare_edit_arguments)
}
