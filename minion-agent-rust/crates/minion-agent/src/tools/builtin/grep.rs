//! TOOL-037: rg owns matching; this module owns Pi's counting and formatting.
use super::{
    numeric::{js_max, js_min, number_to_string},
    paths::{OPERATION_ABORTED, argument_path, path_message, preprocess_path},
    search_engines::{SearchEngine, SearchEngines},
    search_paths,
    search_run::{self, Window},
    search_text,
    write::text_result,
};
use crate::{
    execution::{
        DirEntryProbeKind, ExecutionWorldError, FileSystem, FsPath, SpawnOptions, Subprocess,
        validate_execution_worlds,
    },
    javascript::{js_json_loads, js_trim},
    tools::{
        AgentToolResult, PreparedValue, ToolCapabilityError, ToolDefinition, ToolExecutionRequest,
    },
};
use serde_json::json;
use std::{collections::HashMap, sync::Arc};
struct Match {
    file: String,
    line: f64,
    text: Option<String>,
}
async fn execute(
    fs: Arc<dyn FileSystem>,
    subprocess: Arc<dyn Subprocess>,
    engines: Arc<dyn SearchEngines>,
    request: ToolExecutionRequest,
) -> Result<AgentToolResult, ToolCapabilityError> {
    if request.signal.as_ref().is_some_and(|s| s.is_cancelled()) {
        return Err(ToolCapabilityError::new(OPERATION_ABORTED));
    }
    let rg = engines
        .resolve(SearchEngine::Ripgrep, subprocess.execution_world())
        .await?;
    let path = request
        .params
        .get("path")
        .and_then(argument_path)
        .filter(|p| !p.code_units().is_empty())
        .unwrap_or_else(|| ".".into());
    let path = preprocess_path(path)?;
    let search = fs
        .absolute_path(&path, None)
        .await
        .map_err(|e| ToolCapabilityError::new(e.to_string()))?;
    let directory = fs
        .probe_dir_entry(&search, None)
        .await
        .map_err(|_| ToolCapabilityError::new(path_message("Path not found: {path}", &search)))?;
    let directory = matches!(
        directory.kind,
        DirEntryProbeKind::Directory | DirEntryProbeKind::SymlinkToDirectory
    );
    let root = String::from_utf16_lossy(search.code_units());
    let platform = subprocess.platform();
    let context = request
        .params
        .get("context")
        .and_then(|v| v.as_f64())
        .filter(|n| *n > 0.0)
        .unwrap_or(0.0);
    let limit = js_max(
        1.0,
        request
            .params
            .get("limit")
            .and_then(|v| v.as_f64())
            .unwrap_or(100.0),
    );
    let pattern = match request.params.get("pattern") {
        Some(PreparedValue::String(s)) => s.to_utf8_lossy(),
        _ => return Err(ToolCapabilityError::new("pattern is required")),
    };
    let mut argv = vec![
        rg.to_string_lossy().into_owned(),
        "--json".into(),
        "--line-number".into(),
        "--color=never".into(),
        "--hidden".into(),
    ];
    if request
        .params
        .get("ignoreCase")
        .is_some_and(|v| matches!(v, PreparedValue::Bool(true)))
    {
        argv.push("--ignore-case".into());
    }
    if request
        .params
        .get("literal")
        .is_some_and(|v| matches!(v, PreparedValue::Bool(true)))
    {
        argv.push("--fixed-strings".into());
    }
    if let Some(PreparedValue::String(glob)) = request.params.get("glob")
        && !glob.code_units().is_empty()
    {
        argv.extend(["--glob".into(), glob.to_utf8_lossy()]);
    }
    argv.extend(["--".into(), pattern, root.clone()]);
    let process = subprocess
        .spawn(&argv, SpawnOptions::default())
        .await
        .map_err(|e| ToolCapabilityError::new(format!("Failed to run ripgrep: {}", e.message)))?;
    let mut count = 0usize;
    let mut reached = false;
    let mut matches = Vec::new();
    let outcome = search_run::run(process, request.signal, Window::new(), false, |line| {
        if js_trim(&line).is_empty() || count as f64 >= limit {
            return false;
        }
        let Ok(event) = js_json_loads(&line) else {
            return false;
        };
        if event.get("type").and_then(|v| v.as_string()).as_deref() != Some("match") {
            return false;
        }
        count += 1;
        if let Some(data) = event.get("data")
            && let Some(file) = data
                .get("path")
                .and_then(|p| p.get("text"))
                .and_then(|v| v.as_string())
                .filter(|s| !s.is_empty())
            && let Some(line) = data.get("line_number").and_then(|v| v.as_f64())
        {
            let text = data
                .get("lines")
                .and_then(|l| l.get("text"))
                .and_then(|v| v.as_string());
            matches.push(Match { file, line, text });
        }
        if count as f64 >= limit {
            reached = true;
            true
        } else {
            false
        }
    })
    .await?;
    if outcome.aborted {
        return Err(ToolCapabilityError::new(OPERATION_ABORTED));
    }
    if !outcome.killed_for_limit && !matches!(outcome.code, Some(0 | 1)) {
        return Err(ToolCapabilityError::new(
            if js_trim(&outcome.stderr).is_empty() {
                format!(
                    "ripgrep exited with code {}",
                    outcome.code.map_or("null".into(), |n| n.to_string())
                )
            } else {
                js_trim(&outcome.stderr).into()
            },
        ));
    }
    if count == 0 {
        return Ok(text_result("No matches found", json!({})));
    }
    let mut output = Vec::new();
    let mut emitted = false;
    let mut truncated = false;
    let mut cache: HashMap<String, Vec<String>> = HashMap::new();
    for m in matches {
        let relative = search_paths::grep_path(&search, &m.file, platform, directory);
        let mut emit = |prefix: String, text: &str| {
            if emitted {
                output.push(10);
            }
            emitted = true;
            search_text::append(&mut output, &prefix);
            let (units, cut) = search_text::cut_line(text);
            output.extend(units);
            truncated |= cut;
        };
        if context == 0.0
            && let Some(text) = m.text
        {
            let normalized = text.replace("\r\n", "\n").replace('\r', "");
            let normalized = normalized.strip_suffix('\n').unwrap_or(&normalized);
            emit(
                format!("{relative}:{}: ", number_to_string(m.line)),
                normalized,
            );
            continue;
        }
        if !cache.contains_key(&m.file) {
            let lines = match fs.read_binary_file(&FsPath::from(&m.file), None).await {
                Ok(data) => String::from_utf8_lossy(&data)
                    .replace("\r\n", "\n")
                    .replace('\r', "\n")
                    .split('\n')
                    .map(str::to_owned)
                    .collect(),
                Err(_) => Vec::new(),
            };
            cache.insert(m.file.clone(), lines);
        }
        let lines = &cache[&m.file];
        if lines.is_empty() {
            emit(
                format!("{relative}:{}: ", number_to_string(m.line)),
                "(unable to read file)",
            );
            continue;
        }
        let mut current = if context > 0.0 {
            js_max(1.0, m.line - context)
        } else {
            m.line
        };
        let end = if context > 0.0 {
            js_min(lines.len() as f64, m.line + context)
        } else {
            m.line
        };
        while current <= end {
            let text = if current.fract() == 0.0 && current >= 1.0 {
                lines
                    .get(current as usize - 1)
                    .map(String::as_str)
                    .unwrap_or("")
            } else {
                ""
            };
            emit(
                if current == m.line {
                    format!("{relative}:{}: ", number_to_string(current))
                } else {
                    format!("{relative}-{}- ", number_to_string(current))
                },
                &text.replace('\r', ""),
            );
            let next = current + 1.0;
            if next == current {
                break;
            }
            current = next;
        }
    }
    Ok(search_text::finish(
        output,
        reached.then_some(("matches", limit)),
        truncated,
    ))
}
pub fn create_grep_tool(
    fs: Arc<dyn FileSystem>,
    subprocess: Arc<dyn Subprocess>,
    engines: Arc<dyn SearchEngines>,
) -> Result<ToolDefinition, ExecutionWorldError> {
    validate_execution_worlds(&[
        ("fs", fs.execution_world()),
        ("subprocess", subprocess.execution_world()),
    ])?;
    let schema=serde_json::from_value(json!({"type":"object","properties":{"pattern":{"type":"string","description":"Search pattern (regex or literal string)"},"path":{"type":"string","description":"Directory or file to search (default: current directory)"},"glob":{"type":"string","description":"Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"},"ignoreCase":{"type":"boolean","description":"Case-insensitive search (default: false)"},"literal":{"type":"boolean","description":"Treat pattern as literal string instead of regex (default: false)"},"context":{"type":"number","description":"Number of lines to show before and after each match (default: 0)"},"limit":{"type":"number","description":"Maximum number of matches to return (default: 100)"}},"required":["pattern"]})).unwrap();
    Ok(ToolDefinition::new(
        "grep",
        "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars.",
        schema,
        "grep",
        move |request| {
            Box::pin(execute(
                fs.clone(),
                subprocess.clone(),
                engines.clone(),
                request,
            ))
        },
    ))
}
