//! TOOL-036: pinned fd through certified capabilities, never a glob emulator.
use super::{
    numeric::number_to_string,
    paths::{OPERATION_ABORTED, argument_path, preprocess_path},
    search_engines::{SearchEngine, SearchEngines},
    search_glob, search_paths,
    search_run::{self, Window},
    search_text,
    write::text_result,
};
use crate::{
    execution::{
        ExecutionWorldError, FileSystem, FsPath, Platform, SpawnOptions, Subprocess,
        validate_execution_worlds,
    },
    javascript::js_trim,
    tools::{
        AgentToolResult, PreparedValue, ToolCapabilityError, ToolDefinition, ToolExecutionRequest,
    },
};
use serde_json::json;
use std::{
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
async fn execute(
    fs: Arc<dyn FileSystem>,
    subprocess: Arc<dyn Subprocess>,
    engines: Arc<dyn SearchEngines>,
    request: ToolExecutionRequest,
    window: Arc<Window>,
) -> Result<AgentToolResult, ToolCapabilityError> {
    let path = request
        .params
        .get("path")
        .and_then(argument_path)
        .filter(|p| !p.code_units().is_empty())
        .unwrap_or_else(|| ".".into());
    let working = preprocess_path(path)?;
    let search = fs
        .absolute_path(&working, None)
        .await
        .map_err(|e| ToolCapabilityError::new(e.to_string()))?;
    let root = String::from_utf16_lossy(search.code_units());
    let fd = engines
        .resolve(SearchEngine::Fd, subprocess.execution_world())
        .await?;
    if request.signal.as_ref().is_some_and(|s| s.is_cancelled()) {
        return Err(ToolCapabilityError::new(OPERATION_ABORTED));
    }
    let limit = request
        .params
        .get("limit")
        .and_then(|v| v.as_f64())
        .unwrap_or(1000.0);
    let pattern = match request.params.get("pattern") {
        Some(PreparedValue::String(s)) => s.to_utf8_lossy(),
        _ => return Err(ToolCapabilityError::new("pattern is required")),
    };
    let platform = subprocess.platform();
    let mut current = search.clone();
    let mut repo = false;
    loop {
        let git: FsPath = search_paths::logical_join(&current, ".git", platform);
        let exists = if platform == Platform::Windows {
            fs.file_info(&git, None).await.is_ok()
        } else {
            fs.probe_dir_entry(&git, None).await.is_ok()
        };
        if exists {
            repo = true;
            break;
        }
        let parent = search_paths::logical_dirname(&current, platform);
        if parent == current {
            break;
        }
        current = parent;
    }
    let mut args = vec![
        "--glob".to_owned(),
        "--color=never".into(),
        "--hidden".into(),
    ];
    if !repo {
        args.push("--no-require-git".into());
    }
    args.extend(["--max-results".into(), number_to_string(limit)]);
    let mut effective = pattern.clone();
    let mut pi = pattern.clone();
    if pattern.contains('/') {
        args.push("--full-path".into());
        if !pattern.starts_with('/') && !pattern.starts_with("**/") && pattern != "**" {
            effective = format!("**/{pattern}");
        }
        pi = effective.clone();
        if platform == Platform::Windows {
            effective = search_glob::windows_full_path(&effective);
            pi = search_glob::pi_windows(&pi);
        }
    }
    let mut first = true;
    let (outcome, lines) = loop {
        // Every spawn has a fresh verification, including the rule-5 retry.
        let binary = if first {
            fd.clone()
        } else {
            engines
                .resolve(SearchEngine::Fd, subprocess.execution_world())
                .await?
        };
        let mut argv = vec![binary.to_string_lossy().into_owned()];
        argv.extend(args.clone());
        argv.extend([
            "--".into(),
            if first { effective.clone() } else { pi.clone() },
            root.clone(),
        ]);
        let process = subprocess
            .spawn(&argv, SpawnOptions::default())
            .await
            .map_err(|e| ToolCapabilityError::new(format!("Failed to run fd: {}", e.message)))?;
        let mut lines = Vec::new();
        let outcome = search_run::run(
            process,
            if first { request.signal.clone() } else { None },
            window.clone(),
            true,
            |line| {
                lines.push(line);
                false
            },
        )
        .await?;
        if first
            && effective != pi
            && lines.join("\n").is_empty()
            && outcome.code != Some(0)
            && !outcome.aborted
            && outcome.stderr.contains("error parsing glob")
        {
            first = false;
            continue;
        }
        break (outcome, lines);
    };
    if outcome.aborted {
        return Err(ToolCapabilityError::new(OPERATION_ABORTED));
    }
    if lines.join("\n").is_empty() {
        if outcome.code != Some(0) {
            return Err(ToolCapabilityError::new(
                if js_trim(&outcome.stderr).is_empty() {
                    format!(
                        "fd exited with code {}",
                        outcome.code.map_or("null".into(), |n| n.to_string())
                    )
                } else {
                    js_trim(&outcome.stderr).into()
                },
            ));
        }
        return Ok(text_result("No files found matching pattern", json!({})));
    }
    let entries: Vec<_> = lines
        .iter()
        .filter_map(|line| {
            let line = js_trim(line.strip_suffix('\r').unwrap_or(line));
            if line.is_empty() {
                None
            } else {
                Some(search_paths::find_path_logical(&search, line, platform))
            }
        })
        .collect();
    Ok(search_text::finish(
        entries.join("\n").encode_utf16().collect(),
        (entries.len() as f64 >= limit).then_some(("results", limit)),
        false,
    ))
}
pub fn create_find_tool(
    fs: Arc<dyn FileSystem>,
    subprocess: Arc<dyn Subprocess>,
    engines: Arc<dyn SearchEngines>,
) -> Result<ToolDefinition, ExecutionWorldError> {
    validate_execution_worlds(&[
        ("fs", fs.execution_world()),
        ("subprocess", subprocess.execution_world()),
    ])?;
    let schema=serde_json::from_value(json!({"type":"object","properties":{"pattern":{"type":"string","description":"Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"},"path":{"type":"string","description":"Directory to search in (default: current directory)"},"limit":{"type":"number","description":"Maximum number of results (default: 1000)"}},"required":["pattern"]})).unwrap();
    Ok(ToolDefinition::new(
        "find",
        "Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to 1000 results or 50KB (whichever is hit first).",
        schema,
        "find",
        move |request: ToolExecutionRequest| {
            let fs = fs.clone();
            let subprocess = subprocess.clone();
            let engines = engines.clone();
            Box::pin(async move {
                if request.signal.as_ref().is_some_and(|s| s.is_cancelled()) {
                    return Err(ToolCapabilityError::new(OPERATION_ABORTED));
                }
                let signal = request.signal.clone();
                let window = Window::new();
                let w = window.clone();
                let mut worker =
                    tokio::spawn(async move { execute(fs, subprocess, engines, request, w).await });
                loop {
                    if worker.is_finished() {
                        return worker
                            .await
                            .map_err(|e| ToolCapabilityError::new(e.to_string()))?;
                    }
                    if window.active.load(Ordering::SeqCst)
                        && signal.as_ref().is_some_and(|s| s.is_cancelled())
                    {
                        window.aborted.store(true, Ordering::SeqCst);
                        return Err(ToolCapabilityError::new(OPERATION_ABORTED));
                    }
                    tokio::select! {result=&mut worker=>return result.map_err(|e|ToolCapabilityError::new(e.to_string()))?,()=tokio::time::sleep(Duration::from_millis(1))=>{}}
                }
            })
        },
    ))
}
