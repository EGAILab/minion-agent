//! TOOL-034/035: capability-neutral bash execution, settlement and persistence.
#[cfg(test)]
#[path = "bash_tests.rs"]
mod tests;
use super::{
    bash_output::Output, bash_shell, environment::compose_spawn_environment,
    numeric::number_to_string, paths::cause, write::text_result,
};
use crate::{
    execution::{
        AbortSignal, ExecutionWorldError, FileSystem, FsPath, SpawnOptions, StdioMode, Subprocess,
        validate_execution_worlds,
    },
    tools::{
        AgentToolResult, PreparedValue, ToolCapabilityError, ToolDefinition, ToolExecutionRequest,
        ToolExecutionSignal,
    },
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio::{sync::mpsc, time::Instant};

/// Only the approved factory options; cwd is owned by the subprocess capability.
#[derive(Clone, Debug)]
pub struct BashToolOptions {
    pub shell_path: Option<String>,
    pub expose_session_environment: bool,
}
impl Default for BashToolOptions {
    fn default() -> Self {
        Self {
            shell_path: None,
            expose_session_environment: true,
        }
    }
}

struct Signal(Arc<dyn ToolExecutionSignal>);
impl AbortSignal for Signal {
    fn aborted(&self) -> bool {
        self.0.is_cancelled()
    }
}

fn timeout(request: &ToolExecutionRequest) -> Result<Option<f64>, ToolCapabilityError> {
    let Some(value) = request
        .params
        .get("timeout")
        .and_then(|value| value.as_f64())
    else {
        return Ok(None);
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(ToolCapabilityError::new(
            "Invalid timeout: must be a finite number of seconds",
        ));
    }
    if value * 1000.0 > 2147483647.0 {
        return Err(ToolCapabilityError::new(
            "Invalid timeout: maximum is 2147483.647 seconds",
        ));
    }
    Ok(Some(value))
}

async fn persist(
    fs: Arc<dyn FileSystem>,
    mut receiver: mpsc::UnboundedReceiver<Vec<Vec<u8>>>,
) -> Result<Option<String>, ToolCapabilityError> {
    let mut path = None;
    while let Some(chunks) = receiver.recv().await {
        if path.is_none() {
            path = Some(
                fs.create_temp_file("minion-bash-", ".log", None)
                    .await
                    .map_err(|e| {
                        ToolCapabilityError::new(format!(
                            "Cannot write the full-output file: {}",
                            cause(e.code)
                        ))
                    })?,
            );
        }
        for chunk in chunks {
            fs.append_file(
                &FsPath::from(path.as_deref().expect("created path")),
                &chunk,
                None,
            )
            .await
            .map_err(|e| {
                ToolCapabilityError::new(format!(
                    "Cannot write the full-output file: {}",
                    cause(e.code)
                ))
            })?;
        }
    }
    Ok(path)
}

async fn execute(
    fs: Arc<dyn FileSystem>,
    subprocess: Arc<dyn Subprocess>,
    options: BashToolOptions,
    request: ToolExecutionRequest,
) -> Result<AgentToolResult, ToolCapabilityError> {
    // Capture before timeout validation and shell lookup, exactly once per call.
    let snapshot = subprocess.base_env();
    let mut inject = BTreeMap::new();
    if options.expose_session_environment
        && let Some(context) = &request.context
    {
        inject.insert("MINION_SESSION_ID".into(), context.session_id().into());
        if let Some(file) = context.session_file() {
            inject.insert("MINION_SESSION_FILE".into(), file.into());
        }
        if let (Some(provider), Some(model)) = (context.provider(), context.model()) {
            inject.insert("MINION_PROVIDER".into(), provider.into());
            inject.insert("MINION_MODEL".into(), model.into());
        }
        if let Some(reasoning) = context.reasoning_level().filter(|s| !s.is_empty()) {
            inject.insert("MINION_REASONING_LEVEL".into(), reasoning.into());
        }
    }
    let env = compose_spawn_environment(
        &snapshot,
        &[
            "MINION_SESSION_ID",
            "MINION_SESSION_FILE",
            "MINION_PROVIDER",
            "MINION_MODEL",
            "MINION_REASONING_LEVEL",
        ],
        &inject,
    )?;
    let seconds = timeout(&request)?;
    if request.signal.as_ref().is_some_and(|s| s.is_cancelled()) {
        return Err(ToolCapabilityError::new("Command aborted"));
    }
    let shell =
        bash_shell::select(&fs, &subprocess, &snapshot, options.shell_path.as_deref()).await?;
    bash_shell::check_cwd(&fs, &subprocess).await?;
    let command = match request.params.get("command") {
        Some(PreparedValue::String(s)) => s.to_utf8_lossy(),
        _ => return Err(ToolCapabilityError::new("command is required")),
    };
    let process = subprocess
        .spawn(
            &shell.argv(&command),
            SpawnOptions {
                env,
                inherit_env: false,
                stdin: if shell.stdin {
                    StdioMode::Piped
                } else {
                    StdioMode::Null
                },
                signal: request
                    .signal
                    .as_ref()
                    .map(|s| Arc::new(Signal(s.clone())) as Arc<dyn AbortSignal>),
                ..SpawnOptions::default()
            },
        )
        .await
        .map_err(|_| {
            ToolCapabilityError::new(format!("Failed to start the shell {}", shell.shell))
        })?;
    let stdin = async {
        if shell.stdin
            && let Some(stream) = process.stdin()
        {
            let _ = stream.write(command.as_bytes()).await;
            stream.close().await;
        }
    };
    tokio::pin!(stdin);
    let mut stdin_done = false;
    let mut output = Output::new();
    let (sender, receiver) = mpsc::unbounded_channel();
    let writer = persist(fs, receiver);
    tokio::pin!(writer);
    let stdout = process.stdout();
    let stderr = process.stderr();
    let mut stdout_done = stdout.is_none();
    let mut stderr_done = stderr.is_none();
    let wait = process.wait();
    tokio::pin!(wait);
    let mut exited = false;
    let mut exit_code = None;
    let mut timed_out = false;
    let deadline = seconds.map(|seconds| {
        Instant::now() + Duration::from_millis((seconds * 1000.0).trunc().max(1.0) as u64)
    });
    let mut grace = None;
    loop {
        if exited && stdout_done && stderr_done {
            break;
        }
        tokio::select! {
            result = &mut writer => { // Only a persistence failure can complete while sender is live.
                process.terminate().await; bash_shell::close(&process).await;
                return Err(result.expect_err("writer cannot finish before sender closes"));
            },
            result = &mut wait, if !exited => { exited = true; exit_code = result.ok().and_then(|s|s.exit_code); grace = Some(Instant::now() + Duration::from_millis(100)); },
            chunk = bash_shell::read(&stdout), if !stdout_done => {
                if let Some(bytes) = chunk {
                    let writes = output.append(bytes);
                    if !writes.is_empty() { let _ = sender.send(writes); }
                    if exited { grace = Some(Instant::now() + Duration::from_millis(100)); }
                } else { stdout_done = true; }
            },
            chunk = bash_shell::read(&stderr), if !stderr_done => {
                if let Some(bytes) = chunk {
                    let writes = output.append(bytes);
                    if !writes.is_empty() { let _ = sender.send(writes); }
                    if exited { grace = Some(Instant::now() + Duration::from_millis(100)); }
                } else { stderr_done = true; }
            },
            () = async { if let Some(deadline) = deadline { tokio::time::sleep_until(deadline).await; } else { std::future::pending().await } }, if !timed_out => { timed_out = true; process.terminate().await; },
            () = async { if let Some(grace) = grace { tokio::time::sleep_until(grace).await; } else { std::future::pending().await } } => break,
            () = &mut stdin, if !stdin_done => { stdin_done = true; },
        }
    }
    // Release read ends, never kill descendants merely to complete grace. Freeze
    // the outcome before joining accepted log writes: persistence has no timeout.
    let aborted = request.signal.as_ref().is_some_and(|s| s.is_cancelled());
    bash_shell::close(&process).await;
    let writes = output.finish();
    if !writes.is_empty() {
        let _ = sender.send(writes);
    }
    drop(sender);
    let path = match writer.await {
        Ok(path) => path,
        Err(error) => {
            process.terminate().await;
            return Err(error);
        }
    };
    let empty = if aborted || timed_out {
        ""
    } else {
        "(no output)"
    };
    let formatted = output.snapshot(path.as_deref(), empty);
    let suffix = if aborted {
        Some("Command aborted".into())
    } else if timed_out {
        Some(format!(
            "Command timed out after {} seconds",
            number_to_string(seconds.expect("timer configured"))
        ))
    } else {
        exit_code
            .filter(|code| *code != 0)
            .map(|code| format!("Command exited with code {code}"))
    };
    if let Some(suffix) = suffix {
        return Err(ToolCapabilityError::new(if formatted.text.is_empty() {
            suffix
        } else {
            format!("{}\n\n{suffix}", formatted.text)
        }));
    }
    Ok(text_result(formatted.text, formatted.details))
}

/// Create the approved bash tool using compatible execution capabilities.
pub fn create_bash_tool(
    fs: Arc<dyn FileSystem>,
    subprocess: Arc<dyn Subprocess>,
    options: BashToolOptions,
) -> Result<ToolDefinition, ExecutionWorldError> {
    validate_execution_worlds(&[
        ("fs", fs.execution_world()),
        ("subprocess", subprocess.execution_world()),
    ])?;
    Ok(ToolDefinition::new("bash", "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.", serde_json::from_value(serde_json::json!({"type":"object","properties":{"command":{"type":"string","description":"Bash command to execute"},"timeout":{"type":"number","description":"Timeout in seconds (optional, no default timeout)"}},"required":["command"]})).expect("static bash schema"), "bash", move |request| Box::pin(execute(fs.clone(), subprocess.clone(), options.clone(), request))))
}
