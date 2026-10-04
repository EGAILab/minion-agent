//! Shell discovery is distinct from command settlement: no lookup idle grace.
use crate::{
    execution::{
        EnvSnapshot, EnvValue, FileSystem, FsErrorCode, FsPath, Platform, Process, ReadableStream,
        SpawnOptions, Subprocess,
    },
    javascript::js_trim,
    tools::ToolCapabilityError,
};
use std::{sync::Arc, time::Duration};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ShellConfig {
    pub shell: String,
    pub stdin: bool,
}

impl ShellConfig {
    fn new(shell: String) -> Self {
        let normalized = shell.replace('/', "\\").to_ascii_lowercase();
        let bytes = normalized.as_bytes();
        let stdin = bytes.len() > 2
            && bytes[0].is_ascii_lowercase()
            && bytes[1] == b':'
            && matches!(
                &normalized[2..],
                "\\windows\\system32\\bash.exe" | "\\windows\\sysnative\\bash.exe"
            );
        Self { shell, stdin }
    }
    pub fn argv(&self, command: &str) -> Vec<String> {
        if self.stdin {
            vec![self.shell.clone(), "-s".into()]
        } else {
            vec![self.shell.clone(), "-c".into(), command.into()]
        }
    }
}

pub(super) async fn read(stream: &Option<Arc<dyn ReadableStream>>) -> Option<Vec<u8>> {
    match stream {
        Some(stream) => stream.read_chunk().await.ok().flatten(),
        None => std::future::pending().await,
    }
}
pub(super) async fn close(process: &Arc<dyn Process>) {
    if let Some(stream) = process.stdout() {
        stream.close().await;
    }
    if let Some(stream) = process.stderr() {
        stream.close().await;
    }
}

async fn exists(fs: &Arc<dyn FileSystem>, path: &str) -> bool {
    fs.probe_dir_entry(&FsPath::from(path), None).await.is_ok()
}

pub(super) async fn check_cwd(
    fs: &Arc<dyn FileSystem>,
    subprocess: &Arc<dyn Subprocess>,
) -> Result<(), ToolCapabilityError> {
    let path = crate::execution::path::from_native(subprocess.cwd());
    let result = if subprocess.platform() == Platform::Windows {
        fs.file_info(&path, None).await.map(|_| ())
    } else {
        fs.probe_dir_entry(&path, None).await.map(|_| ())
    };
    match result {
        Ok(()) => Ok(()),
        Err(error) if error.code == FsErrorCode::NotSupported => {
            Err(ToolCapabilityError::new(format!(
                "bash requires a filesystem provider that supports {}",
                if subprocess.platform() == Platform::Windows {
                    "file_info"
                } else {
                    "probe_dir_entry"
                }
            )))
        }
        Err(_) => Err(ToolCapabilityError::new(format!(
            "Working directory does not exist: {}\nCannot execute bash commands.",
            subprocess.cwd().display()
        ))),
    }
}

fn native_value(snapshot: &EnvSnapshot, name: &str) -> Option<String> {
    snapshot
        .get(name)
        .map(|value| match value {
            EnvValue::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            EnvValue::Windows(value) => String::from_utf16_lossy(value.code_units()),
        })
        .filter(|value| !value.is_empty())
}

async fn lookup(subprocess: &Arc<dyn Subprocess>) -> Option<String> {
    let argv = if subprocess.platform() == Platform::Windows {
        vec!["where".into(), "bash.exe".into()]
    } else {
        vec!["which".into(), "bash".into()]
    };
    let process = subprocess
        .spawn(&argv, SpawnOptions::default())
        .await
        .ok()?;
    let stdout = process.stdout();
    let stderr = process.stderr();
    let mut stdout_ended = stdout.is_none();
    let mut stderr_ended = stderr.is_none();
    let mut exited = false;
    let mut code = None;
    let mut bytes = Vec::new();
    let mut total = 0usize;
    let timer = tokio::time::sleep(Duration::from_millis(5000));
    tokio::pin!(timer);
    let wait = process.wait();
    tokio::pin!(wait);
    loop {
        if exited && stdout_ended && stderr_ended {
            break;
        }
        let interrupted = tokio::select! {
            result = &mut wait, if !exited => { exited = true; code = result.ok().and_then(|s| s.exit_code); false },
            chunk = read(&stdout), if !stdout_ended => {
                if let Some(chunk) = chunk { total += chunk.len(); bytes.extend(chunk); } else { stdout_ended = true; }
                total > 1048576
            },
            chunk = read(&stderr), if !stderr_ended => {
                if let Some(chunk) = chunk { total += chunk.len(); } else { stderr_ended = true; }
                total > 1048576
            },
            () = &mut timer => true,
        };
        if interrupted {
            // Observe an already-recorded exit before requesting the best-effort
            // kill. Neither an interruption nor the request rewrites a real 0.
            if !exited {
                if let Some(result) = futures::FutureExt::now_or_never(process.wait()) {
                    code = result.ok().and_then(|s| s.exit_code);
                } else {
                    process.terminate().await;
                    code = process.wait().await.ok().and_then(|s| s.exit_code);
                }
            }
            break;
        }
    }
    close(&process).await;
    if code != Some(0) || bytes.is_empty() {
        return None;
    }
    let decoded = String::from_utf8_lossy(&bytes);
    let first = js_trim(&decoded).split('\n').next().unwrap_or("");
    let first = first.strip_suffix('\r').unwrap_or(first);
    (!first.is_empty()).then(|| first.into())
}

pub(super) async fn select(
    fs: &Arc<dyn FileSystem>,
    subprocess: &Arc<dyn Subprocess>,
    snapshot: &EnvSnapshot,
    shell_path: Option<&str>,
) -> Result<ShellConfig, ToolCapabilityError> {
    if let Some(path) = shell_path.filter(|p| !p.is_empty()) {
        return if exists(fs, path).await {
            Ok(ShellConfig::new(path.into()))
        } else {
            Err(ToolCapabilityError::new(format!(
                "Custom shell path not found: {path}"
            )))
        };
    }
    if subprocess.platform() == Platform::Windows {
        let candidates: Vec<_> = ["ProgramFiles", "ProgramFiles(x86)"]
            .into_iter()
            .filter_map(|name| native_value(snapshot, name))
            .map(|prefix| format!("{prefix}\\Git\\bin\\bash.exe"))
            .collect();
        for path in &candidates {
            if exists(fs, path).await {
                return Ok(ShellConfig::new(path.clone()));
            }
        }
        if let Some(path) = lookup(subprocess).await
            && exists(fs, &path).await
        {
            return Ok(ShellConfig::new(path));
        }
        Err(ToolCapabilityError::new(format!(
            "No bash shell found. Options:\n  1. Install Git for Windows: https://git-scm.com/download/win\n  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n  3. Set shellPath in settings.json\n\nSearched Git Bash in:\n{}",
            candidates
                .iter()
                .map(|p| format!("  {p}"))
                .collect::<Vec<_>>()
                .join("\n")
        )))
    } else {
        if exists(fs, "/bin/bash").await {
            return Ok(ShellConfig::new("/bin/bash".into()));
        }
        Ok(ShellConfig::new(
            lookup(subprocess).await.unwrap_or_else(|| "sh".into()),
        ))
    }
}
