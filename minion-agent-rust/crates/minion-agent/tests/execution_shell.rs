use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

use minion_agent::execution::{
    CancellationController, LocalShell, LocalSubprocess, Shell, ShellErrorCode, ShellExecOptions,
};
use uuid::Uuid;

fn temp_root() -> PathBuf {
    let path = std::env::temp_dir().join(format!("minion-shell-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

async fn remove_root(root: PathBuf) {
    for attempt in 0..20 {
        match std::fs::remove_dir_all(&root) {
            Ok(()) => return,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(error) if attempt < 19 => {
                let _ = error;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(error) => panic!("failed to remove {}: {error}", root.display()),
        }
    }
}

fn shell_path() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("PATH")
            .and_then(|path| {
                std::env::split_paths(&path)
                    .map(|directory| directory.join("bash.exe"))
                    .find(|candidate| candidate.is_file())
            })
            .or_else(|| {
                let path = PathBuf::from(r"C:\Program Files\Git\bin\bash.exe");
                path.is_file().then_some(path)
            })
    } else {
        Some(PathBuf::from("/bin/sh"))
    }
}

fn provider(root: &PathBuf) -> Option<Arc<LocalShell>> {
    let shell = shell_path()?;
    let subprocess = Arc::new(LocalSubprocess::new(root));
    Some(Arc::new(
        LocalShell::new(subprocess).with_configured_shell(shell),
    ))
}

#[tokio::test]
async fn shell_accumulates_streams_invokes_callbacks_and_preserves_nonzero_exit() {
    let root = temp_root();
    let Some(shell) = provider(&root) else { return };
    let chunks = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let output = shell
        .exec(
            "printf 'out'; printf 'err' >&2; exit 7",
            ShellExecOptions {
                on_stdout: Some({
                    let chunks = Arc::clone(&chunks);
                    Arc::new(move |chunk| {
                        chunks.lock().push(chunk.to_owned());
                        Ok(())
                    })
                }),
                ..ShellExecOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(output.stdout, "out");
    assert_eq!(output.stderr, "err");
    assert_eq!(output.exit_code, 7);
    assert_eq!(chunks.lock().concat(), "out");
    remove_root(root).await;
}

#[tokio::test]
async fn callback_failure_terminates_and_wins_classification() {
    let root = temp_root();
    let Some(shell) = provider(&root) else { return };
    let error = shell
        .exec(
            "printf 'data'; sleep 2",
            ShellExecOptions {
                timeout_seconds: Some(0.5),
                on_stdout: Some(Arc::new(|_| Err("callback failed".into()))),
                ..ShellExecOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ShellErrorCode::CallbackError);
    remove_root(root).await;
}

#[tokio::test]
async fn timeout_validation_and_preabort_happen_before_spawn() {
    let root = temp_root();
    let Some(shell) = provider(&root) else { return };
    assert_eq!(
        shell
            .exec(
                "echo never",
                ShellExecOptions {
                    timeout_seconds: Some(2_147_483.648),
                    ..ShellExecOptions::default()
                },
            )
            .await
            .unwrap_err()
            .code,
        ShellErrorCode::Timeout
    );
    assert!(
        shell
            .exec(
                "exit 0",
                ShellExecOptions {
                    timeout_seconds: Some(2_147_483.647),
                    ..ShellExecOptions::default()
                },
            )
            .await
            .is_ok()
    );
    let controller = CancellationController::default();
    controller.abort();
    assert_eq!(
        shell
            .exec(
                "echo never",
                ShellExecOptions {
                    timeout_seconds: Some(f64::NAN),
                    signal: Some(Arc::new(controller.signal())),
                    ..ShellExecOptions::default()
                },
            )
            .await
            .unwrap_err()
            .code,
        ShellErrorCode::Aborted
    );
    remove_root(root).await;
}

#[tokio::test]
async fn shell_resolution_precedes_the_cwd_existence_check() {
    let root = temp_root();
    let subprocess = Arc::new(LocalSubprocess::new(&root));
    let shell = LocalShell::new(subprocess).with_configured_shell(root.join("missing-shell"));
    let error = shell
        .exec(
            "echo never",
            ShellExecOptions {
                cwd: Some(root.join("missing-cwd")),
                ..ShellExecOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ShellErrorCode::ShellUnavailable);
    remove_root(root).await;
}

#[tokio::test]
async fn inherit_env_controls_the_provider_base_environment() {
    if cfg!(windows) {
        // MSYS bash requires Windows process-bootstrap variables even when the child contract
        // intentionally requests an otherwise empty environment. The subprocess seam's direct
        // environment witness covers the Windows binding; this shell-specific assertion is POSIX.
        return;
    }
    let root = temp_root();
    let Some(shell_path) = shell_path() else {
        return;
    };
    let mut base = BTreeMap::new();
    base.insert("BASE_ONLY".into(), "base".into());
    let subprocess = Arc::new(LocalSubprocess::new(&root).with_base_env(base));
    let shell = LocalShell::new(subprocess).with_configured_shell(shell_path);
    let inherited = shell
        .exec("printf \"${BASE_ONLY-unset}\"", ShellExecOptions::default())
        .await
        .unwrap();
    let isolated = shell
        .exec(
            "printf \"${BASE_ONLY-unset}\"",
            ShellExecOptions {
                inherit_env: false,
                ..ShellExecOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(inherited.stdout, "base");
    assert_eq!(isolated.stdout, "unset");
    remove_root(root).await;
}

#[tokio::test]
async fn runtime_timeout_and_signal_are_distinct() {
    let root = temp_root();
    let Some(shell) = provider(&root) else { return };
    assert_eq!(
        shell
            .exec(
                "sleep 2",
                ShellExecOptions {
                    timeout_seconds: Some(0.05),
                    ..ShellExecOptions::default()
                },
            )
            .await
            .unwrap_err()
            .code,
        ShellErrorCode::Timeout
    );

    let controller = CancellationController::default();
    let controller_for_task = controller.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        controller_for_task.abort();
    });
    assert_eq!(
        shell
            .exec(
                "sleep 2",
                ShellExecOptions {
                    signal: Some(Arc::new(controller.signal())),
                    ..ShellExecOptions::default()
                },
            )
            .await
            .unwrap_err()
            .code,
        ShellErrorCode::Aborted
    );
    remove_root(root).await;
}

#[tokio::test]
async fn cleanup_terminates_active_commands_without_classifying_abort() {
    let root = temp_root();
    let Some(shell) = provider(&root) else { return };
    let running = {
        let shell = Arc::clone(&shell);
        tokio::spawn(async move { shell.exec("sleep 10", ShellExecOptions::default()).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    shell.cleanup().await;
    let output = tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(output.exit_code >= 0);
    remove_root(root).await;
}
