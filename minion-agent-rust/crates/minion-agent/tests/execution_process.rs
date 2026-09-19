use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

use minion_agent::execution::{
    CancellationController, LocalSubprocess, Process, SpawnOptions, StdioMode, Subprocess,
    SubprocessErrorCode,
};
use url::Url;
use uuid::Uuid;

fn temp_root() -> PathBuf {
    let path = std::env::temp_dir().join(format!("minion-process-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn shell_argv(command: &str) -> Vec<String> {
    if cfg!(windows) {
        vec!["cmd.exe".into(), "/C".into(), command.into()]
    } else {
        vec!["sh".into(), "-c".into(), command.into()]
    }
}

fn long_running_argv() -> Vec<String> {
    if cfg!(windows) {
        shell_argv("ping 127.0.0.1 -n 10 >NUL")
    } else {
        shell_argv("sleep 10")
    }
}

async fn read_all(stream: Arc<dyn minion_agent::execution::ReadableStream>) -> Vec<u8> {
    let mut result = Vec::new();
    while let Some(chunk) = stream.read_chunk().await.unwrap() {
        result.extend(chunk);
    }
    result
}

#[tokio::test]
async fn spawn_is_argv_direct_and_wait_is_repeatable() {
    let root = temp_root();
    let provider = LocalSubprocess::new(&root);
    let process = provider
        .spawn(&shell_argv("echo hello"), SpawnOptions::default())
        .await
        .unwrap();
    let output = read_all(process.stdout().unwrap()).await;
    let (first, second) = tokio::join!(process.wait(), process.wait());
    assert_eq!(first.unwrap(), second.unwrap());
    assert_eq!(String::from_utf8_lossy(&output).trim(), "hello");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn piped_stdin_and_environment_are_typed_and_live() {
    let root = temp_root();
    let provider = LocalSubprocess::new(&root);
    let argv = if cfg!(windows) {
        vec![
            "powershell.exe".into(),
            "-NoProfile".into(),
            "-Command".into(),
            "$x=[Console]::In.ReadLine(); Write-Output ($env:TEST_VALUE + ':' + $x)".into(),
        ]
    } else {
        shell_argv("read X; printf '%s:%s' \"$TEST_VALUE\" \"$X\"")
    };
    let mut env = BTreeMap::new();
    env.insert("TEST_VALUE".into(), "env".into());
    let process = provider
        .spawn(
            &argv,
            SpawnOptions {
                env,
                inherit_env: true,
                stdin: StdioMode::Piped,
                ..SpawnOptions::default()
            },
        )
        .await
        .unwrap();
    let stdin = process.stdin().unwrap();
    stdin.write(b"input\n").await.unwrap();
    stdin.close().await;
    let output = read_all(process.stdout().unwrap()).await;
    assert_eq!(process.wait().await.unwrap().exit_code, Some(0));
    assert_eq!(String::from_utf8_lossy(&output).trim(), "env:input");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn explicit_termination_is_success_with_the_os_reported_exit_code() {
    let root = temp_root();
    let provider = LocalSubprocess::new(&root);
    let process = provider
        .spawn(&long_running_argv(), SpawnOptions::default())
        .await
        .unwrap();
    process.terminate().await;
    process.terminate().await;
    let first = process.wait().await.unwrap();
    let second = process.wait().await.unwrap();
    assert_eq!(first, second);
    #[cfg(windows)]
    assert!(first.exit_code.is_some());
    #[cfg(unix)]
    assert_eq!(first.exit_code, None);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn cwd_uses_the_filesystem_lexical_normalization_rule() {
    let root = temp_root();
    let nested = root.join("nested");
    std::fs::create_dir(&nested).unwrap();
    let provider = LocalSubprocess::new(&root);
    let cwd_url = Url::from_directory_path(&nested).unwrap();
    let process = provider
        .spawn(
            &shell_argv("echo normalized"),
            SpawnOptions {
                cwd: Some(PathBuf::from(cwd_url.as_str())),
                ..SpawnOptions::default()
            },
        )
        .await
        .unwrap();
    let output = read_all(process.stdout().unwrap()).await;
    assert_eq!(process.wait().await.unwrap().exit_code, Some(0));
    assert_eq!(String::from_utf8_lossy(&output).trim(), "normalized");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn spawn_signal_controls_pre_and_post_spawn_cancellation() {
    let root = temp_root();
    let provider = LocalSubprocess::new(&root);
    let controller = CancellationController::default();
    let signal = Arc::new(controller.signal());
    controller.abort();
    assert_eq!(
        provider
            .spawn(
                &shell_argv("echo never"),
                SpawnOptions {
                    signal: Some(signal),
                    ..SpawnOptions::default()
                },
            )
            .await
            .err()
            .expect("pre-aborted spawn must fail")
            .code,
        SubprocessErrorCode::Aborted
    );

    let controller = CancellationController::default();
    let process = provider
        .spawn(
            &long_running_argv(),
            SpawnOptions {
                signal: Some(Arc::new(controller.signal())),
                ..SpawnOptions::default()
            },
        )
        .await
        .unwrap();
    controller.abort();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), process.wait())
            .await
            .unwrap()
            .unwrap_err()
            .code,
        SubprocessErrorCode::Aborted
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn missing_program_and_missing_cwd_are_spawn_errors() {
    let root = temp_root();
    let provider = LocalSubprocess::new(&root);
    let missing_program = provider
        .spawn(
            &["definitely-not-a-real-minion-binary".into()],
            SpawnOptions::default(),
        )
        .await
        .err()
        .unwrap();
    assert_eq!(missing_program.code, SubprocessErrorCode::SpawnError);
    let missing_cwd = provider
        .spawn(
            &shell_argv("echo never"),
            SpawnOptions {
                cwd: Some(root.join("missing")),
                ..SpawnOptions::default()
            },
        )
        .await
        .err()
        .unwrap();
    assert_eq!(missing_cwd.code, SubprocessErrorCode::SpawnError);
    std::fs::remove_dir_all(root).unwrap();
}

#[allow(dead_code)]
fn _process_is_object_safe(_: Arc<dyn Process>) {}
