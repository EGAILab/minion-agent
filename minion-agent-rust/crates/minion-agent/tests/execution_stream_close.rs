//! EXEC-012: real inherited pipes, controlled by stdin handshakes rather than
//! sleeps/retries. The fixture runs this test executable in a separate process.
use std::{
    collections::BTreeMap,
    future::Future,
    io::{Read, Write},
    process::{Command, Stdio},
    sync::Arc,
    task::Poll,
    time::Duration,
};

use minion_agent::execution::{
    LocalSubprocess, Process, ReadableStream, SpawnOptions, StdioMode, Subprocess,
};

const BOUND: Duration = Duration::from_secs(5);

#[test]
fn d002_child_fixture() {
    let Ok(mode) = std::env::var("MINION_D002_FIXTURE") else {
        return;
    };
    if mode == "parent" {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "d002_child_fixture", "--nocapture"])
            .env("MINION_D002_FIXTURE", "descendant")
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        // Keep the descendant's inherited descriptors, but do not wait for it.
        drop(child);
        std::process::exit(23);
    }
    if mode == "stdout-after-input" {
        eprintln!("D002-READY");
        std::io::stderr().flush().unwrap();
    } else {
        if mode == "buffered" {
            println!("D002-BUFFERED");
            std::io::stdout().flush().unwrap();
            eprintln!("D002-BUFFER-READY");
            std::io::stderr().flush().unwrap();
        }
        println!("D002-READY");
        std::io::stdout().flush().unwrap();
    }
    let mut input = [0];
    if std::io::stdin().read_exact(&mut input).is_err() {
        std::process::exit(0);
    }
    match mode.as_str() {
        "stderr-after-input" => eprintln!("D002-SURVIVED"),
        "stdout-after-input" => println!("D002-SURVIVED"),
        "descendant" => {
            let result = std::io::stdout().write_all(b"D002-LATE\n");
            let _ = std::io::stdout().flush();
            eprintln!(
                "D002-WRITE-{}",
                if result.is_ok() { "OK" } else { "CLOSED" }
            );
        }
        _ => {}
    }
    std::process::exit(if mode == "descendant" { 0 } else { 23 });
}

async fn spawn(mode: &str) -> Arc<dyn Process> {
    let argv = vec![
        std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        "--exact".into(),
        "d002_child_fixture".into(),
        "--nocapture".into(),
    ];
    LocalSubprocess::new(std::env::current_dir().unwrap())
        .spawn(
            &argv,
            SpawnOptions {
                stdin: StdioMode::Piped,
                env: BTreeMap::from([("MINION_D002_FIXTURE".into(), mode.into())]),
                ..SpawnOptions::default()
            },
        )
        .await
        .unwrap()
}

async fn until(stream: &Arc<dyn ReadableStream>, marker: &str) -> Vec<u8> {
    tokio::time::timeout(BOUND, async {
        let mut bytes = Vec::new();
        loop {
            bytes.extend(
                stream
                    .read_chunk()
                    .await
                    .unwrap()
                    .expect("marker before EOF"),
            );
            if String::from_utf8_lossy(&bytes).contains(marker) {
                return bytes;
            }
        }
    })
    .await
    .expect("fixture handshake")
}

async fn drain(stream: &Arc<dyn ReadableStream>) -> Vec<u8> {
    tokio::time::timeout(BOUND, async {
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.read_chunk().await.unwrap() {
            bytes.extend(chunk);
        }
        bytes
    })
    .await
    .expect("EOF")
}

#[tokio::test]
async fn exit_is_independent_of_inherited_pipes_and_late_output_is_readable() {
    let process = spawn("parent").await;
    let stdout = process.stdout().unwrap();
    until(&stdout, "D002-READY").await;
    // The descendant is blocked on stdin, so its output cannot reach EOF yet.
    assert_eq!(
        tokio::time::timeout(BOUND, process.wait())
            .await
            .unwrap()
            .unwrap()
            .exit_code,
        Some(23)
    );
    let (first, second) = tokio::join!(process.wait(), process.wait());
    assert_eq!(first.unwrap(), second.unwrap());
    process.stdin().unwrap().write(b"x").await.unwrap();
    let late = drain(&stdout).await;
    assert!(String::from_utf8_lossy(&late).contains("D002-LATE"));
    drain(&process.stderr().unwrap()).await;
    assert_eq!(stdout.read_chunk().await.unwrap(), None);
    stdout.close().await;
    process.stdin().unwrap().close().await;
}

#[tokio::test]
async fn close_wakes_a_pending_read_and_does_not_close_the_sibling_or_kill() {
    for mode in ["stderr-after-input", "stdout-after-input"] {
        let process = spawn(mode).await;
        let (stream, sibling) = if mode == "stderr-after-input" {
            (process.stdout().unwrap(), process.stderr().unwrap())
        } else {
            (process.stderr().unwrap(), process.stdout().unwrap())
        };
        until(&stream, "D002-READY").await;
        let mut pending = Box::pin(stream.read_chunk());
        std::future::poll_fn(|cx| {
            assert!(pending.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
        // Close must release the OS handle even while this Pending future is
        // parked, not relying on its caller resuming polling to release a lock.
        tokio::time::timeout(BOUND, stream.close())
            .await
            .expect("close cannot deadlock on a parked read");
        assert_eq!(
            tokio::time::timeout(BOUND, pending).await.unwrap().unwrap(),
            None
        );
        assert_eq!(stream.read_chunk().await.unwrap(), None);
        stream.close().await;
        // A live child produces this only after close and a fresh stdin command.
        process.stdin().unwrap().write(b"x").await.unwrap();
        let output = drain(&sibling).await;
        assert!(String::from_utf8_lossy(&output).contains("D002-SURVIVED"));
        assert_eq!(process.wait().await.unwrap().exit_code, Some(23));
        sibling.close().await;
        process.stdin().unwrap().close().await;
    }
}

#[tokio::test]
async fn close_abandons_unread_output() {
    let process = spawn("buffered").await;
    let stdout = process.stdout().unwrap();
    until(&process.stderr().unwrap(), "D002-BUFFER-READY").await;
    stdout.close().await;
    assert_eq!(stdout.read_chunk().await.unwrap(), None);
    process.stdin().unwrap().write(b"x").await.unwrap();
    assert_eq!(process.wait().await.unwrap().exit_code, Some(23));
    drain(&process.stderr().unwrap()).await;
    process.stdin().unwrap().close().await;
}

#[tokio::test]
async fn closing_the_read_end_preserves_a_descendant_and_releases_the_pipe() {
    let process = spawn("parent").await;
    let stdout = process.stdout().unwrap();
    let stderr = process.stderr().unwrap();
    until(&stdout, "D002-READY").await;
    assert_eq!(process.wait().await.unwrap().exit_code, Some(23));
    stdout.close().await;
    // This descendant's response distinguishes closing a read end from killing
    // it: it remains alive to receive input, attempt a write and report failure.
    process.stdin().unwrap().write(b"x").await.unwrap();
    let output = drain(&stderr).await;
    assert!(String::from_utf8_lossy(&output).contains("D002-WRITE-CLOSED"));
    assert_eq!(stdout.read_chunk().await.unwrap(), None);
    stderr.close().await;
    process.stdin().unwrap().close().await;
}

#[tokio::test]
async fn cancellation_of_a_waiter_does_not_poison_later_waits() {
    let process = spawn("stderr-after-input").await;
    until(&process.stdout().unwrap(), "D002-READY").await;
    let mut cancelled = Box::pin(process.wait());
    std::future::poll_fn(|cx| {
        assert!(cancelled.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    })
    .await;
    drop(cancelled);
    process.stdin().unwrap().write(b"x").await.unwrap();
    assert_eq!(
        tokio::time::timeout(BOUND, process.wait())
            .await
            .unwrap()
            .unwrap()
            .exit_code,
        Some(23)
    );
    drain(&process.stdout().unwrap()).await;
    drain(&process.stderr().unwrap()).await;
    process.stdin().unwrap().close().await;
}

#[tokio::test]
async fn terminate_after_settlement_releases_even_retained_stream_handles() {
    let process = spawn("parent").await;
    let stdout = process.stdout().unwrap();
    let stderr = process.stderr().unwrap();
    let stdin = process.stdin().unwrap();
    until(&stdout, "D002-READY").await;
    assert_eq!(process.wait().await.unwrap().exit_code, Some(23));
    process.terminate().await;
    process.terminate().await;
    assert_eq!(
        tokio::time::timeout(BOUND, stdout.read_chunk())
            .await
            .unwrap()
            .unwrap(),
        None
    );
    assert_eq!(
        tokio::time::timeout(BOUND, stderr.read_chunk())
            .await
            .unwrap()
            .unwrap(),
        None
    );
    assert!(stdin.write(b"x").await.is_err());
    assert_eq!(process.wait().await.unwrap().exit_code, Some(23));
}
