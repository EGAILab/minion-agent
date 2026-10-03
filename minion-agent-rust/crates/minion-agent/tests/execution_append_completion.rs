//! EXEC-002-R1: append success means the actual write has completed, not merely
//! that Tokio accepted bytes into its pending blocking-write buffer.
use std::{
    future::Future,
    sync::{Arc, mpsc},
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

use minion_agent::execution::{CancellationController, LocalFileSystem};
use uuid::Uuid;

struct Notify(mpsc::Sender<()>);

impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        let _ = self.0.send(());
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.send(());
    }
}

fn hold_worker() -> (mpsc::Sender<()>, tokio::task::JoinHandle<()>) {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let blocker = tokio::task::spawn_blocking(move || {
        entered_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    (release_tx, blocker)
}

fn controlled_append(use_aborted_signal: bool) {
    let root = std::env::temp_dir().join(format!("minion-append-completion-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("file.txt"), "content").unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let fs = LocalFileSystem::new(&root);
        let controller = CancellationController::default();
        let signal = controller.signal();
        controller.abort();
        let (wake_tx, wake_rx) = mpsc::channel();
        let waker = Waker::from(Arc::new(Notify(wake_tx)));
        let mut cx = Context::from_waker(&waker);
        let mut append = Box::pin(fs.append_file(
            "file.txt",
            b"!",
            if use_aborted_signal {
                Some(&signal)
            } else {
                None
            },
        ));
        // Drive the real Tokio mkdir and open tasks to completion. The timeout
        // guards deadlock/setup failure; it is not a scheduling delay or retry.
        // Hold the worker at each setup poll too: otherwise a very fast mkdir
        // or open could finish within its first poll and collapse two stages.
        for _ in 0..2 {
            let (release, blocker) = hold_worker();
            let setup = append.as_mut().poll(&mut cx);
            release.send(()).unwrap();
            blocker.await.unwrap();
            assert!(setup.is_pending());
            wake_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        }

        // Occupy the sole blocking worker before the real append schedules its
        // write. There is no mock file/write and no production callback bypass.
        let (release_tx, blocker) = hold_worker();
        let completion = append.as_mut().poll(&mut cx);
        let before = std::fs::read_to_string(root.join("file.txt"));

        // Release/drain even for a failing mutant, so runtime shutdown cannot
        // deadlock on a worker held by the test. No sleep/retry masks completion.
        release_tx.send(()).unwrap();
        blocker.await.unwrap();
        let was_pending = completion.is_pending();
        match completion {
            Poll::Ready(result) => result.unwrap(),
            Poll::Pending => append.await.unwrap(),
        }
        let after = tokio::fs::read_to_string(root.join("file.txt"))
            .await
            .unwrap();
        assert_eq!(before.unwrap(), "content");
        assert_eq!(after, "content!");
        assert!(
            was_pending,
            "append returned success before its blocking write could execute"
        );
    });
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn append_waits_for_write_completion_without_a_signal() {
    controlled_append(false);
}

#[test]
fn append_waits_for_write_completion_and_does_not_inspect_an_aborted_signal() {
    controlled_append(true);
}
