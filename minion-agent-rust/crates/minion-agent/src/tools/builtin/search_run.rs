//! The search completion boundary is own exit AND both pipe EOFs, not wait alone.
use crate::{
    execution::{Process, ReadableStream, SubprocessError},
    tools::{ToolCapabilityError, ToolExecutionSignal},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[derive(Default)]
pub(super) struct Window {
    pub active: AtomicBool,
    pub aborted: AtomicBool,
}
impl Window {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            active: AtomicBool::new(true),
            aborted: AtomicBool::new(false),
        })
    }
    pub fn complete(&self) {
        self.active.store(false, Ordering::SeqCst);
    }
}

#[derive(Default)]
struct Decoder {
    pending: Vec<u8>,
}
impl Decoder {
    fn feed(&mut self, bytes: &[u8], finish: bool) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut start = 0;
        while start < self.pending.len() {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(valid) => {
                    out.push_str(valid);
                    start = self.pending.len();
                }
                Err(error) => {
                    let end = start + error.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[start..end]).unwrap());
                    start = end;
                    if let Some(n) = error.error_len() {
                        out.push('\u{fffd}');
                        start += n;
                    } else if finish {
                        out.push('\u{fffd}');
                        start = self.pending.len();
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending.drain(..start);
        out
    }
}
#[derive(Default)]
struct Lines {
    decoder: Decoder,
    line: String,
    skip_lf: bool,
}
impl Lines {
    fn feed(&mut self, bytes: &[u8], finish: bool) -> Vec<String> {
        let text = self.decoder.feed(bytes, finish);
        let mut out = Vec::new();
        for ch in text.chars() {
            if self.skip_lf {
                self.skip_lf = false;
                if ch == '\n' {
                    continue;
                }
            }
            if ch == '\r' || ch == '\n' {
                out.push(std::mem::take(&mut self.line));
                self.skip_lf = ch == '\r';
            } else {
                self.line.push(ch);
            }
        }
        if finish && !self.line.is_empty() {
            out.push(std::mem::take(&mut self.line));
        }
        out
    }
}
async fn read(
    stream: &Option<Arc<dyn ReadableStream>>,
) -> Result<Option<Vec<u8>>, SubprocessError> {
    match stream {
        Some(s) => s.read_chunk().await,
        None => Ok(None),
    }
}
pub(super) struct EngineOutcome {
    pub code: Option<i32>,
    pub stderr: String,
    pub killed_for_limit: bool,
    pub aborted: bool,
}

pub(super) async fn run(
    process: Arc<dyn Process>,
    signal: Option<Arc<dyn ToolExecutionSignal>>,
    window: Arc<Window>,
    listen_to_preexisting_abort: bool,
    on_line: impl FnMut(String) -> bool,
) -> Result<EngineOutcome, ToolCapabilityError> {
    run_with_completion(
        process,
        signal,
        window,
        listen_to_preexisting_abort,
        on_line,
        |_| true,
    )
    .await
}

/// Decide synchronously at exit + both EOFs whether this run's outcome stands.
/// A diagnostic retry retains the same live window; it is never closed/reopened.
pub(super) async fn run_with_completion(
    process: Arc<dyn Process>,
    signal: Option<Arc<dyn ToolExecutionSignal>>,
    window: Arc<Window>,
    listen_to_preexisting_abort: bool,
    mut on_line: impl FnMut(String) -> bool,
    mut outcome_stands: impl FnMut(&EngineOutcome) -> bool,
) -> Result<EngineOutcome, ToolCapabilityError> {
    let stdout = process.stdout();
    let stderr = process.stderr();
    let mut lines = Lines::default();
    let mut err_decoder = Decoder::default();
    let mut err = String::new();
    let mut out_eof = false;
    let mut err_eof = false;
    let mut exit = None;
    // grep registers after spawn: an already-fired signal is not a new event.
    let preexisting = signal.as_ref().is_some_and(|s| s.is_cancelled());
    let mut stop = None;
    let mut killed_for_limit = false;
    let result=async {
        loop {
            // Re-read the signal after each awaited exit/read operation, including
            // the operation which completes the last prerequisite. Cleanup and
            // diagnostic reruns retain it when their first outcome does not stand.
            if window.active.load(Ordering::SeqCst) && (!preexisting||listen_to_preexisting_abort) && signal.as_ref().is_some_and(|s|s.is_cancelled()) {
                window.aborted.store(true,Ordering::SeqCst);
                if stop.is_none() {let p=process.clone();stop=Some(tokio::spawn(async move {p.terminate().await;}));}
            }
            if out_eof && err_eof && let Some(code) = exit {
                // Settle the abort window BEFORE cleanup or a pending terminate acknowledgement.
                let outcome=EngineOutcome{code,stderr:err,killed_for_limit,aborted:window.aborted.load(Ordering::SeqCst)};
                if outcome_stands(&outcome) { window.complete(); }
                return Ok(outcome);
            }
            tokio::select! { biased;
                status=process.wait(), if exit.is_none()=>{exit=Some(status.map_err(|e|ToolCapabilityError::new(e.to_string()))?.exit_code);},
                chunk=read(&stdout), if !out_eof=>{
                    let chunk=chunk.map_err(|e|ToolCapabilityError::new(e.to_string()))?;
                    for line in lines.feed(chunk.as_deref().unwrap_or_default(),chunk.is_none()) {
                        if on_line(line)&&stop.is_none() {killed_for_limit=true;let p=process.clone();stop=Some(tokio::spawn(async move {p.terminate().await;}));}
                    }
                    out_eof=chunk.is_none();
                },
                chunk=read(&stderr), if !err_eof=>{let chunk=chunk.map_err(|e|ToolCapabilityError::new(e.to_string()))?;err.push_str(&err_decoder.feed(chunk.as_deref().unwrap_or_default(),chunk.is_none()));err_eof=chunk.is_none();},
                ()=tokio::time::sleep(Duration::from_millis(1))=>{},
            }
        }
    }.await;
    if result.is_err() {
        window.complete();
    }
    if result.is_err() && stop.is_none() {
        let p = process.clone();
        stop = Some(tokio::spawn(async move {
            p.terminate().await;
        }));
    }
    if let Some(stop) = stop {
        let _ = stop.await;
    }
    if let Some(s) = stdout {
        s.close().await;
    }
    if let Some(s) = stderr {
        s.close().await;
    }
    result
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::execution::{ExitStatus, WritableStream};
    use async_trait::async_trait;
    use tokio::sync::{Mutex, Notify};
    pub(crate) struct Signal(AtomicBool);
    impl Signal {
        pub(crate) fn abort(&self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    impl ToolExecutionSignal for Signal {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }
    struct Stream {
        chunks: Mutex<std::collections::VecDeque<Vec<u8>>>,
        eof: AtomicBool,
        point: &'static str,
        abort_at: &'static str,
        signal: Arc<Signal>,
    }
    impl Stream {
        fn fire(&self, point: &str) {
            if point == self.abort_at {
                self.signal.0.store(true, Ordering::SeqCst);
            }
        }
    }
    #[async_trait]
    impl ReadableStream for Stream {
        async fn read_chunk(&self) -> Result<Option<Vec<u8>>, SubprocessError> {
            let chunk = self.chunks.lock().await.pop_front();
            if chunk.is_none() {
                self.eof.store(true, Ordering::SeqCst);
                self.fire(self.point);
            } else if self.point == "stdout_eof" {
                self.fire("stdout_data");
            }
            Ok(chunk)
        }
        async fn close(&self) {
            self.fire(if self.point == "stdout_eof" {
                "stdout_close"
            } else {
                "stderr_close"
            });
        }
    }
    struct Child {
        out: Arc<Stream>,
        err: Arc<Stream>,
        stopped: Notify,
        release: Notify,
        hold_stop: bool,
        wait_first: bool,
    }
    #[async_trait]
    impl Process for Child {
        fn pid(&self) -> u32 {
            1
        }
        fn stdin(&self) -> Option<Arc<dyn WritableStream>> {
            None
        }
        fn stdout(&self) -> Option<Arc<dyn ReadableStream>> {
            Some(self.out.clone())
        }
        fn stderr(&self) -> Option<Arc<dyn ReadableStream>> {
            Some(self.err.clone())
        }
        async fn wait(&self) -> Result<ExitStatus, SubprocessError> {
            while !self.wait_first
                && (!self.out.eof.load(Ordering::SeqCst) || !self.err.eof.load(Ordering::SeqCst))
            {
                tokio::task::yield_now().await;
            }
            self.out.fire("wait");
            Ok(ExitStatus { exit_code: Some(0) })
        }
        async fn terminate(&self) {
            self.stopped.notify_one();
            if self.hold_stop {
                self.release.notified().await;
            }
        }
    }
    fn child(point: &'static str, signal: Arc<Signal>, hold_stop: bool) -> Arc<Child> {
        Arc::new(Child {
            out: Arc::new(Stream {
                chunks: Mutex::new([b"a.ts\n".to_vec()].into()),
                eof: AtomicBool::new(false),
                point: "stdout_eof",
                abort_at: point,
                signal: signal.clone(),
            }),
            err: Arc::new(Stream {
                chunks: Mutex::new(Default::default()),
                eof: AtomicBool::new(false),
                point: "stderr_eof",
                abort_at: point,
                signal,
            }),
            stopped: Notify::new(),
            release: Notify::new(),
            hold_stop,
            wait_first: false,
        })
    }
    pub(crate) fn partition_fixture(
        point: &'static str,
        output: Vec<u8>,
    ) -> (Arc<dyn Process>, Arc<Signal>) {
        let signal = Arc::new(Signal(AtomicBool::new(false)));
        let process = child(point, signal.clone(), false);
        *process.out.chunks.try_lock().unwrap() = [output].into();
        (process, signal)
    }
    #[tokio::test]
    async fn completion_requires_exit_and_both_eofs() {
        let signal = Arc::new(Signal(AtomicBool::new(false)));
        let mut process = child("", signal.clone(), false);
        Arc::get_mut(&mut process).unwrap().wait_first = true;
        let mut lines = Vec::new();
        run(
            process.clone(),
            Some(signal),
            Window::new(),
            false,
            |line| {
                lines.push(line);
                false
            },
        )
        .await
        .unwrap();
        assert_eq!(lines, ["a.ts"]);
        assert!(process.out.eof.load(Ordering::SeqCst));
        assert!(process.err.eof.load(Ordering::SeqCst));
    }
    #[tokio::test]
    async fn every_exit_and_eof_abort_is_inside_the_window_but_close_is_outside() {
        for find in [true, false] {
            for point in [
                "spawn",
                "stdout_data",
                "stdout_eof",
                "stderr_eof",
                "wait",
                "stdout_close",
                "stderr_close",
            ] {
                let signal = Arc::new(Signal(AtomicBool::new(point == "spawn")));
                let process = child(point, signal.clone(), false);
                let mut lines = Vec::new();
                let result = run(process, Some(signal), Window::new(), find, |line| {
                    lines.push(line);
                    false
                })
                .await
                .unwrap();
                assert_eq!(
                    result.aborted,
                    !point.ends_with("close") && (find || point != "spawn"),
                    "{} {point}",
                    if find { "find" } else { "grep" }
                );
                assert_eq!(lines, ["a.ts"]);
            }
        }
    }
    #[tokio::test]
    async fn held_limit_stop_acknowledgement_does_not_extend_completion() {
        for mode in 0..3 {
            let signal = Arc::new(Signal(AtomicBool::new(false)));
            let process = child(
                if mode == 2 { "stdout_close" } else { "" },
                signal.clone(),
                true,
            );
            let window = Window::new();
            let task = tokio::spawn({
                let p = process.clone();
                let s = signal.clone();
                let w = window.clone();
                async move { run(p, Some(s), w, false, |_| true).await.unwrap() }
            });
            process.stopped.notified().await;
            // A scheduling yield observes the explicit completion flag, not elapsed time.
            tokio::time::timeout(Duration::from_secs(5), async {
                while window.active.load(Ordering::SeqCst) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            if mode == 1 {
                signal.0.store(true, Ordering::SeqCst);
            }
            assert!(!task.is_finished(), "the acknowledgement is actually held");
            process.release.notify_one();
            let result = task.await.unwrap();
            assert!(!result.aborted);
            assert!(result.killed_for_limit);
            assert_eq!(signal.is_cancelled(), mode != 0);
        }
    }
    #[test]
    fn readline_handles_cr_and_utf8_chunks_without_stripping_bom() {
        let mut lines = Lines::default();
        assert_eq!(lines.feed(b"\xef\xbb\xbfA\r", false), vec!["\u{feff}A"]);
        assert!(lines.feed(b"\n\xf0\x9f", false).is_empty());
        assert_eq!(lines.feed(b"\x98\x80\rB\nC", true), vec!["😀", "B", "C"]);
    }
}
