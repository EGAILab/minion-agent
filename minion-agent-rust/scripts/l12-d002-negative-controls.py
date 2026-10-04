"""EXEC-012 wrong-implementation controls, in disposable workspace copies.

Inherit the same pinned ICU/Cargo environment as the positive gates. Run serially
with other Cargo commands sharing the target directory: package rebuilding
replaces test executables. The script never edits the candidate. A compiler/setup
error is NOT a killed mutant. Rebuild the positive candidate after the controls.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def replace(source, old, new):
    if source.count(old) != 1:
        raise AssertionError(f"expected unique mutation anchor: {old!r}")
    return source.replace(old, new)


WAIT = """            if let Some(outcome) = receiver.borrow().clone() {
                return outcome;
            }"""
CLOSE = """        self.closed.send_replace(true);
        // This never waits for a read future's serialization permit. Taking the
        // reader drops exactly this OS handle, even with retained public Arcs.
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();"""
PENDING = "close_wakes_a_pending_read_and_does_not_close_the_sibling_or_kill"
EXIT = "exit_is_independent_of_inherited_pipes_and_late_output_is_readable"


def mutate(source, name):
    if name in ("wait-joins-pipes", "wait-closes-pipes"):
        action = (
            "while stdout.read_chunk().await?.is_some() {}"
            if name == "wait-joins-pipes"
            else "stdout.close().await;"
        )
        source = replace(source, WAIT, """            let snapshot = receiver.borrow().clone();
            if let Some(outcome) = snapshot {
                if let Some(stdout) = &self.stdout {
                    """ + action + """
                }
                return outcome;
            }""")
        return source, EXIT
    if name == "close-deadlocks":
        return replace(source, CLOSE, """        let _permit = self.read_lock.lock().await;
        self.closed.send_replace(true);
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).take();"""), PENDING
    if name == "close-keeps-handle":
        return replace(source, CLOSE, "        self.closed.send_replace(true);"), "closing_the_read_end_preserves_a_descendant_and_releases_the_pipe"
    if name == "closed-read-errors":
        return replace(source, """        if *closed.borrow() {
            return Ok(None);
        }""", """        if *closed.borrow() {
            return Err(SubprocessError::new(SubprocessErrorCode::PipeError, "closed"));
        }"""), PENDING
    if name == "close-affects-sibling":
        source = replace(source, "struct LocalReadableStream {", """static MUTANT_STREAMS: std::sync::OnceLock<std::sync::Mutex<Vec<watch::Sender<bool>>>> = std::sync::OnceLock::new();
struct LocalReadableStream {""")
        source = replace(source, "        let (closed, _) = watch::channel(false);", """        let (closed, _) = watch::channel(false);
        MUTANT_STREAMS.get_or_init(Default::default).lock().unwrap().push(closed.clone());""")
        source = replace(source, CLOSE, """        for sender in MUTANT_STREAMS.get_or_init(Default::default).lock().unwrap().iter() {
            sender.send_replace(true);
        }
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).take();""")
        return source, PENDING
    if name == "close-kills-process":
        source = replace(source, "struct LocalReadableStream {", """static MUTANT_PID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
struct LocalReadableStream {""")
        source = replace(source, "        let stdin = child.stdin.take().map(LocalWritableStream::new);", """        MUTANT_PID.store(child.id().unwrap(), Ordering::Release);
        let stdin = child.stdin.take().map(LocalWritableStream::new);""")
        return replace(source, CLOSE, CLOSE + """
        #[cfg(windows)]
        kill_process_tree(MUTANT_PID.load(Ordering::Acquire)).await;
        #[cfg(unix)]
        let _ = Command::new("kill").args(["-KILL", &MUTANT_PID.load(Ordering::Acquire).to_string()]).status().await;
"""), PENDING
    if name == "cancelled-waiter-poisons":
        source = replace(source, """        let mut receiver = self.outcome.clone();""", """        struct Poison<'a>(&'a LocalProcess);
        impl Drop for Poison<'_> {
            fn drop(&mut self) {
                self.0.cause.store(CAUSE_SIGNAL, Ordering::Release);
                self.0.terminate.notify_one();
            }
        }
        let poison = Poison(self);
        let mut receiver = self.outcome.clone();""")
        source = replace(source, "                return outcome;", "                std::mem::forget(poison);\n                return outcome;")
        return source, "cancellation_of_a_waiter_does_not_poison_later_waits"
    if name == "settled-terminate-keeps-handles":
        return replace(source, "        if settled {", "        if settled && false {"), "terminate_after_settlement_releases_even_retained_stream_handles"
    raise AssertionError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    relative = Path("crates/minion-agent/src/execution/subprocess.rs")
    original = (workspace / relative).read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    controls = ["wait-joins-pipes", "wait-closes-pipes", "close-deadlocks",
                "close-keeps-handle", "closed-read-errors", "close-affects-sibling",
                "close-kills-process", "cancelled-waiter-poisons",
                "settled-terminate-keeps-handles"]
    results = []
    with tempfile.TemporaryDirectory(prefix="minion-exec012-") as directory:
        scratch = Path(directory) / "rust"
        shutil.copytree(workspace, scratch, ignore=shutil.ignore_patterns("target", ".git"))
        for name in controls:
            changed, test = mutate(original, name)
            (scratch / relative).write_text(changed, encoding="utf-8")
            subprocess.run(["cargo", "clean", "-p", "minion-agent"], cwd=scratch,
                           check=True, capture_output=True, text=True)
            run = subprocess.run(["cargo", "test", "--locked", "--offline", "-p", "minion-agent", "--all-features",
                                  "--test", "execution_stream_close", test],
                                 cwd=scratch, capture_output=True, text=True, timeout=180)
            log = run.stdout + run.stderr
            (args.out / f"{name}.log").write_text(log, encoding="utf-8")
            killed = run.returncode != 0 and "test result: FAILED. 0 passed; 1 failed;" in log
            results.append({"control": name, "witness": test, "killed": killed})
            print(f"{name}: {'KILLED' if killed else 'INVALID / SURVIVED'}", flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if not all(row["killed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
