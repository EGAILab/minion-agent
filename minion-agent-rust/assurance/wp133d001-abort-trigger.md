# WP133-D001 Rust test-fixture record

Status: Rust candidate ready for independent review. Not certified; merge waits
for Claude's exact-triple review of #218, #295 and #220.
Coordination: minion-agent#217; provenance #197.

## Exact shared inputs and scope

Shared/Python code #218 @ 2266c4b6b53991dbeded96bf7e5f33d9b76e4371,
docs #295 @ e47338e8589a42298dbb4e3d9ddd46aa4af1f3bd, independently
approved in #217 issuecomment-6130548185, remain unmerged until both runners
support the fixture. This Rust branch starts at that exact code head; inherited
Python/schema/scenario changes are byte-unchanged. Review the Rust-only delta
against 2266c4b6. The later code SHA is supplied by the handoff, not self-pinned
inside its own commit.

Authority: pinned Pi b7bb00b936dbe21b8e160b3e89efdec361846699 / Node22.15.1,
approved WP-13.3 contract and the #217 test-only shared change. The expected
`before\n\n\nCommand aborted` is unchanged. Owner Q2: zero partial tool updates.
Python is secondary evidence, not the Rust oracle. This changes no bash product,
execution capability, filesystem mapper, public API, semantic requirement or
canonical expectation.

## Thin fixture mapping

`tests/builtin_bash_conformance.rs` still runs the real LocalFileSystem,
LocalSubprocess, bash tool and Layer-06 preflight/execution. It consumes the
additive abort_on_output string from the validated canonical document.
`tests/support/bash_abort_on_output.rs` wraps only that call's signal-bearing
command spawn; the certified where/which lookup uses no signal and passes
through unchanged. Within this single-call fixture, this is the existing Rust
seam distinction; it does not invent a new public signal-identity API.

Both stdout and stderr retain the original read/close APIs. The completing
chunk is returned to the tool in the current poll; cancellation is posted to
a later Tokio turn. The marker buffer tracks delivery order, split chunks and
complete UTF-8 decoding without a premature replacement for an incomplete
suffix. A one-shot flag prevents repeated aborts. Process pid/stdin/wait/
terminate delegate to the original process. No update callback is used to
trigger cancellation; the real Layer-06 runner explicitly asserts zero updates.

## Permanent witnesses and controls

- Real Windows-shaped bash shell discovery through controlled providers, on
  every host: lookup output is a bash path containing `before`; the command
  remains successful and its signal is not cancelled. It asserts lookup None
  signal versus command Some signal.
- Split marker: befo then re-newline is delivered before cancellation, exactly
  one abort even after another marker; close and process APIs delegate.
- Stderr completes a marker after stdout, in delivery order.
- Incomplete UTF-8 is not replacement output; a complete malformed byte is.
- Real LocalSubprocess/Layer-06 delayed command start600ms: output-event trigger
  preserves the exact canonical observation. Switching the same document to
  the old400ms timer loses `before`, reproducing the flake deterministically.

`scripts/bash-abort-output-controls.py` resolves all five mutations in memory
with --check-anchors before Cargo or source/log writes. Native batches require
a disposable control-code tree and an explicit dedicated per-platform controls
target different from the gate target. Every intended baseline must be selected
and pass before any mutation. Only exit101, the named test FAILED, a panic and
its specific assertion count as a kill; setup/compile/other failures are INVALID.
Source bytes are restored in finally and the restored baseline is checked.

| Wrong behavior | Intended witness / signature |
| --- | --- |
| Tap every spawn, including lookup | real lookup / lookup must not abort the call |
| Forget earlier chunks | split marker / split output marker must abort |
| Cancel before returning the chunk | delivery order / completing output must be delivered before abort |
| Ignore stderr | cross-stream marker / stderr marker must abort |
| Decode an incomplete UTF8 suffix as replacement | UTF8 / incomplete UTF8 is not replacement output yet |

The old wall-clock trigger is also a permanent runtime negative control in the
slow-start witness, not a source-mutant build in the gate cache.

## Build ownership and evidence

Owner #209 decision issuecomment-6131831805 applies: Codex owns the shared
targets and host queue; Claude's independent reviews use only Claude-private
targets. Four Cargo jobs, memory preflight, fresh workspace-source mtimes before
each gate, dedicated control targets, no cleaning. All host writes and caches
stay inside the project on E:; Linux uses contained E: binds and private tmpfs
/tmp. Claude and Minion-Memory held their batches during the sequential run;
batch-done was sent to both after the Linux launcher finished with exit 0.

Static control anchors: fresh 5/5 resolved; no Cargo, source touch or log creation.
Native witnesses, controls and full Windows/Linux G3: complete as below. Raw
logs and exit codes are in `data/wp133d001-abort-trigger/`, with the SHA-256
ledger `sha256.json` (70 files). These are observations, not self-certification.

Next: exact #218/#295/#220 triple handed to Claude for independent review.

## Native execution disclosure

Windows: focused tap witnesses 4/4 and integration tests 4/4, including the
36 canonical documents. All five controls had selected green baselines, were
killed by their intended assertion, and the restored baseline passed. Full G3:
681 passed, zero failed/ignored; fmt, strict clippy, strict rustdoc and xtask
conformance verify exited 0. The existing LIBCMT linker warning is disclosed.

Linux run 1 is NOT credited as full G3: it stopped with exit 101 at the existing
Layer-06 manifest read, because this new container recipe omitted the manifest
from its private copy. Doc, xtask, uid-1000 replay and controls did not run.
The contained manifest copy was added in d8bbe0c; this changes only the Linux
gate recipe, not the Rust source bytes used for the Windows evidence. Fresh
Linux run 2 completed with launcher exit 0 and complete.exit 0. Full suite:
673 passed, zero failed, one ignored (the existing real POSIX permission witness
requiring an explicit non-root run). Fmt, strict clippy, strict rustdoc and xtask
conformance verify exited 0. Focused tap witnesses 4/4 and integration 4/4,
including 36 canonical documents, passed. The bash integration binary then
passed again as uid 1000: 4/4, including the same 36 documents. This focused
non-root replay does not claim to run the unrelated ignored permission witness.

Linux controls: all five named baselines selected and green; 5/5 intended
assertion kills; source restoration and the restored baseline green. The Linux
and Windows mutation builds used only their dedicated controls targets, never
the shared gate targets. No native batch overlap was observed or credited.

Tools: Rust/Cargo 1.97.1, Node v22.15.1, pinned ICU 78.3. Linux used the pinned
image b1b3c9c0d921d7fa0a6d1f9ec7e4eab87f8c8ec97644c3d791450f131dec813f,
read-only host source and private tmpfs copies. Every workspace-member source
was given a fresh mtime before each gate. The committed reproducible Linux
recipe is `scripts/wp133d001-linux-gates.py`; required bindings are documented
in its header. Windows gate commands (from `minion-agent-rust`, with the
project environment, pinned ICU native link path and Node on PATH) were:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps
cargo run -p xtask -- conformance verify
```

Docs additionally set RUSTDOCFLAGS to `-D warnings` and the pinned ICU native
link path. Focused tests use `cargo test --locked -p minion-agent --all-features`
with `--lib abort_output_fixture_` or `--test builtin_bash_conformance`, followed
by `-- --nocapture`. Controls use the committed Python script with `--tree`
pointing to a fresh contained control-code copy, `--logs` to a contained fresh
directory and `--target-dir` to the dedicated platform controls target.

The final evidence-only commit changes no Rust source or canonical input from
the tested tree d8bbe0c. The Windows Rust source bytes are identical to 1d1d1d9;
the intervening d8bbe0c only adds the contained Linux manifest copy. Shared
inputs remain the exact unmerged approved heads recorded above. No production
code or semantic contract change, no target cleaning and no certification.
