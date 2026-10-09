"""L12-D005 intended-witness controls; only an explicitly disposable tree is mutated.

Run in the pinned Cargo/ICU environment: --tree <scratch-code>/minion-agent-rust
--logs <project scratch>. The candidate is never edited. Compiler/setup failures
are INVALID. Each intended baseline must be selected and pass before any mutant.
"""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--logs", type=Path, required=True)
args = parser.parse_args()
tree = args.tree.resolve()
if "control" not in str(tree):
    raise RuntimeError("refusing a non-control scratch tree")
source = tree / "crates/minion-agent/src/execution/filesystem.rs"
native = tree / "crates/minion-agent-native-fs/src/lib.rs"
originals = {p: p.read_text(encoding="utf-8") for p in [source, native]}
prefix = "execution::filesystem::readonly_tests::"
controls = [
    ("no-directory-correction", source, "match operations.clear_readonly_entry(path).await {", "match Ok::<bool, io::Error>(false) {", "readonly_directory_retry_preserves_the_distinct_retry_error_and_origin", "NotDirectory"),
    ("retry-keeps-first-error", source, "                    retry => retry,", "                    Err(_) => Err(original),\n                    Ok(()) => Ok(()),", "readonly_directory_retry_preserves_the_distinct_retry_error_and_origin", "NotDirectory"),
    ("correction-error-replaces-original", source, "                _ => Err(original),", "                Err(correction) => Err(correction),\n                Ok(false) => Err(original),", "readonly_directory_correction_failure_preserves_original_without_retry", "first delete"),
    ("vanished-correction-is-error", source, "Err(correction) if correction.kind() == io::ErrorKind::NotFound => Ok(())", "Err(correction) if correction.kind() == io::ErrorKind::NotFound => Err(original)", "readonly_directory_concurrent_vanish_during_correction_or_retry_is_success", "unwrap"),
    ("vanished-retry-keeps-original", source, "Err(retry) if retry.kind() == io::ErrorKind::NotFound => Ok(())", "Err(retry) if retry.kind() == io::ErrorKind::NotFound => Err(original)", "readonly_directory_concurrent_vanish_during_correction_or_retry_is_success", "unwrap"),
    ("retry-twice", source, "                    retry => retry,", "                    Err(_) => operations.remove_dir(path).await,\n                    Ok(()) => Ok(()),", "readonly_directory_retries_only_once", "left: 3"),
]
if sys.platform == "win32":
    controls += [
        ("restore-directory-failure", source, "match remove_directory_with(path, operations).await {", "match operations.remove_dir(path).await {", "readonly_directory_real_remove_handles_target_and_tree", "unwrap"),
        ("follow-reparse-target", native, "FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)", "FILE_FLAG_BACKUP_SEMANTICS | (FILE_FLAG_OPEN_REPARSE_POINT & 0))", "readonly_correction_addresses_the_link_not_its_external_target", "assertion failed"),
    ]
args.logs.mkdir(parents=True, exist_ok=True)

def run(label, witness, mutant, signature):
    witness = prefix + witness
    result = subprocess.run(["cargo", "test", "-p", "minion-agent", "--all-features", "--lib", witness, "--", "--exact", "--nocapture"], cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.logs / (label + ".log")).write_text(result.stdout, encoding="utf-8")
    if not mutant:
        valid = result.returncode == 0 and f"test {witness} ... ok" in result.stdout and "1 passed" in result.stdout
    else:
        valid = result.returncode == 101 and f"test {witness} ... FAILED" in result.stdout and "1 failed" in result.stdout and "panicked at" in result.stdout and signature in result.stdout
    if not valid:
        raise RuntimeError(f"INVALID {label}: exit {result.returncode}; see log")

try:
    for name, _, _, _, witness, signature in controls:
        run(name + "-baseline", witness, False, signature)
    print(f"BASELINE: all {len(controls)} intended witnesses selected and passing", flush=True)
    for name, path, old, new, witness, signature in controls:
        for p, content in originals.items():
            p.write_text(content, encoding="utf-8")
        if originals[path].count(old) != 1:
            raise RuntimeError(f"INVALID anchor {name}")
        path.write_text(originals[path].replace(old, new, 1), encoding="utf-8")
        run(name, witness, True, signature)
        print(f"KILLED {name} by {prefix + witness}", flush=True)
finally:
    for p, content in originals.items():
        p.write_text(content, encoding="utf-8")
    # Worktrees share the required single Cargo target. A restored file in one
    # copy can be older than another copy's last compiled mutant: source restore
    # alone is not sufficient. Invalidate the tiny native package and prove a
    # rebuilt, restored baseline before another worktree consumes that target.
    clean = subprocess.run(["cargo", "clean", "-p", "minion-agent-native-fs"], cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.logs / "restore-clean.log").write_text(clean.stdout, encoding="utf-8")
    if clean.returncode:
        raise RuntimeError("INVALID restoration: native package clean failed")
    restored = subprocess.run(["cargo", "test", "-p", "minion-agent", "--all-features", "--lib", "readonly_", "--", "--nocapture"], cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.logs / "restored-baseline.log").write_text(restored.stdout, encoding="utf-8")
    if restored.returncode or any(f"test {prefix + witness} ... ok" not in restored.stdout for _, _, _, _, witness, _ in controls):
        raise RuntimeError("INVALID restoration: every intended witness must pass again")
    print("RESTORED: sources and compiled baseline verified green", flush=True)
