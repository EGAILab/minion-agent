"""L12-D006 intended-witness controls over a disposable copy, never the candidate.

The caller supplies the pinned offline Cargo/ICU environment and E-local caches.
Every intended witness must first pass. Compilation/setup errors and unrelated
test failures are INVALID, not kills. Sources and compiled baseline are restored.
"""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--logs", type=Path, required=True)
args = parser.parse_args()
tree = args.tree.resolve()
if "control" not in str(tree):
    raise RuntimeError("refusing a non-control scratch tree")
source = tree / "crates/minion-agent/src/execution/filesystem.rs"
original = source.read_text(encoding="utf-8")
controls = [
    ("global-invalid-input-remap", "        io::ErrorKind::InvalidInput | io::ErrorKind::InvalidData => FsErrorCode::Invalid,", "        io::ErrorKind::InvalidInput => FsErrorCode::Unknown,\n        io::ErrorKind::InvalidData => FsErrorCode::Invalid,", "--lib", "execution::filesystem::tests::l12d006_other_invalid_input_keeps_invalid", "L12-D006 preserves unrelated InvalidInput"),
    ("nul-stays-invalid", "        FsError::new(FsErrorCode::Unknown, error.to_string())", "        FsError::new(FsErrorCode::Invalid, error.to_string())", "execution_nul", "nul_is_unknown_and_the_fallback_is_lossless", "L12-D006 NUL is unknown"),
    ("native-projected-fallback", "        || (error.code == FsErrorCode::Unknown && os.to_string_lossy().contains('\\0'))", "        || false /* mutant: no logical NUL fallback */", "execution_nul", "nul_is_unknown_and_the_fallback_is_lossless", "L12-D006 fallback stays logical"),
    ("premature-write-validation", "        abortable_io(signal, &os, tokio::fs::write(&os, content))", "        abortable_io(signal, &os, tokio::fs::write(&os, content))", "execution_nul", "a_final_nul_keeps_the_parent_creation_effect", "L12-D006 parent exists before NUL leaf rejection"),
    ("raw-argument-only-remap", "        abortable_io(signal, &os, tokio::fs::read(&os))\n            .await\n            .map_err(|e| io_origin(e, &logical, &os, true))", "        abortable_io(signal, &os, tokio::fs::read(&os))\n            .await\n            .map_err(|e| {\n                let mut e = io_origin(e, &logical, &os, true);\n                if e.code == FsErrorCode::Unknown && !path.code_units().contains(&0) {\n                    e.code = FsErrorCode::Invalid;\n                }\n                e\n            })", "execution_nul", "the_resolved_file_url_is_the_nul_argument", "L12-D006 decoded URL is unknown"),
    ("canonical-walk-before-check", "        if logical.code_units().contains(&0) {", "        if logical.code_units().contains(&0) {\n            if let Some(parent) = native(&logical).parent() {\n                tokio::fs::metadata(parent).await.map_err(|e| call_error(e, parent, &logical))?;\n            }", "execution_nul", "canonical_path_checks_the_whole_argument_before_a_missing_prefix", "L12-D006 whole argument precedes walk"),
]
# The premature-validation mutation changes the write operation before mkdirp,
# not a generic helper (the latter would retain the required parent side effect).
write_start = "    async fn write_file(\n        &self,\n        path: &FsPath,\n        content: &[u8],\n        signal: Option<&dyn AbortSignal>,\n    ) -> Result<(), FsError> {\n        let logical = self.resolved(path);"
early = write_start + "\n        if logical.code_units().contains(&0) {\n            return Err(FsError::new(FsErrorCode::Unknown, \"premature NUL rejection\").with_path(&logical));\n        }"
controls[3] = (controls[3][0], write_start, early, *controls[3][3:])
args.logs.mkdir(parents=True, exist_ok=True)


def run(label, target, witness, mutant, signature):
    selection = ["--lib"] if target == "--lib" else ["--test", target]
    result = subprocess.run(
        ["cargo", "test", "--offline", "-p", "minion-agent", "--all-features", *selection, witness, "--", "--exact", "--nocapture"],
        cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (args.logs / (label + ".log")).write_text(result.stdout, encoding="utf-8")
    if mutant:
        valid = (result.returncode == 101 and f"test {witness} ... FAILED" in result.stdout
                 and "1 failed" in result.stdout and "panicked at" in result.stdout
                 and signature in result.stdout and "could not compile" not in result.stdout)
    else:
        valid = result.returncode == 0 and f"test {witness} ... ok" in result.stdout and "1 passed" in result.stdout
    if not valid:
        raise RuntimeError(f"INVALID {label}: exit {result.returncode}; see log")


try:
    for name, _, _, target, witness, signature in controls:
        run(name + "-baseline", target, witness, False, signature)
    print(f"BASELINE: all {len(controls)} intended witnesses selected and passing", flush=True)
    for name, old, new, target, witness, signature in controls:
        if original.count(old) != 1 or old == new:
            raise RuntimeError(f"INVALID anchor {name}")
        source.write_text(original.replace(old, new, 1), encoding="utf-8")
        run(name, target, witness, True, signature)
        print(f"KILLED {name} by {witness}: {signature}", flush=True)
        source.write_text(original, encoding="utf-8")
finally:
    source.write_text(original, encoding="utf-8")
    for name, _, _, target, witness, signature in controls:
        run(name + "-restored", target, witness, False, signature)
    print("RESTORED: every intended witness selected and passing", flush=True)
