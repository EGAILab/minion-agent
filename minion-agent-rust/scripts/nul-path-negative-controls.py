"""L12-D006 intended-witness controls over a disposable copy, never the candidate.

The caller supplies the pinned offline Cargo/ICU environment and E-local caches.
Every intended witness must first pass. Compilation/setup errors and unrelated
test failures are INVALID, not kills. Sources and compiled baseline are restored.
"""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--logs", type=Path, required=True)
parser.add_argument("--project-root", type=Path, required=True)
parser.add_argument("--container-sandbox", type=Path)
args = parser.parse_args()
project = args.project_root.absolute()
expected_project = Path("E:/AI/Projects/OpenMinds/Minions/Minion-Agent") if os.name == "nt" else Path("/project")
if project != expected_project:
    raise RuntimeError("unexpected project root; refusing all control writes")
if project.resolve(strict=True) != project:
    raise RuntimeError("project root must not traverse a link")
private = args.container_sandbox
if private is not None:
    if os.name == "nt" or private != Path("/tmp/l12d006-session") or private.resolve(strict=True) != private:
        raise RuntimeError("unexpected container sandbox")


def checked(path, sandbox=None):
    """Check lexical containment first, then every existing link in the path.

    Absolute CLI paths select sandboxes; mutation targets are fixed relative
    descendants, never arbitrary CLI names or corpus strings.
    """
    lexical = Path(os.path.abspath(path))
    if ".." in Path(path).parts:
        raise RuntimeError(f"refusing parent traversal: {path}")
    boundary = (private if private is not None and lexical.is_relative_to(private) else project) if sandbox is None else sandbox
    if not lexical.is_relative_to(boundary) or lexical == boundary:
        raise RuntimeError(f"outside project: {path}")
    real = lexical.resolve(strict=False)
    if not real.is_relative_to(boundary.resolve(strict=True)) or real == boundary.resolve(strict=True):
        raise RuntimeError(f"link escapes project: {path}")
    return lexical


tree = checked(args.tree)
if "control-code" not in tree.parts:
    raise RuntimeError("refusing a non-control scratch tree")
logs = checked(args.logs)
source = tree / "crates/minion-agent/src/execution/filesystem.rs"
if tree.resolve(strict=True) != tree:
    raise RuntimeError("control tree must not traverse a link")
original = checked(source, tree).read_text(encoding="utf-8")
controls = [
    ("global-invalid-input-remap", "        io::ErrorKind::InvalidInput | io::ErrorKind::InvalidData => FsErrorCode::Invalid,", "        io::ErrorKind::InvalidInput => FsErrorCode::Unknown,\n        io::ErrorKind::InvalidData => FsErrorCode::Invalid,", "--lib", "execution::filesystem::tests::l12d006_other_invalid_input_keeps_invalid", "L12-D006 preserves unrelated InvalidInput"),
    ("nul-stays-invalid", "fn map_path_error(error: io::Error, path: &Path) -> FsError {\n    if nul_binding_error(&error, path) {\n        FsError::new(FsErrorCode::Unknown, error.to_string())", "fn map_path_error(error: io::Error, path: &Path) -> FsError {\n    if nul_binding_error(&error, path) {\n        FsError::new(FsErrorCode::Invalid, error.to_string())", "execution_nul", "nul_is_unknown_and_the_fallback_is_lossless", "L12-D006 NUL is unknown"),
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
checked(logs).mkdir(parents=True, exist_ok=True)


def run(label, target, witness, mutant, signature):
    checked(tree)
    for name in ("TMP", "TEMP", "TMPDIR", "CARGO_HOME", "CARGO_TARGET_DIR"):
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"missing contained {name}; refusing Cargo")
        checked(Path(value))
    selection = ["--lib"] if target == "--lib" else ["--test", target]
    result = subprocess.run(
        ["cargo", "test", "--offline", "-p", "minion-agent", "--all-features", *selection, witness, "--", "--exact", "--nocapture"],
        cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    checked(logs / (label + ".log"), logs).write_text(result.stdout, encoding="utf-8")
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
        checked(source, tree).write_text(original.replace(old, new, 1), encoding="utf-8")
        run(name, target, witness, True, signature)
        print(f"KILLED {name} by {witness}: {signature}", flush=True)
        checked(source, tree).write_text(original, encoding="utf-8")
finally:
    checked(source, tree).write_text(original, encoding="utf-8")
    for name, _, _, target, witness, signature in controls:
        run(name + "-restored", target, witness, False, signature)
    print("RESTORED: every intended witness selected and passing", flush=True)
