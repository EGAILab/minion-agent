"""Sequential Windows evidence; source controls run only in a disposable copy.

Dot-source .agents/project-environment.ps1 and pass a NEW contained --logs path
and a --control-tree ending in control-code/minion-agent-rust. Both Cargo caches
use separate per-platform gate/control targets. No process is killed or target cleaned.
"""
import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys
import importlib.util

p = argparse.ArgumentParser()
p.add_argument("--logs", type=Path, required=True)
p.add_argument("--control-tree", type=Path, required=True)
a = p.parse_args()
if os.name != "nt":
    p.error("Windows launcher only")
root = Path("E:/AI/Projects/OpenMinds/Minions/Minion-Agent").resolve(strict=True)
tree = Path(__file__).resolve().parents[1]

def checked(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root) or path == root:
        raise RuntimeError(f"outside gate sandbox: {path}")
    for entry in [*reversed(path.parents), path]:
        try:
            metadata = entry.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 1024:
            raise RuntimeError(f"link/reparse in gate target: {entry}")
    if not path.resolve(strict=False).is_relative_to(root):
        raise RuntimeError(f"outside real gate target: {path}")
    return path

checked(tree)
control_tree = checked(a.control_tree)
if "control-code" not in control_tree.parts:
    raise RuntimeError("disposable control-code tree required")
logs = checked(a.logs)
checked(logs).mkdir()  # existing log directory is an error; never overwrite a run
os.environ["RUST_ICU_MAJOR_VERSION_NUMBER"] = "78"
os.environ["RUSTFLAGS"] = "-L native=" + str(root / ".toolchain/icu-78.3-src/icu/lib64").replace("\\", "/")
os.environ["RUSTDOCFLAGS"] = "-D warnings " + os.environ["RUSTFLAGS"]
os.environ["MINION_AGENT_ICU_IDENTITY"] = str(root / ".toolchain/icu-78.3-src/pinned-icu-identity.txt")
os.environ["MINION_SEARCH_ENGINE_ARTIFACTS"] = str(root / ".toolchain/search-engines/dl")
os.environ["PATH"] = str(root / ".toolchain/icu-78.3-src/icu/bin64") + os.pathsep + os.environ["PATH"]
if os.environ.get("CARGO_BUILD_JOBS") != "4":
    raise RuntimeError("source project-environment.ps1 first (four Cargo jobs)")
gate_target = checked(root / '.tmp/cache/cargo-target/windows')
os.environ['CARGO_TARGET_DIR'] = str(gate_target)
os.environ['MINION_CONTROL_CARGO_TARGET'] = str(root / '.tmp/cache/cargo-target/windows-controls')
spec = importlib.util.spec_from_file_location('cargo_cache_safety', tree / 'scripts/cargo-cache-safety.py')
cache_safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_safety)
cache_safety.refresh(tree)  # content-neutral: defeat foreign/mutant mtime freshness

commands = [
    ("cache-safety-check", [sys.executable, "scripts/cargo-cache-safety-check.py"]),
    ("error-codes-anchor-check", [sys.executable, "scripts/error-codes-negative-controls.py", "--tree", str(tree), "--check-anchors"]),
    ("native-fs-check", ["cargo", "check", "-p", "minion-agent-native-fs"]),
    ("fmt", ["cargo", "fmt", "--all", "--", "--check"]),
    ("clippy", ["cargo", "clippy", "--workspace", "--all-targets", "--all-features", "--", "-D", "warnings"]),
    ("test", ["cargo", "test", "--workspace", "--all-features"]),
    ("doc", ["cargo", "doc", "--workspace", "--no-deps"]),
    ("xtask", ["cargo", "run", "-p", "xtask", "--", "conformance", "verify"]),
    ("rimraf", ["cargo", "test", "-p", "minion-agent", "--lib", "execution::filesystem::rimraf::tests", "--", "--nocapture"]),
    ("error-codes-controls", [sys.executable, "scripts/error-codes-negative-controls.py", "--tree", str(control_tree), "--logs", str(logs / "error-codes-control-details")]),
    ("div008-controls", [sys.executable, "scripts/div008-negative-controls.py", "--tree", str(control_tree), "--logs", str(logs / "div008-control-details")]),
    ("cache-regression", [sys.executable, "scripts/cargo-cache-regression.py", "--tree", str(tree), "--control-tree", str(control_tree), "--logs", str(logs / 'cache-regression')]),
]
for name, command in commands:
    with checked(logs / (name + ".log")).open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=tree, stdout=output, stderr=subprocess.STDOUT)
    checked(logs / (name + ".exit")).write_text(str(result.returncode) + "\n", encoding="ascii")
    if result.returncode:
        raise RuntimeError(f"{name} exit {result.returncode}; inspect the log, never credit or blindly rerun")
    print(name + " PASS (native exit 0)", flush=True)
print("L12-D008 WINDOWS G3 AND BOTH CONTROL BATCHES PASS", flush=True)
