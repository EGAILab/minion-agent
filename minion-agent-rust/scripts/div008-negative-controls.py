"""L12-D008 controls in an explicitly supplied disposable control-code tree.

Compilation/setup errors, absent witnesses and unrelated failures are INVALID.
Each baseline must select and pass its intended named witness before mutation.
"""
import argparse
import os
from pathlib import Path
import stat
import subprocess
import importlib.util

_spec = importlib.util.spec_from_file_location('cargo_cache_safety', Path(__file__).with_name('cargo-cache-safety.py'))
cache_safety = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache_safety)

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--logs", type=Path, required=True)
args = parser.parse_args()
boundary = Path("E:/AI/Projects/OpenMinds/Minions/Minion-Agent") if os.name == "nt" else Path("/tmp/l12d008")
boundary = boundary.resolve(strict=True)

def checked(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(boundary) or path == boundary:
        raise RuntimeError(f"outside control sandbox: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 1024:
            raise RuntimeError(f"link/reparse in control target: {current}")
    if not path.resolve(strict=False).is_relative_to(boundary):
        raise RuntimeError(f"outside real control sandbox: {path}")
    return path

tree = checked(args.tree)
if "control-code" not in tree.parts:
    raise RuntimeError("only a disposable control-code copy may be mutated")
os.environ['CARGO_TARGET_DIR'] = cache_safety.control_target()
cache_safety.refresh(tree)
print('CONTROL TARGET ' + os.environ['CARGO_TARGET_DIR'], flush=True)
logs = checked(args.logs)
checked(logs).mkdir(parents=True, exist_ok=True)
source = checked(tree / "crates/minion-agent/src/execution/filesystem/rimraf.rs")
original = source.read_text(encoding="utf-8")
prefix = "execution::filesystem::rimraf::tests::div008::"
controls = [
    ("stop-at-first-terminal", "div008_failed_subtree_keeps_ancestors_but_successful_sibling_directory_is_removed",
     "            first_error.get_or_insert(error);", "            return Err(error);" , "continue siblings, retaining only"),
    ("report-last-terminal", "div008_reports_first_terminal_error_unchanged_not_last",
     "            first_error.get_or_insert(error);", "            first_error = Some(error);", "first terminal origin, not last"),
    ("global-failure-skips-unaffected-directory", "div008_failed_subtree_keeps_ancestors_but_successful_sibling_directory_is_removed",
     "Work::FinalDirectory(_, before) if before != failures", "Work::FinalDirectory(_, _before) if failures != 0", "continue siblings, retaining only"),
    ("final-rmdir-after-failed-descendant", "div008_failed_subtree_keeps_ancestors_but_successful_sibling_directory_is_removed",
     "Work::FinalDirectory(_, before) if before != failures", "Work::FinalDirectory(_, before) if before > failures", "no final rmdir of failed subtree"),
]

def run(name, witness):
    output = checked(logs / (name + ".log"))
    with checked(output).open("w", encoding="utf-8") as stream:
        result = subprocess.run(["cargo", "test", "-p", "minion-agent", "--lib", prefix + witness,
                                 "--", "--exact", "--nocapture"], cwd=tree, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode, output.read_text(encoding="utf-8")

try:
    for name, witness, old, new, reason in controls:
        code, text = run(name + "-baseline", witness)
        if code != 0 or prefix + witness + " ... ok" not in text or "1 passed; 0 failed" not in text:
            raise RuntimeError(f"INVALID baseline: {name}")
        if original.count(old) != 1:
            raise RuntimeError(f"INVALID mutation anchor: {name}")
        checked(source).write_text(original.replace(old, new), encoding="utf-8")
        code, text = run(name, witness)
        if code != 101 or prefix + witness + " ... FAILED" not in text or reason not in text or "error[E" in text:
            raise RuntimeError(f"INVALID or SURVIVED: {name}")
        print(f"KILLED {name}: {witness}: {reason}", flush=True)
        checked(source).write_text(original, encoding="utf-8")
finally:
    checked(source).write_text(original, encoding="utf-8")
code, text = run("restored-baseline", controls[0][1])
if code != 0 or "1 passed; 0 failed" not in text:
    raise RuntimeError("INVALID restored baseline")
print("DIV008 CONTROLS PASS 4/4; restored baseline green", flush=True)
