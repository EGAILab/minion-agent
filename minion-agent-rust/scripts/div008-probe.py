"""Execute the real Rust provider probe and the shared comparator v3.

All writable paths are in the project (or the private /tmp/l12d008 sandbox).
Native evidence is never derived from a failed or partial Rust run.
"""
import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--logs", type=Path, required=True)
parser.add_argument("--natural", action="store_true")
parser.add_argument("--binary", type=Path, help="Cargo-built test executable for unprivileged replay")
args = parser.parse_args()
boundary = Path("E:/AI/Projects/OpenMinds/Minions/Minion-Agent") if os.name == "nt" else Path("/tmp/l12d008")
boundary = boundary.resolve(strict=True)

def checked(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(boundary) or path == boundary:
        raise RuntimeError(f"outside probe sandbox: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 1024:
            raise RuntimeError(f"link/reparse in probe target: {current}")
    if not path.resolve(strict=False).is_relative_to(boundary):
        raise RuntimeError(f"outside real probe sandbox: {path}")
    return path

tree = checked(args.tree)
data = args.data.resolve(strict=True)  # Read-only shared evidence may be mounted elsewhere.
logs = checked(args.logs)
checked(logs).mkdir(parents=True, exist_ok=True)
platform = "win32" if os.name == "nt" else "linux"
if args.natural and (os.name == "nt" or os.getuid() == 0):
    raise RuntimeError("natural permission evidence requires unprivileged Linux")
environment = dict(os.environ)
environment.pop("DIV008_NATURAL", None)
if args.natural:
    environment["DIV008_NATURAL"] = "1"

def execute(name, command):
    output = checked(logs / (name + ".log"))
    with checked(output).open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=tree, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; inspect {output}, do not credit partial evidence")
    return output.read_text(encoding="utf-8")

witness = "execution::filesystem::rimraf::tests::div008::div008_controlled_rows_continue_and_preserve_drained_extent"
command = ([str(args.binary.resolve(strict=True)), witness, "--exact", "--nocapture"] if args.binary else
           ["cargo", "test", "-p", "minion-agent", "--lib", witness, "--", "--exact", "--nocapture"])
text = execute("probe", command)
if witness + " ... ok" not in text or "1 passed; 0 failed" not in text:
    raise RuntimeError("probe witness absent or incomplete")
observed = [json.loads(line[7:]) for line in text.splitlines() if line.startswith("DIV008 ")]
matrix = json.loads((data / "div008-matrix.json").read_text(encoding="utf-8"))
expected = {(row["id"], run) for row in matrix["rows"]
            if "platforms" not in row or args.natural
            for run in range(1, row.get("repeat", 1) + 1)}
if {(row["id"], row["run"]) for row in observed} != expected or len(observed) != len(expected):
    raise RuntimeError("probe matrix incomplete or duplicated")
result = checked(logs / f"div008-rust-{platform}.jsonl")
checked(result).write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in observed), encoding="utf-8")
# Without natural rows, filter the oracle/matrix consistently for a development
# run. Full Linux certification must call --natural as uid 1000.
if platform == "linux" and not args.natural:
    raise RuntimeError("focused rows passed; full Linux comparator requires --natural")
command = [sys.executable, str(data / "div008_compare.py"), str(data / "div008-matrix.json"),
           str(data / f"div008-pi-{platform}.jsonl"), str(result)]
execute("comparator", command)
execute("comparator-controls", command + ["--controls"])
execute("comparator-selftest", [sys.executable, str(data / "div008_compare.py"), "--self-test"])
print(f"DIV008 PROBE/COMPARATOR/CONTROLS PASS: {len(observed)} {platform} rows", flush=True)
