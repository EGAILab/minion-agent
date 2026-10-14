"""L12-D012 sequential Rust gates, native corpus and isolated controls.

Windows: source project-environment.ps1, run preflight, then this script.
Linux: read-only /source, /build, /node, /search-artifacts; E:-backed writable
/cargo-home, /target, /control-target, /logs; private executable /tmp tmpfs.
Gate/control target parameters allow independent reviewer caches. Never clean.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

PROJECT = Path("E:/AI/Projects/OpenMinds/Minions/Minion-Agent")
WINDOWS = os.name == "nt"
SCRATCH = PROJECT if WINDOWS else Path("/tmp")
HERE = Path(__file__).resolve().parents[1]


def checked(path, boundary=SCRATCH):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path == boundary or not path.is_relative_to(boundary):
        raise RuntimeError("outside gate sandbox: " + str(path))
    for ancestor in [*reversed(path.parents), path]:
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 1024:
            raise RuntimeError("link/reparse in gate target: " + str(ancestor))
    if not path.resolve(strict=False).is_relative_to(boundary.resolve(strict=True)):
        raise RuntimeError("outside real gate sandbox")
    return path


def mkdir(path, boundary=SCRATCH):
    checked(path, boundary)
    missing = []
    current = path
    while True:
        try:
            current.lstat()
            break
        except FileNotFoundError:
            missing.append(current)
            current = current.parent
    for target in reversed(missing):
        checked(target, boundary).mkdir()


def copy_tree(source, destination, boundary=SCRATCH):
    for directory, dirs, files in os.walk(source, followlinks=False):
        target = destination / Path(directory).relative_to(source)
        mkdir(target, boundary)
        for name in list(dirs):
            if name in ("target", ".git", "__pycache__"):
                dirs.remove(name)
            else:
                metadata = (Path(directory) / name).lstat()
                if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 1024:
                    raise RuntimeError("non-plain directory copy source")
        for name in files:
            src = Path(directory) / name
            metadata = src.lstat()
            if not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 1024:
                raise RuntimeError("non-regular copy source")
            destination_file = checked(target / name, boundary)
            shutil.copyfile(src, destination_file)  # fresh mtimes, never copy2
            if not WINDOWS and destination_file.suffix == ".sh":
                checked(destination_file, boundary).write_bytes(destination_file.read_bytes().replace(b"\r\n", b"\n"))


def workspace_copy(source, destination):
    assert not destination.exists(), "NEW workspace copy required"
    copy_tree(source / "minion-agent-rust", destination / "minion-agent-rust")
    copy_tree(source / "conformance", destination / "conformance")
    for relative in ("pi-parity-manifest.yaml",
                     "minion-agent-python/tests/skills/data/frontmatter-corpus.json",
                     "minion-agent-python/tests/skills/data/ignore-corpus.json",
                     "minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt"):
        dst = destination / relative
        mkdir(dst.parent)
        shutil.copyfile(source / relative, checked(dst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--gate-target", type=Path, default=PROJECT / ".tmp/cache/cargo-target/windows" if WINDOWS else Path("/target"))
    parser.add_argument("--control-target", type=Path, default=PROJECT / ".tmp/cache/cargo-target/windows-controls" if WINDOWS else Path("/control-target"))
    parser.add_argument("--phase", choices=("all", "gates", "controls"), default="all")
    parser.add_argument("--only-controls", nargs="+")
    args = parser.parse_args()
    log_boundary = PROJECT if WINDOWS else Path("/logs")
    logs = checked(args.logs, log_boundary)
    assert not logs.exists(), "NEW log directory required"
    mkdir(logs, log_boundary)
    if WINDOWS:
        tree = HERE
        checked(tree)
        args.gate_target = checked(args.gate_target).resolve(strict=False)
        args.control_target = checked(args.control_target).resolve(strict=False)
        for platform in ("windows", "linux"):
            shared_gate = (PROJECT / ".tmp/cache/cargo-target" / platform).resolve(strict=False)
            assert not args.control_target.is_relative_to(shared_gate) and not shared_gate.is_relative_to(args.control_target), "shared gate cache cannot be a controls target"
        assert os.environ.get("CARGO_BUILD_JOBS") == "4", "load project environment"
        os.environ.update({
            "RUST_ICU_MAJOR_VERSION_NUMBER": "78",
            "RUSTFLAGS": "-L native=" + str(PROJECT / ".toolchain/icu-78.3-src/icu/lib64").replace("\\", "/"),
            "MINION_AGENT_ICU_IDENTITY": str(PROJECT / ".toolchain/icu-78.3-src/pinned-icu-identity.txt"),
            "MINION_SEARCH_ENGINE_ARTIFACTS": str(PROJECT / ".toolchain/search-engines/dl"),
            "PATH": str(PROJECT / ".toolchain/icu-78.3-src/icu/bin64") + os.pathsep + os.environ["PATH"],
        })
        os.environ["RUSTDOCFLAGS"] = "-D warnings " + os.environ["RUSTFLAGS"]
        private = logs
    else:
        assert args.gate_target == Path("/target") and args.control_target == Path("/control-target")
        private = Path("/tmp/l12d012-gates")
        assert not private.exists(), "fresh tmpfs required"
        mkdir(private)
        workspace_copy(Path("/source"), private / "work")
        tree = private / "work/minion-agent-rust"
        mkdir(private / "bin")
        shutil.copyfile("/node", checked(private / "bin/node"))
        os.chmod(checked(private / "bin/node"), 0o755)
        fixtures = private / "fixtures"
        mkdir(fixtures)
        os.chmod(checked(fixtures), 0o777)
        os.environ.update({
            "CARGO_HOME": "/cargo-home", "CARGO_BUILD_JOBS": "4", "RUSTUP_HOME": "/cargo-home/rustup", "RUSTUP_TOOLCHAIN": "1.97.1",
            "RUST_ICU_MAJOR_VERSION_NUMBER": "78", "RUSTFLAGS": "-L native=/build/icu/lib", "RUSTDOCFLAGS": "-D warnings -L native=/build/icu/lib",
            "MINION_AGENT_ICU_IDENTITY": "/build/icu-identity.txt", "LD_LIBRARY_PATH": "/build/icu/lib", "MINION_SEARCH_ENGINE_ARTIFACTS": "/search-artifacts",
            "MINION_FIXTURE_ROOT": str(fixtures), "TMP": str(fixtures), "TEMP": str(fixtures), "TMPDIR": str(fixtures),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": "/cargo-home/pycache", "XDG_CACHE_HOME": "/cargo-home/xdg",
            "NODE_COMPILE_CACHE": "/cargo-home/node-cache", "npm_config_cache": "/cargo-home/npm", "PIP_CACHE_DIR": "/cargo-home/pip", "UV_CACHE_DIR": "/cargo-home/uv",
            "PATH": str(private / "bin") + ":/cargo-home/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:" + os.environ["PATH"],
        })
    assert args.gate_target != args.control_target and not args.gate_target.is_relative_to(args.control_target) and not args.control_target.is_relative_to(args.gate_target)
    os.environ["CARGO_TARGET_DIR"] = str(args.gate_target)
    with checked(logs / "targets.json", log_boundary).open("w", encoding="utf-8") as output:
        json.dump({"gate_target": str(args.gate_target), "control_target": str(args.control_target),
                   "source": str(tree), "cargo_jobs": os.environ["CARGO_BUILD_JOBS"],
                   "phase": args.phase}, output, indent=2)
    spec = importlib.util.spec_from_file_location("cache_safety", tree / "scripts/cargo-cache-safety.py")
    safety = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safety)

    def run(name, command):
        safety.refresh(tree)  # every gate, not merely the first Cargo invocation
        print("START", name, flush=True)
        with checked(logs / (name + ".log"), log_boundary).open("w", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=tree, stdout=output, stderr=subprocess.STDOUT)
        checked(logs / (name + ".exit"), log_boundary).write_text(str(result.returncode) + "\n", encoding="ascii")
        print("END", name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    if WINDOWS:
        run("preflight", ["powershell", "-NoProfile", "-File", str(PROJECT / ".agents/gate-preflight.ps1"), "-MaxWaitMinutes", "0"])
    run("versions", [sys.executable, "-c", "import subprocess,sys; print(sys.version); commands=[['rustc','--version'],['cargo','--version'],['node','--version']]; outputs=[subprocess.check_output(command,text=True).strip() for command in commands]; print('\\n'.join(outputs)); assert outputs[0].startswith('rustc 1.97.1 '); assert outputs[2]=='v22.15.1'"])
    run("path-anchors", [sys.executable, "scripts/path-resolution-negative-controls.py", "--tree", str(tree), "--check-anchors"])
    run("error-anchors", [sys.executable, "scripts/error-codes-negative-controls.py", "--tree", str(tree), "--check-anchors"])
    if args.phase != "controls":
        for name, command in [
            ("fmt", ["cargo", "fmt", "--all", "--", "--check"]),
            ("native-fs-check", ["cargo", "check", "-p", "minion-agent-native-fs"]),
            ("clippy", ["cargo", "clippy", "--workspace", "--all-targets", "--all-features", "--", "-D", "warnings"]),
            ("tests", ["cargo", "test", "--workspace", "--all-features"]),
            ("doc", ["cargo", "doc", "--workspace", "--no-deps"]),
            ("xtask", ["cargo", "run", "-p", "xtask", "--", "conformance", "verify"]),
        ]:
            run(name, command)
        run("observer-build", ["cargo", "test", "--locked", "-p", "minion-agent", "--lib", "--no-run", "--message-format=json"])
        events = [json.loads(line) for line in (logs / "observer-build.log").read_text().splitlines() if line.startswith("{")]
        binaries = [row["executable"] for row in events if row.get("executable") and row.get("target", {}).get("name") == "minion_agent"]
        assert len(binaries) == 1
        probe = [sys.executable, "scripts/path-resolution-probe.py", "--binary", binaries[0], "--sandbox", str(private / "probe-fixture" if WINDOWS else fixtures / "probe-fixture")]
        if not WINDOWS:
            probe = ["setpriv", "--reuid", "1000", "--regid", "1000", "--clear-groups", *probe]
        run("path-probe", probe)
        if not WINDOWS:
            targets = ("fs_path_domain_conformance", "fs_remove_readonly_conformance", "fs_path_domain")
            command = ["cargo", "test", "-p", "minion-agent", "--all-features"]
            for target in targets:
                command += ["--test", target]
            run("uid-corpora-build", command + ["--no-run", "--message-format=json"])
            events = [json.loads(line) for line in (logs / "uid-corpora-build.log").read_text().splitlines() if line.startswith("{")]
            built = {row["target"]["name"]: row["executable"] for row in events if row.get("executable") and row.get("target", {}).get("name") in targets}
            assert set(built) == set(targets)
            for target, binary in built.items():
                extra = ["--exact", "recursive_remove_single_failure_reports_the_inner_native_call", "--ignored"] if target == "fs_path_domain" else []
                run("uid1000-" + target, ["setpriv", "--reuid", "1000", "--regid", "1000", "--clear-groups", binary, *extra, "--nocapture"])
    if args.phase != "gates":
        control_root = private / "control-code"
        workspace_copy(tree.parent, control_root)
        command = [sys.executable, "scripts/path-resolution-negative-controls.py", "--tree", str(control_root / "minion-agent-rust"),
                   "--logs", str(private / "control-details"), "--control-target", str(args.control_target)]
        if args.only_controls:
            command += ["--only", *args.only_controls]
        run("path-controls", command)
        if not WINDOWS:
            copy_tree(private / "control-details", logs / "control-details", log_boundary)
    checked(logs / "complete.exit", log_boundary).write_text("0\n", encoding="ascii")
    print("L12D012", args.phase, "COMPLETE", flush=True)


if __name__ == "__main__":
    main()
