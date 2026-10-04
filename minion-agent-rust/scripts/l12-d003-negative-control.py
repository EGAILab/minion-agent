"""Linux procps-ng regression control; never edits the candidate.

Run with the positive gate's Cargo/ICU environment. The rejected helper can
kill its caller's process group, so execute the compiled test in a NEW SESSION.
A build failure does not count as a killed mutant.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    if os.uname().sysname != "Linux":
        raise SystemExit("This control requires Linux and procps-ng kill")
    subprocess.run(["kill", "--version"], check=True)
    workspace = Path(__file__).resolve().parents[1]
    relative = Path("crates/minion-agent/src/execution/subprocess.rs")
    original = (workspace / relative).read_text(encoding="utf-8")
    anchor = '.args(["-KILL", "--", &group])'
    assert original.count(anchor) == 1
    with tempfile.TemporaryDirectory(prefix="minion-l12d003-") as directory:
        scratch = Path(directory) / "rust"
        shutil.copytree(workspace, scratch, ignore=shutil.ignore_patterns("target", ".git"))
        shutil.copytree(workspace.parent / "conformance", Path(directory) / "conformance")
        fixture = Path("minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt")
        destination = Path(directory) / fixture
        destination.parent.mkdir(parents=True)
        shutil.copyfile(workspace.parent / fixture, destination)
        (scratch / relative).write_text(
            original.replace(anchor, '.args(["-KILL", &group])'), encoding="utf-8"
        )
        subprocess.run(["cargo", "clean", "-p", "minion-agent"], cwd=scratch, check=True)
        build = subprocess.run(
            ["cargo", "test", "--locked", "--offline", "-p", "minion-agent",
             "--all-features", "--lib", "--no-run", "--message-format=json"],
            cwd=scratch, capture_output=True, text=True,
        )
        if build.returncode:
            for line in build.stdout.splitlines():
                message = json.loads(line)
                if message.get("reason") == "compiler-message":
                    print(message["message"]["rendered"])
            print(build.stderr)
            raise SystemExit("CONTROL INVALID: compilation/setup failed")
        artifacts = [json.loads(line) for line in build.stdout.splitlines()]
        binaries = [a["executable"] for a in artifacts
                    if a.get("reason") == "compiler-artifact"
                    and a.get("profile", {}).get("test") and a.get("executable")]
        assert len(binaries) == 1, binaries
        run = subprocess.run(
            [binaries[0], "group_kill_terminates_ready_parent_and_pipe_holding_descendant",
             "--nocapture"], capture_output=True, text=True,
            start_new_session=True, timeout=20,
        )
        print(run.stdout, run.stderr, sep="")
        killed = run.returncode == -9 or (
            run.returncode != 0 and "test result: FAILED. 0 passed; 1 failed;" in run.stdout
        )
        print(json.dumps({"control": "missing-option-terminator",
                          "compiled": True, "exit_code": run.returncode,
                          "killed": killed}))
        if not killed:
            raise SystemExit("MUTANT SURVIVED")


if __name__ == "__main__":
    main()
