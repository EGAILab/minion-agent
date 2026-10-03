"""WP-12.E5 negative controls (`spec/execution.md` section 16.5): each fault is applied to a
temporary copy of `src`, and `tests/execution/test_terminate_child.py` must fail against it.

    python scripts/e5_negative_controls.py [--list] [NAME ...]

Faults are platform-specific: a POSIX fault is only meaningful on a POSIX host and a Windows fault
on Windows; the others run anywhere. Exit status 0 iff every selected applicable fault was killed.
Set `E5_BARE_PYTEST=1` in a container that lacks the project's pytest-cov configuration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = Path("minion_agent/execution/subprocess.py")
TESTS = "tests/execution/test_terminate_child.py"

_CALLED = "        self._terminate_child_called = True\n"
_CALL_GATE = "            self._terminate_child_called\n            or self._terminate_called"
_POSIX_CALL = "        _posix_terminate_child(self.pid)  # pragma: no cover\n"
_PIDFD_SEND = "_os_signal.pidfd_send_signal(pidfd, _os_signal.SIGTERM)"
_KILL_SEND = "os.kill(pid, _os_signal.SIGTERM)"
_ZOMBIE_GATE = "        if exited is not None:\n            return\n"
_WIN_TERMINATE = "                popen.terminate()\n"
_WAIT_MAP = "if returncode >= 0 and not self._terminated_by_child_request"
_IGNORE = "  # type: ignore[attr-defined]"

FAULTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    # name: (platform, [(anchor, replacement)])
    "as-terminate": (
        "any",
        [(_CALLED, _CALLED + "        await self.terminate()\n        return\n")],
    ),
    "posix-sigkill": (
        "posix",
        [
            (_PIDFD_SEND, "_os_signal.pidfd_send_signal(pidfd, _os_signal.SIGKILL)"),
            (_KILL_SEND, "os.kill(pid, _os_signal.SIGKILL)"),
        ],
    ),
    "posix-killpg": (
        "posix",
        [
            (_PIDFD_SEND, "os.killpg(pid, _os_signal.SIGTERM)"),
            (_KILL_SEND, "os.killpg(pid, _os_signal.SIGTERM)"),
        ],
    ),
    "handled-exit-as-no-status": (
        "any",
        [(_WAIT_MAP, "if returncode >= 0 and not self._terminate_child_called")],
    ),
    "delayed-exit-completed-at-request": (
        "posix",
        [
            (
                _POSIX_CALL,
                _POSIX_CALL + "        self._wait_result = Ok(ExitStatus(exit_code=None))\n",
            )
        ],
    ),
    # WP12E5-I001: the rejected candidate -- an unsynchronized signal to the integer PID
    "unsynchronized-kill": (
        "posix",
        [(_POSIX_CALL, f"        os.kill(self.pid, _os_signal.SIGTERM){_IGNORE}\n")],
    ),
    "zombie-signalled": ("posix", [(_ZOMBIE_GATE, "")]),
    "resend-on-repeat": (
        "posix",
        [(_CALL_GATE, "            False\n            or self._terminate_called")],
    ),
    "windows-synthesized-1": ("nt", [(_WAIT_MAP, "if returncode >= 0")]),
    "windows-tree-kill": (
        "nt",
        [(_WIN_TERMINATE, "                await _confirm_kill(await _issue_kill(self.pid))\n")],
    ),
    "claims-the-cause": (
        "posix",
        [
            (
                _CALLED,
                _CALLED
                + "        if self._kill_cause is None:\n"
                + '            self._kill_cause = "explicit"\n',
            )
        ],
    ),
}


def applicable(platform: str) -> bool:
    return platform == "any" or (platform == "nt") == (os.name == "nt")


def _command() -> list[str]:
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q"]
    if os.environ.get("E5_BARE_PYTEST"):
        return [*base, "-c", os.devnull, f"--rootdir={ROOT}", "-o", "asyncio_mode=auto", TESTS]
    return [*base, "--no-cov", TESTS]


def run(name: str) -> dict[str, object]:
    platform, edits = FAULTS[name]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        shutil.copytree(ROOT / "src", src)
        path = src / TARGET
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        for anchor, replacement in edits:
            if text.count(anchor) != 1:
                return {"name": name, "error": f"anchor matched {text.count(anchor)} times"}
            text = text.replace(anchor, replacement)
        path.write_bytes(text.encode("utf-8"))
        env = {**os.environ, "PYTHONPATH": str(src)}
        result = subprocess.run(
            _command(), cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
        )
        lines = result.stdout.splitlines()
        failed = [line.split(" ")[1] for line in lines if line.startswith("FAILED ")]
        summary = lines[-1] if lines else ""
        killed = result.returncode != 0
        return {"name": name, "platform": platform, "killed": killed, "summary": summary,
                "failed": failed}  # fmt: skip


def main(argv: list[str]) -> int:
    if argv[:1] == ["--list"]:
        print(json.dumps({k: v[0] for k, v in FAULTS.items()}, indent=1))
        return 0
    names = argv or list(FAULTS)
    results = [run(n) for n in names if applicable(FAULTS[n][0])]
    print(json.dumps({"os": os.name, "results": results}, indent=1))
    return 0 if all(r.get("killed") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
