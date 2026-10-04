"""L12-D002 negative controls (spec/execution.md section 16.6): each fault is applied to a temporary
copy of `src`, and `tests/execution/test_subprocess_wait_exit.py` (with the Layer-12 subprocess
suite) must fail against it.

    python scripts/l12_d002_negative_controls.py [--list] [NAME ...]

Exit status 0 iff every selected fault was killed. Set `E5_BARE_PYTEST=1` in a container that lacks
the project's pytest-cov configuration.
"""

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
TESTS = ["tests/execution/test_subprocess_wait_exit.py", "tests/execution/test_subprocess.py"]

_WAIT = "            await asyncio.shield(self._exited)\n"
_CLOSE_SET = "        self._closed = True\n        _close_owned_transport(self._reader)\n"

FAULTS: dict[str, list[tuple[str, str]]] = {
    # wait() waits for every pipe to disconnect (the original Windows defect)
    "wait-waits-for-pipes": [(_WAIT, "            await self._proc.wait()\n")],
    # wait() closes the process transport, which closes still-open pipes (read-after-wait lost)
    "wait-closes-streams": [(_WAIT, _WAIT + "            _close_owned_transport(self._proc)\n")],
    # a cancelled waiter cancels the shared exit notification
    "unshielded-exit-future": [(_WAIT, "            await self._exited\n")],
    # close() only sets the flag: a pending read never settles
    "close-leaves-read-pending": [(_CLOSE_SET, "        self._closed = True\n")],
    # close() does not mark the stream: reads after close keep returning buffered/live data
    "close-without-eof-state": [(_CLOSE_SET, "        _close_owned_transport(self._reader)\n")],
    # close() releases every stream created so far, the sibling included
    "close-affects-sibling": [
        (
            "    _CHUNK_SIZE = 65536\n",
            "    _CHUNK_SIZE = 65536\n    _ALL: list = []  # type: ignore[type-arg]\n",
        ),
        (
            "        self._reader = reader\n        self._closed = False\n",
            "        self._reader = reader\n        self._closed = False\n"
            "        ReadableStream._ALL.append(self)\n",
        ),
        (
            _CLOSE_SET,
            _CLOSE_SET + "        for other in ReadableStream._ALL:\n"
            "            other._closed = True\n            _close_owned_transport(other._reader)\n",
        ),
    ],
    # close() closes the owning subprocess transport, which kills a running child
    "close-kills-process": [
        (
            _CLOSE_SET,
            _CLOSE_SET + "        pipe = getattr(self._reader, '_transport', None)\n"
            "        owner = getattr(getattr(pipe, '_protocol', None), 'proc', None)\n"
            "        if owner is not None:\n            owner.close()\n",
        ),
    ],
    # an intentional close surfaces as pipe_error
    "close-reports-pipe-error": [
        (
            "            if self._closed:\n"
            "                return Ok(None)  # the caller's own close(), not a pipe failure\n",
            "",
        ),
        (
            "        if self._closed:\n            return Ok(None)\n        try:\n",
            "        if self._closed:\n"
            "            return Err(SubprocessError(SubprocessErrorCode.PIPE_ERROR, 'closed'))\n"
            "        try:\n",
        ),
    ],
}


def _command() -> list[str]:
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly", "-q"]
    strict = ["-W", "error::pytest.PytestUnraisableExceptionWarning"]
    if os.environ.get("E5_BARE_PYTEST"):
        return [
            *base,
            "-c",
            os.devnull,
            f"--rootdir={ROOT}",
            "-o",
            "asyncio_mode=auto",
            *strict,
            *TESTS,
        ]
    return [*base, "--no-cov", *strict, *TESTS]


def run(name: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        shutil.copytree(ROOT / "src", src)
        path = src / TARGET
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        for anchor, replacement in FAULTS[name]:
            if text.count(anchor) != 1:
                return {"name": name, "error": f"anchor matched {text.count(anchor)} times"}
            text = text.replace(anchor, replacement)
        path.write_bytes(text.encode("utf-8"))
        env = {**os.environ, "PYTHONPATH": str(src)}
        try:
            result = subprocess.run(
                _command(), cwd=ROOT, env=env, capture_output=True, text=True, timeout=180
            )
        except subprocess.TimeoutExpired:
            return {"name": name, "killed": True, "summary": "hung (timeout)", "failed": []}
        lines = result.stdout.splitlines()
        failed = [line.split(" ")[1] for line in lines if line.startswith(("FAILED ", "ERROR "))]
        return {"name": name, "killed": result.returncode != 0,
                "summary": lines[-1] if lines else "", "failed": failed}  # fmt: skip


def main(argv: list[str]) -> int:
    if argv[:1] == ["--list"]:
        print(json.dumps(sorted(FAULTS), indent=1))
        return 0
    results = [run(name) for name in (argv or list(FAULTS))]
    print(json.dumps({"os": os.name, "results": results}, indent=1))
    return 0 if all(r.get("killed") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
