"""WP-13.3 `bash` negative controls (spec/tools.md WP-13.3 "Negative controls"; spec/execution.md
section 16.6 for the two pipe-release controls). Each fault is applied to a temporary copy of `src`
and the `bash` suites (unit, shell, output and the canonical scenarios) must fail against it.

    python scripts/wp133_bash_negative_controls.py [--list] [NAME ...]

Exit status 0 iff every selected fault was killed. Set `E5_BARE_PYTEST=1` in a container that lacks
the project's pytest-cov configuration. Two listed controls have no injectable seam in this tool and
are covered structurally (`STRUCTURAL` below, reported but not run).
"""

# ruff: noqa: E501 -- the fault table quotes source lines verbatim as unique anchors

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = Path("minion_agent/tools/builtin")
TESTS = [
    "tests/tools/builtin/test_bash.py",
    "tests/tools/builtin/test_bash_shell.py",
    "tests/tools/builtin/test_bash_output.py",
    "tests/conformance/test_builtin_bash_conformance.py",
]

BASH = PKG / "bash.py"
SHELL = PKG / "bash_shell.py"
OUTPUT = PKG / "bash_output.py"

_ABORT = "        if signal is not None and signal.aborted:\n            raise _Aborted\n"
_TIMEOUT = "        timeout_ms = resolve_timeout_ms(timeout)\n"
_SELECT = (
    "        try:\n"
    "            config: ShellConfig = await select_shell(fs, subprocess, shell_path)\n"
    "        except ShellNotFoundError as error:\n"
    "            raise BuiltinToolError(str(error)) from error\n"
)
_CLASSIFY = (
    "        if signal is not None and signal.aborted:\n"
    "            raise _AbortedWith(run)\n"
    "        if run.timed_out:\n"
    "            raise _TimedOutWith(run)\n"
)
_NONZERO = '            raise BuiltinToolError(_with_status(text, f"Command exited with code {exit_code}"))\n'
_DECODER = '        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")\n'
_DECODE = "        self._append_decoded(self._decode(data, final=False))\n"
_JOIN = (
    "                await writer  # joins every accepted write; never cancels one (WP133-I002)\n"
)
_LOOKUP_KILL = (
    "            # Interrupted while the lookup is still running: the certified hard kill (DIV-001).\n"
    "            await process.terminate()\n"
)

FAULTS: dict[str, list[tuple[Path, str, str]]] = {
    # ---- precheck order (Pi's exec): timeout, then abort, then shell / cwd ----
    "abort-before-timeout-validation": [(BASH, _TIMEOUT + _ABORT, _ABORT + _TIMEOUT)],
    "shell-selection-before-abort": [(BASH, _ABORT + _SELECT, _SELECT + _ABORT)],
    # ---- classification: the signal first, then the timeout ----
    "timeout-first-classification": [
        (
            BASH,
            _CLASSIFY,
            "        if run.timed_out:\n            raise _TimedOutWith(run)\n"
            "        if signal is not None and signal.aborted:\n            raise _AbortedWith(run)\n",
        )
    ],
    # ---- settlement ----
    "settle-on-wait-alone": [
        (BASH, "            while not pumps.done():\n", "            while False:\n")
    ],
    "eof-without-grace": [
        (
            BASH,
            "waiters, timeout=EXIT_STDIO_GRACE_S, return_when",
            "waiters, timeout=None, return_when",
        )
    ],
    # ---- decoding ----
    "decoder-per-stream": [
        (
            OUTPUT,
            "    def append(self, data: bytes) -> None:\n",
            "    def append(self, data: bytes, stream: int = 0) -> None:\n",
        ),
        (
            OUTPUT,
            _DECODE,
            "        decoders = self.__dict__.setdefault('_per_stream', {})\n"
            "        decoder = decoders.setdefault(\n"
            "            stream, codecs.getincrementaldecoder('utf-8')(errors='replace')\n"
            "        )\n"
            "        self._append_decoded(decoder.decode(data))\n",
        ),
        (
            BASH,
            "    def accept(self, chunk: bytes) -> None:\n",
            "    def accept(self, chunk: bytes, stream: int = 0) -> None:\n",
        ),
        (
            BASH,
            "        self.output.append(chunk)\n",
            "        self.output.append(chunk, stream)\n",
        ),
        (
            BASH,
            "            self.accept(chunk.value)\n",
            "            self.accept(chunk.value, id(stream))\n",
        ),
    ],
    "decoder-reset-per-chunk": [(OUTPUT, _DECODE, _DECODE + _DECODER)],
    "per-chunk-bom-strip": [(OUTPUT, _DECODE, _DECODE + "        self._at_stream_start = True\n")],
    "no-bom-strip": [
        (
            OUTPUT,
            "            if text[0] == _BOM:\n"
            "                return text[1:]  # exactly one leading BOM of the whole stream\n",
            "",
        )
    ],
    # WP133-I001: CPython's utf-8-sig drops an incomplete leading BOM prefix at EOF
    "bom-prefix-dropped-at-eof": [(OUTPUT, _DECODER, _DECODER.replace('"utf-8"', '"utf-8-sig"'))],
    # ---- the full-output file ----
    "decoded-text-to-full-output-file": [
        (
            BASH,
            "            written = await self.fs.append_file(self.full_output_path, pending)\n",
            "            written = await self.fs.append_file(\n"
            "                self.full_output_path, pending.decode('utf-8', 'replace').encode('utf-8')\n"
            "            )\n",
        )
    ],
    # ---- truncation and formatting ----
    "truncation-computed-from-tail": [
        (
            OUTPUT,
            "        truncated = self.total_lines > self.max_lines or self.total_decoded_bytes > self.max_bytes\n",
            "        truncated = tail['truncated']\n",
        )
    ],
    # WP133-I002: settlement must join accepted persistence, never cancel it
    "settlement-cancels-persistence": [(BASH, _JOIN, "                writer.cancel()\n")],
    "persist-inline-in-pump": [
        (
            BASH,
            "            self.accept(chunk.value)\n",
            "            self.accept(chunk.value)\n"
            "            await self._persist(self._accepted.get_nowait())\n",
        )
    ],
    "status-before-truncation-notice": [
        (
            BASH,
            _NONZERO,
            "            shown = run.output.snapshot()['content']\n"
            "            status = f'Command exited with code {exit_code}'\n"
            "            raise BuiltinToolError(_with_status(shown, status) + text[len(shown) :])\n",
        )
    ],
    "empty-text-of-nonzero-exit": [
        (
            BASH,
            '        text, details = _format_output(run, "(no output)")\n',
            '        text, details = _format_output(run, "(no output)" if not exit_code else "")\n',
        )
    ],
    "always-drop-partial-first-line": [
        (
            OUTPUT,
            "        if self._tail_starts_at_line_boundary:\n            return self._tail_text\n",
            "",
        )
    ],
    "never-drop-partial-first-line": [
        (OUTPUT, "        if self._tail_starts_at_line_boundary:\n", "        if True:\n")
    ],
    # ---- the timer (WP133-CON-R003) ----
    "timer-raw-ms": [
        (
            BASH,
            "    return max(1, math.trunc(timeout_ms))\n",
            "    return timeout_ms  # type: ignore[return-value]\n",
        )
    ],
    "timer-rounded-ms": [
        (
            BASH,
            "    return max(1, math.trunc(timeout_ms))\n",
            "    return max(1, round(timeout_ms))\n",
        )
    ],
    # ---- the lookup (WP133-CON-R005 / R006, DIV-001, C002) ----
    "lookup-unbounded": [
        (SHELL, "            if total > LOOKUP_OUTPUT_BUDGET:\n", "            if False:\n")
    ],
    "lookup-budget-per-stream": [(SHELL, "        nonlocal total\n", "        total = 0\n")],
    "lookup-settles-at-exit": [
        (
            SHELL,
            "    completion = asyncio.ensure_future(asyncio.wait(finished))\n",
            "    completion = asyncio.ensure_future(asyncio.wait({exited}))\n",
        )
    ],
    "lookup-idle-grace": [
        (
            SHELL,
            "    completion = asyncio.ensure_future(asyncio.wait(finished))\n",
            "    async def _grace() -> None:\n"
            "        await exited\n"
            "        await asyncio.wait({pumps}, timeout=0.1)\n\n"
            "    completion = asyncio.ensure_future(_grace())\n",
        )
    ],
    "lookup-fails-on-any-interruption": [
        (
            SHELL,
            "        status = await exited\n    finally:\n",
            "        status = await exited\n        if not completion.done():\n            return None\n    finally:\n",
        )
    ],
    "lookup-direct-sigterm": [
        (
            SHELL,
            _LOOKUP_KILL,
            "            import signal as _signal\n\n"
            "            process._proc.send_signal(getattr(_signal, 'SIGTERM', 15))  # type: ignore[attr-defined]\n",
        )
    ],
    "lookup-keeps-pipes": [
        (
            SHELL,
            "        await close_streams(process)  # collection finished or interrupted (section 16.4)\n",
            "",
        )
    ],
    # ---- existence checks (CE-WP133-01) ----
    "existence-via-canonical-path": [
        (
            SHELL,
            "    return not isinstance(await fs.probe_dir_entry(path), Err)\n",
            "    return not isinstance(await fs.canonical_path(path), Err)\n",
        )
    ],
    "existence-via-exists": [
        (
            SHELL,
            "    return not isinstance(await fs.probe_dir_entry(path), Err)\n",
            "    return not isinstance(await fs.exists(path), Err)\n",
        )
    ],
    "windows-cwd-check-follows": [
        (
            BASH,
            "    result = await (fs.file_info(cwd) if platform is Platform.WINDOWS else fs.probe_dir_entry(cwd))\n",
            "    result = await fs.probe_dir_entry(cwd)\n",
        )
    ],
    # ---- the session environment (Owner Q1) ----
    "case-insensitive-minion-removal": [
        (
            PKG / "environment.py",
            "    for name in remove:\n        env.pop(name, None)\n",
            "    doomed = {name.upper() for name in remove}\n"
            "    for name in [key for key in env if key.upper() in doomed]:\n"
            "        env.pop(name, None)\n",
        )
    ],
    "inject-none-for-absent": [
        (
            BASH,
            '    if context.session_file:\n        injected["MINION_SESSION_FILE"] = context.session_file\n',
            '    injected["MINION_SESSION_FILE"] = context.session_file or "none"\n',
        )
    ],
    "inject-empty-for-absent": [
        (
            BASH,
            '    if context.reasoning_level:\n        injected["MINION_REASONING_LEVEL"] = context.reasoning_level\n',
            '    injected["MINION_REASONING_LEVEL"] = context.reasoning_level or ""\n',
        )
    ],
    # ---- command projection (WP133-AUD-R001) ----
    "no-command-projection": [
        (BASH, "        projected = scalar_command(command)\n", "        projected = command\n")
    ],
    "surrogateescape-projection": [
        (
            BASH,
            '    return command.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")\n',
            '    return command.encode("utf-8", "surrogateescape").decode("utf-8", "replace")\n',
        )
    ],
    # ---- pipe release at settlement (spec/execution.md section 16.6) ----
    "bash-terminates-to-release-pipes": [
        (
            BASH,
            "            await _close_streams(self.process)\n",
            "            await self.process.terminate()\n",
        )
    ],
    "bash-keeps-pipes": [
        (BASH, "            await _close_streams(self.process)\n", "            pass\n")
    ],
}

STRUCTURAL = {
    "partial-updates-emitted": "the bash ToolDefinition takes no update callback (Q2: live partial "
    "updates are not certified); witness test_zero_partial_updates_with_the_final_result",
}


def _command() -> list[str]:
    base = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        "-q",
        "-x",
    ]
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
        texts: dict[Path, str] = {}
        for target, anchor, replacement in FAULTS[name]:
            path = src / target
            text = texts.get(path) or path.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if text.count(anchor) != 1:
                return {
                    "name": name,
                    "error": f"anchor matched {text.count(anchor)} times in {target}",
                }
            texts[path] = text.replace(anchor, replacement)
        for path, text in texts.items():
            path.write_bytes(text.encode("utf-8"))
        env = {**os.environ, "PYTHONPATH": str(src)}
        try:
            result = subprocess.run(
                _command(), cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired:
            return {"name": name, "killed": True, "summary": "hung (timeout)", "failed": []}
        lines = result.stdout.splitlines()
        failed = [line.split(" ")[1] for line in lines if line.startswith(("FAILED ", "ERROR "))]
        return {"name": name, "killed": result.returncode != 0,
                "summary": lines[-1] if lines else "", "failed": failed}  # fmt: skip


def main(argv: list[str]) -> int:
    if argv[:1] == ["--list"]:
        print(json.dumps({"injected": sorted(FAULTS), "structural": STRUCTURAL}, indent=1))
        return 0
    results = [run(name) for name in (argv or list(FAULTS))]
    print(json.dumps({"os": os.name, "results": results, "structural": STRUCTURAL}, indent=1))
    return 0 if all(r.get("killed") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
