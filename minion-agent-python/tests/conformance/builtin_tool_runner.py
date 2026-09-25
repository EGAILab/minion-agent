"""Runner for `builtin_tool` scenarios (`conformance/schema/builtin-tool-scenario.schema.json`).

Thin by design: it builds the fixture on disk, layers the scenario's scripted provider responses
over the REAL `LocalFileSystem` (recording every `ctx.fs` call), constructs the REAL `read`/`ls`
tools and runs each case through the REAL Layer 06 `execute_call` pipeline. It never computes a
tool's output itself.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from minion_agent.execution import Err, FsError, FsErrorCode, LocalFileSystem, Ok
from minion_agent.execution.filesystem import DirEntryProbe, DirEntryProbeKind
from minion_agent.llm import ImageBlock, TextBlock, ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin import ReadToolOptions, create_ls_tool, create_read_tool
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "fixtures"


def _content(spec: dict[str, Any]) -> bytes:
    if "text" in spec:
        return spec["text"].encode("utf-8")
    if "base64" in spec:
        return base64.b64decode(spec["base64"])
    if "lines" in spec:
        template = spec["lines"]["template"]
        count = spec["lines"]["count"]
        return "\n".join(template.replace("{n}", str(n)) for n in range(1, count + 1)).encode()
    if "repeat" in spec:
        return (spec["repeat"]["unit"] * spec["repeat"]["times"]).encode("utf-8")
    return (FIXTURES / spec["fixture_file"]).read_bytes()


def build_fixture(root: Path, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if "dir" in entry:
            target.mkdir(exist_ok=True)
        elif "symlink" in entry:
            link_target = root / entry["symlink"]
            os.symlink(link_target, target, target_is_directory=link_target.is_dir())
        else:
            target.write_bytes(_content(entry["file"]))


class ScriptedFileSystem:
    """The real local `ctx.fs`, with the scenario's scripted answers for named paths and a log of
    every call. Unscripted calls go to the real provider unchanged."""

    def __init__(
        self,
        root: Path,
        provider: dict[str, Any],
        abort_after: str | None = None,
        controller: RunAbortController | None = None,
    ) -> None:
        self._root = root
        self._local = LocalFileSystem(str(root))
        self._provider = provider
        self._abort_after = abort_after
        self._controller = controller
        self.calls: list[str] = []

    def _answered(self, operation: str) -> None:
        """The case's `abort_after` point: abort as this operation hands back its answer."""
        if operation == self._abort_after and self._controller is not None:
            self._controller.abort()

    def _scripted_error(self, operation: str, path: str) -> Any:
        scripted = self._scripted(operation, path)
        return (
            None
            if scripted is None
            else Err(FsError(FsErrorCode(scripted["error"]), "scripted", path))
        )

    def relative(self, path: str) -> str:
        absolute = Path(path) if os.path.isabs(path) else self._root / path
        try:
            rel = os.path.relpath(absolute, self._root)
        except ValueError:
            return path
        return "." if rel == "." else rel.replace(os.sep, "/")

    def _scripted(self, operation: str, path: str) -> dict[str, Any] | None:
        rel = self.relative(path)
        for entry in self._provider.get(operation, []):
            if entry["path"] == rel:
                return entry
        return None

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._local, name)

        async def recorded(*args: Any, **kwargs: Any) -> Any:
            first = args[0] if args else ""
            self.calls.append(f"{name} {self.relative(first) if isinstance(first, str) else '*'}")
            return await method(*args, **kwargs)

        return recorded

    async def probe_dir_entry(self, path: str, signal: Any = None) -> Any:
        self.calls.append(f"probe_dir_entry {self.relative(path)}")
        if self._provider.get("without_exec_007"):
            return Err(FsError(FsErrorCode.NOT_SUPPORTED, "not supported", path))
        scripted = self._scripted("probe_dir_entry", path)
        if scripted is None:
            return await self._local.probe_dir_entry(path, signal)
        if "error" in scripted:
            return Err(FsError(FsErrorCode(scripted["error"]), "scripted", path))
        return Ok(DirEntryProbe(os.path.basename(path), path, DirEntryProbeKind(scripted["kind"])))

    async def list_dir_raw(self, path: str, signal: Any = None) -> Any:
        self.calls.append(f"list_dir_raw {self.relative(path)}")
        if self._provider.get("without_exec_007"):
            return Err(FsError(FsErrorCode.NOT_SUPPORTED, "not supported", path))
        scripted = self._scripted("list_dir_raw", path)
        if scripted is None:
            return await self._local.list_dir_raw(path, signal)
        if "error" in scripted:
            return Err(FsError(FsErrorCode(scripted["error"]), "scripted", path))
        return Ok(list(scripted["names"]))

    async def check_readable(self, path: str, signal: Any = None) -> Any:
        self.calls.append(f"check_readable {self.relative(path)}")
        if self._provider.get("without_exec_008"):
            answer: Any = Err(FsError(FsErrorCode.NOT_SUPPORTED, "not supported", path))
        else:
            answer = self._scripted_error("check_readable", path)
            if answer is None:
                answer = await self._local.check_readable(path, signal)
        self._answered("check_readable")
        return answer

    async def canonical_path(self, path: str, signal: Any = None) -> Any:
        self.calls.append(f"canonical_path {self.relative(path)}")
        answer = self._scripted_error("canonical_path", path)
        return answer if answer is not None else await self._local.canonical_path(path, signal)

    async def file_info(self, path: str, signal: Any = None) -> Any:
        self.calls.append(f"file_info {self.relative(path)}")
        scripted = self._scripted("file_info", path)
        if scripted is None:
            return await self._local.file_info(path, signal)
        return Err(FsError(FsErrorCode(scripted["error"]), "scripted", path))

    async def read_binary_file(self, path: str, signal: Any = None) -> Any:
        self.calls.append(f"read_binary_file {self.relative(path)}")
        scripted = self._scripted("read_binary_file", path)
        if scripted is None:
            return await self._local.read_binary_file(path, signal)
        return Err(FsError(FsErrorCode(scripted["error"]), "scripted", path))


_ABS_TOKEN = re.compile(r"\{abs:([^}]*)\}")


async def _expand(text: str, fs: LocalFileSystem) -> str:
    out = text
    for token in set(_ABS_TOKEN.findall(text)):
        resolved = await fs.absolute_path(token)
        assert isinstance(resolved, Ok)
        out = out.replace("{abs:" + token + "}", resolved.value)
    return out


async def run_builtin_tool_scenario(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns one `(observed, expected)` pair per case, expected text already expanded."""
    spec = document["builtin_tool"]
    results: list[dict[str, Any]] = []
    for case in spec["cases"]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            build_fixture(root, spec.get("fixture", []))
            controller = RunAbortController()
            fs = ScriptedFileSystem(
                root, spec.get("provider", {}), case.get("abort_after"), controller
            )
            options = spec.get("options", {})
            supports = options.get("model_supports_images")
            read_options = ReadToolOptions(
                auto_resize_images=options.get("auto_resize_images", True),
                model_supports_images=None if supports is None else (lambda s=supports: s),
            )
            tool = (
                create_read_tool(fs, read_options)  # type: ignore[arg-type]
                if spec["tool"] == "read"
                else create_ls_tool(fs)  # type: ignore[arg-type]
            )
            registry = ToolRegistry()
            registry.register(tool)
            ctx = Context()
            declare_tools_events(ctx.events)
            if case.get("signal") == "pre_aborted":
                controller.abort()
            result = await execute_call(
                ToolCallBlock(id="call-1", name=spec["tool"], arguments=case["arguments"]),
                registry=registry,
                ctx=ctx,
                signal=controller.signal,
            )
            first = result.content[0]
            assert isinstance(first, TextBlock)
            image = None
            if len(result.content) > 1:
                block = result.content[1]
                assert isinstance(block, ImageBlock) and block.data is not None
                image = {
                    "mime_type": block.mime_type,
                    "sha256": hashlib.sha256(block.data).hexdigest(),
                    "bytes": len(block.data),
                }
            observed: dict[str, Any] = {
                "is_error": result.is_error,
                "text": first.text,
                "text_sha256": hashlib.sha256(first.text.encode("utf-8")).hexdigest(),
                "image": image,
                "details": result.details,
                "fs_calls": fs.calls,
                "probed_entries": [
                    call.split(" ", 1)[1]
                    for call in fs.calls
                    if call.startswith("probe_dir_entry ")
                ],
            }
            expected = dict(case["expect"])
            for key in ("text", "text_tail"):
                if key in expected:
                    expected[key] = await _expand(expected[key], LocalFileSystem(str(root)))
            results.append({"id": case.get("id", ""), "observed": observed, "expected": expected})
    return results
