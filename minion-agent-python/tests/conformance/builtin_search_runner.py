"""Runner for the WP-13.4 `find`/`grep` canonical scenarios
(`conformance/schema/builtin-search-scenario.schema.json`). It builds the named corpus from
`conformance/agent/builtin-search/corpus.json`, constructs the real tool over the real local
`ctx.fs` and `ctx.subprocess` with the certified pinned engine store, and runs the case through
the REAL Layer 06 `execute_call` pipeline. It never performs tool behaviour: it only builds the
tree and normalizes the corpus root to `<ROOT>`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from minion_agent.execution import LocalFileSystem
from minion_agent.execution.subprocess import LocalSubprocess
from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context, RunAbortController
from minion_agent.tools.builtin.find import create_find_tool
from minion_agent.tools.builtin.grep import create_grep_tool
from minion_agent.tools.builtin.search_engines import EngineStore
from minion_agent.tools.events import declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry

HOST_PLATFORM = "win32" if os.name == "nt" else "linux"
SEARCH_DIR = Path(__file__).resolve().parents[3] / "conformance" / "agent" / "builtin-search"
CORPORA: dict[str, Any] = json.loads((SEARCH_DIR / "corpus.json").read_text(encoding="utf-8"))


def _write(root: Path, relative: str, content: dict[str, str]) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(content["hex"]) if "hex" in content else content["text"].encode("utf-8")
    )


def _link(target: Path, link: Path, *, directory: bool) -> None:
    if directory and os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=directory)


@contextmanager
def corpus(kind: str) -> Iterator[Path]:
    spec = CORPORA[kind]
    with tempfile.TemporaryDirectory(prefix=f"wp134-{kind}-") as tmp:
        root = Path(tmp).resolve()
        for relative, content in spec["files"].items():
            _write(root, relative, content)
        for directory in spec.get("directories", []):
            root.joinpath(*directory.split("/")).mkdir(parents=True, exist_ok=True)
        for relative, entry in spec.get("raw_name_files", {}).get(HOST_PLATFORM, {}).items():
            parent = root.joinpath(*relative.split("/")[:-1])
            parent.mkdir(parents=True, exist_ok=True)
            Path(
                os.fsdecode(os.fsencode(parent) + b"/" + bytes.fromhex(entry["name_hex"]))
            ).write_text(entry["text"])
        for link in spec.get("links", []):
            target = root.joinpath(*link["target"].split("/"))
            _link(target, root.joinpath(*link["path"].split("/")), directory=link["kind"] == "dir")
        if spec.get("dangling_git_junction"):
            gone = root / "gone-target"
            gone.mkdir()
            _link(gone, root / ".git", directory=True)
            shutil.rmtree(gone)
        yield root


def normalize(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        for form in (str(root), str(root).replace("\\", "/")):
            value = value.replace(form, "<ROOT>")
        return value
    if isinstance(value, list):
        return [normalize(v, root) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v, root) for k, v in value.items()}
    return value


async def run_builtin_search_scenario(
    document: dict[str, Any], store: EngineStore
) -> dict[str, Any] | None:
    spec = document["builtin_search"]
    expected = spec["expect"].get(HOST_PLATFORM)
    if expected is None:
        return None
    with corpus(spec["corpus"]) as root:
        fs, subprocess = LocalFileSystem(str(root)), LocalSubprocess(str(root))
        factory = create_find_tool if spec["tool"] == "find" else create_grep_tool
        registry = ToolRegistry()
        registry.register(factory(fs, subprocess, store))
        ctx = Context()
        declare_tools_events(ctx.events)
        controller = RunAbortController()
        if spec.get("signal") == "pre_aborted":
            controller.abort()
        arguments = {
            k: v.replace("<ROOT>", str(root)) if isinstance(v, str) else v
            for k, v in spec["arguments"].items()
        }
        result = await execute_call(
            ToolCallBlock(id="call-1", name=spec["tool"], arguments=arguments),
            registry=registry,
            ctx=ctx,
            signal=controller.signal,
        )
        first = result.content[0]
        assert isinstance(first, TextBlock)
        observed = {
            "is_error": result.is_error,
            "text": normalize(first.text, root),
            "details": normalize(dict(result.details), root),
        }
    return {"observed": observed, "expected": expected}


def temporary_store(artifacts: str) -> EngineStore:
    from minion_agent.tools.builtin.search_engines import provision_search_engines

    store = EngineStore(root=Path(tempfile.mkdtemp(prefix="wp134-engines-")))
    provision_search_engines(store, source=artifacts)
    return store
