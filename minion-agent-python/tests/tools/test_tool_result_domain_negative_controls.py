"""L0506-D003 negative controls (Owner decision WP132-RUST-C002-Q001 section 16): each realistic
WRONG implementation of
the tool-result runtime value domain, installed at the real pipeline seam it would live in, must
make the canonical
corpus FAIL -- at the boundary where the defect first becomes observable -- while the unmodified
pipeline passes.

Every mutant is a runtime monkeypatch of production code; the runner and the scenarios are
unchanged."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from minion_agent.runtime.events import EventBus
from minion_agent.session import derive
from minion_agent.tools import execute as execute_module
from minion_agent.tools.events import TOOLS_EXECUTION_END
from minion_agent.tools.result import ToolResult

from ..conformance import tool_result_domain_runner as runner
from ..conformance.test_tool_result_domain_conformance import CASES


def _fffd(text: str) -> str:
    """An unpaired surrogate (a surrogate code point in a Python str) replaced by U+FFFD."""
    return "".join("�" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in text)


def _map(
    value: Any,
    *,
    strings: bool = True,
    keys: bool = True,
    numbers: bool = False,
    depth: int = 0,
    min_depth: int = 0,
) -> Any:
    """`value` with the selected leaves normalized the way a lossy carrier would."""
    deeper = {
        "strings": strings,
        "keys": keys,
        "numbers": numbers,
        "depth": depth + 1,
        "min_depth": min_depth,
    }
    if isinstance(value, str):
        return _fffd(value) if strings and depth >= min_depth else value
    if isinstance(value, float) and numbers:
        if math.isnan(value) or math.isinf(value):
            return None  # JSON.stringify's projection
        return 0.0 if value == 0 else value  # -0 -> 0
    if isinstance(value, list | tuple):
        return [_map(item, **deeper) for item in value]
    if isinstance(value, dict):
        return {
            (_fffd(k) if keys and depth >= min_depth else k): _map(v, **deeper)
            for k, v in value.items()
        }
    return value


def _normalized(
    result: ToolResult, *, content: bool = True, details: Callable[[Any], Any] | None = _map
) -> ToolResult:
    blocks = (
        tuple(dataclasses.replace(b, text=_fffd(b.text)) for b in result.content)
        if content
        else result.content
    )
    return dataclasses.replace(
        result,
        content=blocks,
        details=details(result.details) if details is not None else result.details,
    )


async def _failures(tmp_path: Path) -> list[tuple[str, str]]:
    """(case id, first failing boundary) for every case the pipeline as currently patched fails."""
    failed = []
    for index, (_, case) in enumerate(CASES):
        root = tmp_path / str(index)
        root.mkdir()
        try:
            runner.check(case, await runner.run_case(case, root))
        except AssertionError as error:
            detail = error.args[0] if error.args else ()
            failed.append(
                (case["id"], detail[1] if isinstance(detail, tuple) and len(detail) > 1 else "?")
            )
        except Exception as error:
            failed.append((case["id"], f"raised {type(error).__name__}"))
    return failed


async def test_the_unmodified_pipeline_passes_every_case(tmp_path: Path) -> None:
    assert await _failures(tmp_path) == []


def _patch_finalize_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lone surrogate -> U+FFFD at tool return: the executed result is normalized before the
    after-hook."""
    original = execute_module._finalize

    async def finalize(executed: ToolResult, *args: Any, **kwargs: Any) -> ToolResult:
        return await original(_normalized(executed), *args, **kwargs)

    monkeypatch.setattr(execute_module, "_finalize", finalize)


def _patch_message(
    monkeypatch: pytest.MonkeyPatch, transform: Callable[[ToolResult], ToolResult]
) -> None:
    original = ToolResult.to_message
    monkeypatch.setattr(ToolResult, "to_message", lambda self: original(transform(self)))


def _patch_hook_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """The after-hook receives U+FFFD while the pipeline's own result stays correct."""
    original = execute_module.register_after_tool_call_hook

    def register(ctx: Any, hook: Any, **kwargs: Any) -> Any:
        return original(ctx, lambda result: hook(_normalized(result)), **kwargs)

    monkeypatch.setattr(runner, "register_after_tool_call_hook", register)


def _patch_end_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """tools/execution-end normalizes its payload; the returned result stays correct."""
    original = EventBus.emit

    def emit(self: EventBus, name: str, *args: Any, **kwargs: Any) -> Any:
        if name == TOOLS_EXECUTION_END:
            args = tuple(_normalized(a) if isinstance(a, ToolResult) else a for a in args)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(EventBus, "emit", emit)


def _patch_encode(
    monkeypatch: pytest.MonkeyPatch, transform: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    original = derive.encode_message
    monkeypatch.setattr(runner, "encode_message", lambda message: transform(original(message)))


def _strict_json_log(encoded: dict[str, Any]) -> dict[str, Any]:
    """A log whose storage is strict UTF-8 JSON: an unpaired surrogate cannot be stored."""
    json.dumps(encoded, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return encoded


MUTANTS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], str]] = {
    "tool-return-fffd": (_patch_finalize_input, "hook"),
    "details-dropped": (
        lambda m: _patch_message(m, lambda r: dataclasses.replace(r, details={})),
        "message",
    ),
    "hook-receives-fffd": (_patch_hook_input, "hook"),
    "execution-end-normalizes": (_patch_end_event, "execution_end"),
    "message-normalizes": (lambda m: _patch_message(m, _normalized), "message"),
    "message-text-only-normalized": (
        lambda m: _patch_message(m, lambda r: _normalized(r, details=None)),
        "message",
    ),
    "message-nested-details-normalized": (
        lambda m: _patch_message(
            m, lambda r: _normalized(r, content=False, details=lambda d: _map(d, min_depth=2))
        ),
        "message",
    ),
    "message-keys-replaced": (
        lambda m: _patch_message(
            m, lambda r: _normalized(r, content=False, details=lambda d: _map(d, strings=False))
        ),
        "message",
    ),
    "message-json-number-projection": (
        lambda m: _patch_message(
            m,
            lambda r: _normalized(
                r, content=False, details=lambda d: _map(d, strings=False, keys=False, numbers=True)
            ),
        ),
        "message",
    ),
    "session-strict-json-storage": (lambda m: _patch_encode(m, _strict_json_log), "raised"),
    "session-stores-fffd": (lambda m: _patch_encode(m, _map), "session"),
    "session-stores-pi-file-projection": (
        lambda m: _patch_encode(m, lambda e: _map(e, strings=False, keys=False, numbers=True)),
        "session",
    ),
    "session-reload-fffd": (
        lambda m: m.setattr(runner, "decode_message", lambda raw: derive.decode_message(_map(raw))),
        "session",
    ),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
async def test_a_wrong_implementation_fails_the_corpus_at_its_boundary(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install, boundary = MUTANTS[name]
    install(monkeypatch)
    failed = await _failures(tmp_path)
    assert failed, f"{name} survived the canonical corpus"
    assert any(b.startswith(boundary) for _, b in failed), (name, sorted({b for _, b in failed}))
