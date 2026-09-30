"""WP-13.2 `write`/`edit`/mutation queue: unit witnesses for Pi guards the canonical scenarios
cannot reach, and the binding-level negative controls spec/tools.md WP-13.2 item 4 requires --
each a single-point mutant of the real source that its canonical witness must kill."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import types
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from minion_agent.execution import Err, FsError, FsErrorCode, LocalFileSystem, Ok
from minion_agent.execution.plugin import fs_plugin
from minion_agent.runtime import Context, RunAbortController
from minion_agent.runtime.fiber import FiberState
from minion_agent.tools.builtin import (
    create_edit_tool,
    create_write_tool,
    edit_diff,
    fs_mutation_tools_plugin,
    mutation_queue,
    prepare_edit_arguments,
)
from minion_agent.tools.builtin import edit as edit_module
from minion_agent.tools.builtin import write as write_module
from minion_agent.tools.builtin._jsdiff import create_two_files_patch, diff_lines
from minion_agent.tools.builtin._utf16 import from_units, to_units
from minion_agent.tools.builtin.collation import pinned_collation
from minion_agent.tools.builtin.paths import BuiltinToolError
from minion_agent.tools.plugin import tools_plugin
from minion_agent.tools.registry import ToolRegistry

from ...conformance import builtin_mutation_runner as runner
from ...conformance.test_builtin_mutation_conformance import (
    FUZZY_FIXTURE,
    SCENARIO_DIR,
    check_case,
    check_queue,
)


def _scenario(name: str) -> dict[str, Any]:
    return yaml.safe_load((SCENARIO_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


async def _passes(name: str) -> bool:
    """Whether the named canonical scenario passes against the (possibly mutated) implementation."""
    document = _scenario(name)
    try:
        if "queue" in document["builtin_mutation"]:
            run = await asyncio.wait_for(runner.run_queue(document), timeout=30)
            check_queue(run, document["builtin_mutation"]["queue"]["expect"])
        else:
            for outcome in await runner.run_cases(document):
                check_case(outcome)
    except AssertionError:
        return False
    return True


def _fuzzy_replay_passes() -> bool:
    cases = json.loads(FUZZY_FIXTURE.read_text(encoding="utf-8"))["cases"]
    return all(
        from_units(edit_diff.fuzzy_normalize(to_units(c["text"]))) == c["normalized"] for c in cases
    )


def _mutant(module: types.ModuleType, old: str, new: str) -> types.ModuleType:
    """A copy of `module` with exactly one source substitution, loaded as a sibling module."""
    source = inspect.getsource(module)
    assert source.count(old) == 1, old
    clone = types.ModuleType(module.__name__ + "_mutant")
    clone.__package__ = module.__package__
    sys.modules[clone.__name__] = clone
    exec(compile(source.replace(old, new), module.__name__ + "_mutant", "exec"), clone.__dict__)
    return clone


@pytest.fixture
def restore_modules() -> Iterator[None]:
    yield
    for name in [n for n in sys.modules if n.endswith("_mutant")]:
        del sys.modules[name]


# ---------------------------------------------------------------- guards unreachable via the tool
def test_identical_inputs_are_one_common_component() -> None:
    parts = diff_lines("a\nb\n", "a\nb\n")
    assert [(p.value, p.added, p.removed) for p in parts] == [("a\nb\n", False, False)]
    assert edit_diff.generate_diff_string("a\nb", "a\nb") == ("", None)


def test_zero_context_patch_has_no_leading_context() -> None:
    assert (
        create_two_files_patch("f", "f", "a\nb\n", "a\nc\n", 0)
        == "--- f\n+++ f\n@@ -2,1 +2,1 @@\n-b\n+c\n"
    )


def test_replacement_ending_past_the_base_is_internal_range() -> None:
    spans = [(0, 2)]
    with pytest.raises(
        BuiltinToolError, match=r"^Replacement range is outside the base content\.$"
    ):
        edit_diff._replacement_line_range(spans, edit_diff.Replacement(0, 1, 5, "x"))


def test_prepare_returns_a_non_object_unchanged() -> None:
    """L13-WP132-R005: unreachable through the object-valued ToolCall, reproduced when called."""
    assert prepare_edit_arguments("just a string") == "just a string"


def test_prepare_rejects_json_constants_js_does_not_parse() -> None:
    assert prepare_edit_arguments({"path": "f", "edits": "NaN"})["edits"] == "NaN"
    assert prepare_edit_arguments({"path": "f", "edits": "[1e999]"})["edits"] == [float("inf")]


async def test_plugin_registers_write_and_edit_over_the_mounted_fs() -> None:
    ctx = Context()
    await ctx.plugin(tools_plugin, None)
    await ctx.plugin(fs_plugin, None)
    fiber = await ctx.plugin(fs_mutation_tools_plugin, None)
    assert fiber.state is FiberState.ACTIVE
    registry: ToolRegistry = ctx.tools
    assert [schema.name for schema in registry.schemas()] == ["write", "edit"]
    await ctx.plugins.unmount(fiber)
    assert registry.resolve("write") is None and registry.resolve("edit") is None


class _InstantKeys:
    """A provider whose canonical_path answers without I/O, so a call reaches the queue wait."""

    async def canonical_path(self, path: str, signal: Any = None) -> Any:
        return Ok(path)


async def test_an_interrupted_waiter_releases_only_after_the_entry_ahead_of_it() -> None:
    fs: Any = _InstantKeys()
    first_holding = asyncio.Event()
    first_may_finish = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        first_holding.set()
        await first_may_finish.wait()
        order.append("first")

    async def second() -> None:
        order.append("second")  # pragma: no cover -- cancelled before it runs

    async def third() -> None:
        order.append("third")

    a = asyncio.ensure_future(mutation_queue.with_mutation_queue(fs, "f.txt", "f.txt", first))
    await first_holding.wait()
    b = asyncio.ensure_future(mutation_queue.with_mutation_queue(fs, "f.txt", "f.txt", second))
    c = asyncio.ensure_future(mutation_queue.with_mutation_queue(fs, "f.txt", "f.txt", third))
    for _ in range(50):
        await asyncio.sleep(0)
    b.cancel()
    for _ in range(50):
        await asyncio.sleep(0)
    assert order == []  # c is still behind a, even though b's wait was interrupted
    first_may_finish.set()
    await a
    await c
    assert b.cancelled() and order == ["first", "third"]


# ---------------------------------------------------------------- binding-level negative controls
async def test_every_witness_passes_unmutated() -> None:
    for name in (
        "builtin-edit-corpus-curated",
        "builtin-edit-error-sites",
        "builtin-mutation-queue-same-target-registers-in-call-order",
        "builtin-mutation-queue-slow-failing-registration-then-proceeds",
        "builtin-mutation-queue-providers-do-not-share-queues",
        "builtin-mutation-queue-aborted-call-holds-lock-until-its-write-settles",
        "builtin-mutation-queue-released-after-error-and-after-abort",
    ):
        assert await _passes(name), name
    assert _fuzzy_replay_passes()


def test_unicode_15_1_nfkc_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert unicodedata.unidata_version != "16.0.0"
    monkeypatch.setattr(
        edit_diff, "_nfkc", lambda units: to_units(unicodedata.normalize("NFKC", from_units(units)))
    )
    assert not _fuzzy_replay_passes()


def test_unicode_17_nfkc_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    import icu

    unfiltered = icu.Normalizer2.getNFKCInstance()
    monkeypatch.setattr(pinned_collation(), "_nfkc16", unfiltered)
    assert not _fuzzy_replay_passes()


async def test_native_whitespace_trim_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edit_diff, "_JS_WHITESPACE", None)  # str.rstrip(None): Python's own set
    assert not await _passes("builtin-edit-corpus-curated")


async def test_non_serialized_registration_is_killed(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    del restore_modules
    mutant = _mutant(
        mutation_queue,
        "        if previous_registration is not None:\n"
        "            await asyncio.shield(previous_registration)\n",
        "",
    )
    monkeypatch.setattr(write_module, "with_mutation_queue", mutant.with_mutation_queue)
    assert not await _passes("builtin-mutation-queue-same-target-registers-in-call-order")
    assert not await _passes("builtin-mutation-queue-slow-failing-registration-then-proceeds")


async def test_unsettled_failed_registration_is_killed(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    del restore_modules
    mutant = _mutant(
        mutation_queue,
        "    finally:\n"
        "        registered.set_result(None)"
        "  # settled either way: the next registration may begin\n",
        "    except BaseException:\n        raise\n    registered.set_result(None)\n",
    )
    monkeypatch.setattr(write_module, "with_mutation_queue", mutant.with_mutation_queue)
    assert not await _passes("builtin-mutation-queue-slow-failing-registration-then-proceeds")


async def test_queue_without_provider_scoping_is_killed(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    del restore_modules
    mutant = _mutant(
        mutation_queue,
        "scope = (id(fs), await mutation_queue_key(fs, p, path))",
        "scope = (0, await mutation_queue_key(fs, p, path))",
    )
    monkeypatch.setattr(write_module, "with_mutation_queue", mutant.with_mutation_queue)
    assert not await _passes("builtin-mutation-queue-providers-do-not-share-queues")


async def test_abort_listener_release_is_killed(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    del restore_modules
    mutant = _mutant(
        write_module,
        "text = await with_mutation_queue(fs, p, path, work)",
        "text = await with_mutation_queue(fs, p, path, lambda: race_abort(work(), signal))",
    )
    from minion_agent.tools.builtin._signal import race_abort

    mutant.race_abort = race_abort  # type: ignore[attr-defined]
    monkeypatch.setattr(runner, "create_write_tool", mutant.create_write_tool)
    assert not await _passes(
        "builtin-mutation-queue-aborted-call-holds-lock-until-its-write-settles"
    )


async def test_queue_wait_abort_listener_is_killed(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    """docs #188's control: the caller is answered while it still waits in the queue."""
    del restore_modules
    mutant = _mutant(
        write_module,
        "text = await with_mutation_queue(fs, p, path, work)",
        "text = await race_abort(with_mutation_queue(fs, p, path, work), signal)",
    )
    from minion_agent.tools.builtin._signal import race_abort

    mutant.race_abort = race_abort  # type: ignore[attr-defined]
    monkeypatch.setattr(runner, "create_write_tool", mutant.create_write_tool)
    assert not await _passes("builtin-mutation-queue-released-after-error-and-after-abort")


async def test_absolute_path_before_registration_is_killed(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    del restore_modules
    mutant = _mutant(
        write_module,
        "        text = await with_mutation_queue(fs, p, path, work)",
        "        await fs.absolute_path(p)\n"
        "        text = await with_mutation_queue(fs, p, path, work)",
    )
    monkeypatch.setattr(runner, "create_write_tool", mutant.create_write_tool)
    assert not await _passes("builtin-mutation-queue-same-target-registers-in-call-order")


async def test_two_probe_access_check_is_killed(
    monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    del restore_modules
    mutant = _mutant(
        edit_module,
        "    answer = await fs.check_read_write(p)\n",
        "    answer = await fs.check_readable(p)\n"
        "    if not isinstance(answer, Err):\n"
        "        answer = await fs.check_read_write(p)\n",
    )
    monkeypatch.setattr(runner, "create_edit_tool", mutant.create_edit_tool)
    assert not await _passes("builtin-edit-error-sites")


# ---------------------------------------------------------------- tool construction
def test_definitions_are_pinned_pi_strings() -> None:
    write = create_write_tool(LocalFileSystem("."))
    edit = create_edit_tool(LocalFileSystem("."))
    assert (write.name, write.label, edit.name, edit.label) == ("write", "write", "edit", "edit")
    assert write.description.startswith(
        "Write content to a file. Creates the file if it doesn't exist"
    )
    assert edit.prepare_arguments is prepare_edit_arguments
    assert edit.parameters["required"] == ["path", "edits"]  # type: ignore[index]
    assert "additionalProperties" not in json.dumps(edit.parameters)


async def test_fs_errors_carry_no_signal(tmp_path: Path) -> None:
    """No ctx.fs call receives the signal (TOOL-033): a provider that fails when given one."""

    class Strict(LocalFileSystem):
        async def write_file(self, path: str, content: Any, signal: Any = None) -> Any:
            if signal is not None:
                return Err(FsError(FsErrorCode.UNKNOWN, "signal passed", path))
            return await super().write_file(path, content)

    tool = create_write_tool(Strict(str(tmp_path)))
    result = await tool.execute(
        "id", {"path": "a.txt", "content": "x"}, RunAbortController().signal
    )
    assert result.content[0].text == "Successfully wrote 1 bytes to a.txt"  # type: ignore[union-attr]
