"""Harness/schema-integrity tests for the tool-registry runner's own reference validation.

`L05-R004`: these prove the runner rejects malformed *fixture* references (a `scope_parent`,
query `scope`, or step plugin id naming something no `plugins[]` entry declares) directly, before
any Runtime side effect runs -- not that the real `ToolRegistry`/`Context`/`Scope` seam produces
some particular visibility/shadowing/ordering outcome. That is what the canonical `conformance/
agent/*.yaml` scenarios in `test_tool_registry_conformance.py` prove; these are deliberately not
canonical scenarios themselves; they exercise the runner module directly with hand-built,
schema-independent documents.
"""

from typing import Any

import pytest

from .tool_registry_runner import run_tool_registry_scenario


def _tool(name: str = "t") -> dict[str, Any]:
    return {
        "name": name,
        "description": "d",
        "label": "T",
        "parameters": {"type": "object", "properties": {}},
    }


def _document(
    plugins: list[dict[str, Any]], steps: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> dict[str, Any]:
    return {"tool_registry": {"plugins": plugins, "steps": steps, "queries": queries}}


async def test_unknown_scope_parent_rejected_before_runtime_side_effects() -> None:
    document = _document(
        plugins=[
            {"id": "p", "scope": "child", "scope_parent": "missing_parent", "tools": [_tool()]}
        ],
        steps=[{"mount": "p"}],
        queries=[{"id": "q"}],
    )
    with pytest.raises(ValueError, match="missing_parent"):
        await run_tool_registry_scenario(document)


async def test_unknown_query_scope_rejected() -> None:
    document = _document(
        plugins=[{"id": "p", "tools": [_tool()]}],
        steps=[{"mount": "p"}],
        queries=[{"id": "q", "scope": "nonexistent"}],
    )
    with pytest.raises(ValueError, match="nonexistent"):
        await run_tool_registry_scenario(document)


async def test_self_parent_rejected() -> None:
    document = _document(
        plugins=[{"id": "p", "scope": "s", "scope_parent": "s", "tools": [_tool()]}],
        steps=[{"mount": "p"}],
        queries=[{"id": "q"}],
    )
    with pytest.raises(ValueError, match="own scope_parent"):
        await run_tool_registry_scenario(document)


async def test_scope_parent_cycle_rejected() -> None:
    document = _document(
        plugins=[
            {"id": "p1", "scope": "a", "scope_parent": "b", "tools": [_tool("t1")]},
            {"id": "p2", "scope": "b", "scope_parent": "a", "tools": [_tool("t2")]},
        ],
        steps=[{"mount": "p1"}, {"mount": "p2"}],
        queries=[{"id": "q"}],
    )
    with pytest.raises(ValueError, match="cycle"):
        await run_tool_registry_scenario(document)


async def test_unknown_mount_plugin_rejected() -> None:
    document = _document(plugins=[], steps=[{"mount": "missing"}], queries=[{"id": "q"}])
    with pytest.raises(ValueError, match="missing"):
        await run_tool_registry_scenario(document)


async def test_unknown_dispose_scope_rejected() -> None:
    document = _document(
        plugins=[{"id": "p", "tools": [_tool()]}],
        steps=[{"mount": "p"}, {"dispose_scope": "missing"}],
        queries=[{"id": "q"}],
    )
    with pytest.raises(ValueError, match="missing"):
        await run_tool_registry_scenario(document)


async def test_valid_parent_graph_proceeds_into_real_runtime() -> None:
    document = _document(
        plugins=[
            {"id": "root", "scope": "parent", "tools": [_tool("root_tool")]},
            {
                "id": "child",
                "scope": "child",
                "scope_parent": "parent",
                "tools": [_tool("child_tool")],
            },
        ],
        steps=[{"mount": "root"}, {"mount": "child"}],
        queries=[{"id": "from_child", "scope": "child"}],
    )
    observed = await run_tool_registry_scenario(document)
    assert observed["from_child"]["names"] == ["child_tool", "root_tool"]
