"""Executes `conformance/agent/*.yaml` tool-registry (Layer 05) scenarios.

Drives the real `ToolRegistry`/`Context`/scope/effect seam directly -- mounts the real
`tools_plugin`, registers real `ToolDefinition`s through the real `register_tool()` effect,
disposes real scopes and unmounts real plugin fibers, and queries `visible_from`/`resolve`/
`schemas` on the real registry. This module implements no visibility, shadowing, or withdrawal
logic itself; that is the library's job (design spec section 7).
"""

from __future__ import annotations

from typing import Any

from minion_agent.llm import (
    ConstrainedSampling,
    GrammarConstrainedSampling,
    JsonSchemaConstrainedSampling,
)
from minion_agent.runtime import Context, PluginSpec, Scope, ScopeKey
from minion_agent.runtime.fiber import Fiber
from minion_agent.runtime.plugin import spec_of
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.plugin import tools_plugin
from minion_agent.tools.registry import ToolRegistry, register_tool


def _constrained_sampling(raw: Any) -> ConstrainedSampling | bool | None:
    if raw is None or raw is False:
        return raw
    if raw["type"] == "json_schema":
        return JsonSchemaConstrainedSampling(strict=raw["strict"])
    variants = raw["variants"]
    return GrammarConstrainedSampling(
        openai_lark=variants.get("openai_lark"), openai_regex=variants.get("openai_regex")
    )


def _tool_definition(spec: dict[str, Any]) -> ToolDefinition:
    return ToolDefinition(
        name=spec["name"],
        description=spec["description"],
        parameters=spec["parameters"],  # a raw, object-valued JSON Schema dict (L05-R005/TOOL-F010)
        execute=lambda args: "ok",  # never invoked -- Layer 05 registry scenarios never execute
        label=spec["label"],
        constrained_sampling=_constrained_sampling(spec.get("constrained_sampling")),
    )


def _schema_as_dict(definition: ToolDefinition) -> dict[str, Any]:
    return definition.schema().as_json()


class _ScopeTable:
    """Real `ScopeKey`/`Scope` objects, keyed by the plain names scenarios use.

    Rejects unresolved parent/query references explicitly (`L05-R004`): a `scope_parent` or
    query `scope` naming a scope no earlier `plugins[]` entry actually created is malformed
    canonical input, not "no parent"/"untagged" -- validating this reference is the runner's own
    input-validation boundary, not a simulation of tool-registry visibility semantics.
    """

    def __init__(self) -> None:
        self.keys: dict[str, ScopeKey] = {}
        self.live: dict[str, Scope] = {}

    def key_for(self, name: str, parent_name: str | None) -> ScopeKey:
        if name not in self.keys:
            parent = None
            if parent_name is not None:
                if parent_name not in self.keys:
                    raise ValueError(
                        f"scope {name!r} declares scope_parent {parent_name!r}, which no earlier "
                        "plugins[] entry created -- malformed canonical input"
                    )
                parent = self.keys[parent_name]
            self.keys[name] = ScopeKey(name, parent=parent)
        return self.keys[name]

    def open(self, ctx: Context, name: str, parent_name: str | None) -> Scope:
        existing = self.live.get(name)
        if existing is not None and not existing.disposed:
            return existing
        scope = ctx.scope(self.key_for(name, parent_name))
        self.live[name] = scope
        return scope

    def require_live(self, name: str) -> Scope:
        """The real `Scope` a query's `scope:` name refers to -- raises if that name was never
        created by any `plugins[]` entry, rather than silently treating it as untagged."""
        if name not in self.live:
            raise ValueError(
                f"query references scope {name!r}, which no plugins[] entry ever created"
            )
        return self.live[name]


def _validate_references(spec_doc: dict[str, Any]) -> None:
    """Reject malformed declarative references before any Runtime object is constructed
    (`L05-R004`): a plugin's `apply()` raising during `reconcile()` is recorded as a fiber
    failure and reconciliation continues, so a `scope_parent`/query `scope` naming a
    never-declared scope previously surfaced only as an incidental `KeyError` later, not the
    direct rejection this boundary promises. This validates the scenario description itself
    (structural fixture integrity), not registry lookup/shadowing semantics -- it never decides
    visibility, ancestry, or ordering; the real `ToolRegistry`/`Context`/`Scope` seam still owns
    all of that once construction begins.
    """
    plugin_ids = {entry["id"] for entry in spec_doc["plugins"]}
    scope_parents: dict[str, str | None] = {
        entry["scope"]: entry.get("scope_parent")
        for entry in spec_doc["plugins"]
        if entry.get("scope") is not None
    }

    for scope_name, parent_name in scope_parents.items():
        if parent_name is None:
            continue
        if parent_name == scope_name:
            raise ValueError(f"scope {scope_name!r} declares itself as its own scope_parent")
        if parent_name not in scope_parents:
            raise ValueError(
                f"scope {scope_name!r} declares scope_parent {parent_name!r}, which no "
                "plugins[] entry declares as its own scope -- malformed canonical input"
            )

    for scope_name in scope_parents:
        seen = {scope_name}
        current = scope_parents[scope_name]
        while current is not None:
            if current in seen:
                raise ValueError(
                    f"scope {scope_name!r}'s scope_parent chain contains a cycle at {current!r}"
                )
            seen.add(current)
            current = scope_parents[current]

    for step in spec_doc["steps"]:
        for key in ("mount", "unmount", "withdraw"):
            if key in step and step[key] not in plugin_ids:
                raise ValueError(
                    f"step {key!r} references plugin {step[key]!r}, which no plugins[] entry "
                    "declares -- malformed canonical input"
                )
        if "dispose_scope" in step and step["dispose_scope"] not in scope_parents:
            raise ValueError(
                f"step dispose_scope references scope {step['dispose_scope']!r}, which no "
                "plugins[] entry declares -- malformed canonical input"
            )

    for query in spec_doc["queries"]:
        scope_name = query.get("scope")
        if scope_name is not None and scope_name not in scope_parents:
            raise ValueError(
                f"query {query['id']!r} references scope {scope_name!r}, which no plugins[] "
                "entry declares -- malformed canonical input"
            )


async def run_tool_registry_scenario(document: dict[str, Any]) -> dict[str, Any]:
    """Run one `tool_registry` scenario and return `{query_id: {...}}` observations."""
    spec_doc = document["tool_registry"]
    _validate_references(spec_doc)
    root = Context()
    root.plugins.mount(spec_of(tools_plugin), None, root)
    await root.plugins.reconcile()

    scopes = _ScopeTable()
    plugin_entries = {entry["id"]: entry for entry in spec_doc["plugins"]}
    fibers: dict[str, Fiber] = {}
    handles: dict[str, list[Any]] = {}

    def make_apply(entry: dict[str, Any]):
        async def apply(ctx: Context, config: Any) -> None:
            target = ctx
            scope_name = entry.get("scope")
            if scope_name is not None:
                target = scopes.open(ctx, scope_name, entry.get("scope_parent")).ctx
            handles[entry["id"]] = [
                register_tool(target, _tool_definition(tool_spec)) for tool_spec in entry["tools"]
            ]

        return apply

    for step in spec_doc["steps"]:
        if "mount" in step:
            plugin_id = step["mount"]
            entry = plugin_entries[plugin_id]
            spec = PluginSpec(
                name=plugin_id, apply=make_apply(entry), inject=(), config_model=None, provides=None
            )
            fibers[plugin_id] = root.plugins.mount(spec, None, root)
            await root.plugins.reconcile()
        elif "unmount" in step:
            await root.plugins.unmount(fibers[step["unmount"]])
        elif "withdraw" in step:
            for handle in handles[step["withdraw"]]:
                await handle()
        elif "dispose_scope" in step:
            await scopes.live[step["dispose_scope"]].dispose()

    registry: ToolRegistry = root.tools
    observations: dict[str, Any] = {}
    for query in spec_doc["queries"]:
        # A real, possibly-disposed Scope object -- not a bare key -- so a disposed scope
        # correctly observes no visibility at all (L05-R002), and an unresolved scope name
        # fails loudly rather than silently meaning "untagged" (L05-R004).
        scope = scopes.require_live(query["scope"]) if query.get("scope") else None
        visible = registry.visible_from(scope)
        observation: dict[str, Any] = {
            "names": [definition.name for definition in visible],
            "schemas": [_schema_as_dict(definition) for definition in visible],
        }
        if "resolve" in query:
            resolved: dict[str, Any] = {}
            for name in query["resolve"]:
                found = registry.resolve(name, scope)
                resolved[name] = (
                    {"found": True, "label": found.label} if found is not None else {"found": False}
                )
            observation["resolve"] = resolved
        observations[query["id"]] = observation

    return observations
