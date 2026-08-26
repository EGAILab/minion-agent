"""The registry: scope-aware visibility, and registration as an effect."""

from minion_agent.runtime import Context, ScopeKey, plugin
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.registry import ToolRegistry, register_tool


def _definition(name: str = "echo", description: str = "d") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
        execute=lambda args: "ok",
        label=name,
    )


def test_a_registered_tool_resolves() -> None:
    registry = ToolRegistry()
    registry.register(_definition())

    resolved = registry.resolve("echo")

    assert resolved is not None
    assert resolved.name == "echo"


def test_an_unregistered_name_resolves_to_none() -> None:
    """Not an exception: an unknown call becomes an error result, and the
    caller decides that, not the registry."""
    assert ToolRegistry().resolve("missing") is None


def test_withdrawing_removes_the_tool() -> None:
    registry = ToolRegistry()
    withdraw = registry.register(_definition())

    withdraw()

    assert registry.resolve("echo") is None


def test_schemas_describe_every_visible_tool() -> None:
    registry = ToolRegistry()
    registry.register(_definition("echo"))
    registry.register(_definition("read"))

    assert {schema.name for schema in registry.schemas()} == {"echo", "read"}


def test_an_untagged_tool_is_visible_from_every_scope() -> None:
    """Global registrations are the base layer every agent sees."""
    registry = ToolRegistry()
    registry.register(_definition())

    assert registry.resolve("echo", ScopeKey("agent-instance:a")) is not None


def test_a_scoped_tool_is_invisible_outside_its_scope() -> None:
    registry = ToolRegistry()
    room_a = ScopeKey("agent-instance:a")
    room_b = ScopeKey("agent-instance:b")
    registry.register(_definition("private"), scope=room_a)

    assert registry.resolve("private", room_a) is not None
    assert registry.resolve("private", room_b) is None
    assert registry.resolve("private") is None


def test_a_scoped_tool_is_visible_to_descendants() -> None:
    """Inherit-down: a definition-level tool belongs to all its instances."""
    definition_scope = ScopeKey("agent-definition:ada")
    instance_scope = ScopeKey("agent-instance:a", parent=definition_scope)
    registry = ToolRegistry()
    registry.register(_definition("shared"), scope=definition_scope)

    assert registry.resolve("shared", instance_scope) is not None


def test_a_nearer_scope_shadows_a_farther_one() -> None:
    definition_scope = ScopeKey("agent-definition:ada")
    instance_scope = ScopeKey("agent-instance:a", parent=definition_scope)
    registry = ToolRegistry()
    registry.register(_definition("read", "generic"), scope=definition_scope)
    registry.register(_definition("read", "specialised"), scope=instance_scope)

    resolved = registry.resolve("read", instance_scope)

    assert resolved is not None
    assert resolved.description == "specialised"


def test_shadowing_publishes_one_schema_per_name() -> None:
    """Sending the model two tools with the same name would be incoherent."""
    definition_scope = ScopeKey("agent-definition:ada")
    instance_scope = ScopeKey("agent-instance:a", parent=definition_scope)
    registry = ToolRegistry()
    registry.register(_definition("read", "generic"), scope=definition_scope)
    registry.register(_definition("read", "specialised"), scope=instance_scope)

    schemas = registry.schemas(instance_scope)

    assert [schema.description for schema in schemas] == ["specialised"]


async def test_registration_through_a_plugin_is_an_effect() -> None:
    """Unloading a plugin withdraws its tools mid-session, and the next
    request's schemas omit them (design spec section 7)."""
    ctx = Context()

    @plugin(name="tools", provides="tools")
    async def tools_provider(inner: Context, config: None) -> None:
        inner.provide("tools", ToolRegistry())

    @plugin(name="extra", inject=["tools"])
    async def extra(inner: Context, config: None) -> None:
        register_tool(inner, _definition("temporary"))

    await ctx.plugin(tools_provider, None)
    fiber = await ctx.plugin(extra, None)
    assert ctx.tools.resolve("temporary") is not None

    await ctx.plugins.unmount(fiber)

    assert ctx.tools.resolve("temporary") is None
