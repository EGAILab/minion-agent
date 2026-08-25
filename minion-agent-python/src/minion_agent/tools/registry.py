"""`ctx.tools`: the authoritative registry of executable tools and schemas.

Single ownership (design spec section 7). Request assembly obtains visible tool
schemas from here and nowhere else; `ctx.system_prompt` owns textual sections
only and never registers a schema.

Visibility follows section 3's scope rules -- nearest first, inheriting down --
and the registry shadows by name, because publishing two tools with one name
would leave the model unable to say which it meant.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..llm import ToolSchema
from ..runtime import Context, ScopedRegistry, ScopeKey, scope_of
from .definition import ToolDefinition


class ToolRegistry:
    """Every registered tool, filed by the scope that registered it."""

    __service_name__ = "tools"

    def __init__(self) -> None:
        self._entries: ScopedRegistry[ToolDefinition] = ScopedRegistry()

    def register(
        self, definition: ToolDefinition, *, scope: ScopeKey | None = None
    ) -> Callable[[], None]:
        """File `definition` under `scope`; returns a withdrawal handle."""
        return self._entries.add(scope, definition.name, definition)

    def visible_from(self, scope: ScopeKey | None = None) -> tuple[ToolDefinition, ...]:
        """Tools visible from `scope`, nearest first, one per name.

        `ScopedRegistry` returns nearest-scope-first and takes no position on
        collisions; shadowing is this registry's composition rule, applied by
        keeping the first entry seen for each name.
        """
        seen: dict[str, ToolDefinition] = {}
        for name, definition in self._entries.visible_from(scope):
            seen.setdefault(name, definition)
        return tuple(seen.values())

    def resolve(self, name: str, scope: ScopeKey | None = None) -> ToolDefinition | None:
        """The definition `name` refers to from `scope`, or None.

        None rather than an exception: an unknown call becomes an error result
        the model can read, and that decision belongs to the executor.
        """
        for definition in self.visible_from(scope):
            if definition.name == name:
                return definition
        return None

    def schemas(self, scope: ScopeKey | None = None) -> tuple[ToolSchema, ...]:
        """Model-facing schemas for every tool visible from `scope`."""
        return tuple(definition.schema() for definition in self.visible_from(scope))


def register_tool(ctx: Context, definition: ToolDefinition) -> Callable[[], Awaitable[None]]:
    """Register `definition` as a reversible effect of `ctx`.

    Registration is an effect, so unloading the owning plugin withdraws the
    tool mid-session and the next request's schemas omit it. The scope is the
    registering context's own, which keeps visibility and ownership on the same
    context -- section 3's governing rule.

    The returned withdrawal handle is `ctx.effect()`'s own handle -- both
    fiber-owned and scope-owned disposal are async by the certified Runtime's
    own design (`Fiber.effect`/`_scoped_effect` both return
    `Callable[[], Awaitable[None]]`), so callers MUST `await` it. Calling it
    synchronously silently creates an un-awaited coroutine and does not
    withdraw the registration (`TOOL-F006`) -- `ToolRegistry.register()`'s own
    return type, `Callable[[], None]`, is accurate only for that lower-level
    method called directly; it is not what this function actually returns.
    """
    scope = scope_of(ctx)
    registry: ToolRegistry = ctx.tools
    return ctx.effect(
        lambda: registry.register(definition, scope=scope), f"tool({definition.name})"
    )
