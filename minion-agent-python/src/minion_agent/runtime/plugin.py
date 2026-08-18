"""Plugin declaration: a body, its injected services, and its config model."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from .context import Context

F = TypeVar("F", bound=Callable[..., Any])

SPEC_ATTRIBUTE = "__plugin_spec__"


@dataclass(frozen=True, slots=True)
class PluginSpec:
    """Everything the runtime needs to mount one plugin."""

    name: str
    apply: Callable[[Context, Any], Awaitable[None] | None]
    inject: tuple[str, ...]
    config_model: type[BaseModel] | None
    provides: str | None


def plugin(
    *,
    name: str,
    inject: Iterable[str] = (),
    config: type[BaseModel] | None = None,
    provides: str | None = None,
) -> Callable[[F], F]:
    """Declare a function as a plugin.

    The decorated function is returned unchanged with a `PluginSpec` attached,
    so it stays directly callable and testable.
    """

    def decorate(body: F) -> F:
        spec = PluginSpec(
            name=name,
            apply=body,
            inject=tuple(inject),
            config_model=config,
            provides=provides,
        )
        object.__setattr__(body, SPEC_ATTRIBUTE, spec)
        return body

    return decorate


def spec_of(candidate: Any) -> PluginSpec:
    """Resolve `candidate` to a PluginSpec.

    Accepts a decorated function, a bare callable taking `(ctx, config)`, or an
    object exposing `apply`.
    """
    existing = getattr(candidate, SPEC_ATTRIBUTE, None)
    if isinstance(existing, PluginSpec):
        return existing

    apply = getattr(candidate, "apply", None)
    if callable(apply):
        return PluginSpec(
            name=getattr(candidate, "name", type(candidate).__name__),
            apply=apply,
            inject=tuple(getattr(candidate, "inject", ())),
            config_model=getattr(candidate, "config_model", None),
            provides=getattr(candidate, "provides", None),
        )

    if callable(candidate) and not isinstance(candidate, type):
        return PluginSpec(
            name=getattr(candidate, "__name__", "<anonymous>"),
            apply=candidate,
            inject=(),
            config_model=None,
            provides=None,
        )

    raise TypeError(
        f"{candidate!r} is not a plugin: expected a callable taking (ctx, config) "
        "or an object with an 'apply' method"
    )
