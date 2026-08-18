"""Plugin declaration: the decorator, config models, and spec resolution."""

import pytest
from pydantic import BaseModel

from minion_agent.runtime.plugin import PluginSpec, plugin, spec_of


class SampleConfig(BaseModel):
    timeout_ms: int = 120_000


def test_decorator_attaches_a_spec() -> None:
    @plugin(name="sample", inject=["tools"], config=SampleConfig, provides="sample")
    async def sample(ctx, config):  # noqa: ANN001, ARG001
        return None

    spec = spec_of(sample)

    assert spec.name == "sample"
    assert spec.inject == ("tools",)
    assert spec.config_model is SampleConfig
    assert spec.provides == "sample"


def test_bare_async_function_resolves_to_a_spec() -> None:
    async def bare(ctx, config):  # noqa: ANN001, ARG001
        return None

    spec = spec_of(bare)

    assert spec.name == "bare"
    assert spec.inject == ()
    assert spec.config_model is None


def test_object_with_apply_resolves_to_a_spec() -> None:
    class Mounted:
        name = "mounted"

        async def apply(self, ctx, config):  # noqa: ANN001, ARG001
            return None

    spec = spec_of(Mounted())

    assert spec.name == "mounted"


def test_non_plugin_raises() -> None:
    with pytest.raises(TypeError, match="not a plugin"):
        spec_of(42)


def test_config_model_validates_and_applies_defaults() -> None:
    @plugin(name="sample", config=SampleConfig)
    async def sample(ctx, config):  # noqa: ANN001, ARG001
        return None

    spec = spec_of(sample)
    assert spec.config_model is not None

    validated = spec.config_model.model_validate({})
    assert validated.timeout_ms == 120_000


def test_spec_is_frozen() -> None:
    spec = PluginSpec(
        name="frozen",
        apply=lambda ctx, config: None,
        inject=(),
        config_model=None,
        provides=None,
    )

    with pytest.raises(Exception):  # noqa: B017 - pydantic/dataclass both raise here
        spec.name = "changed"  # type: ignore[misc]


def test_decorated_function_stays_directly_callable() -> None:
    """The decorator returns the function unchanged, so it remains testable
    without going through the runtime."""
    calls: list[str] = []

    @plugin(name="sample")
    async def sample(ctx, config):  # noqa: ANN001, ARG001
        calls.append("ran")

    import asyncio

    asyncio.run(sample(None, None))

    assert calls == ["ran"]
