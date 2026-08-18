"""The services compose on the runtime as ordinary plugins."""

from minion_agent.llm.plugin import llm_plugin, mock_adapter_plugin
from minion_agent.llm.service import ModelId, Request
from minion_agent.llm.stream import collect
from minion_agent.runtime import Context, FiberState
from minion_agent.session.service import session_plugin
from minion_agent.telemetry.plugin import telemetry_plugin
from minion_agent.telemetry.spans import Span, SpanKind


async def _composed() -> Context:
    ctx = Context()
    await ctx.plugin(llm_plugin)
    await ctx.plugin(session_plugin)
    await ctx.plugin(telemetry_plugin)
    return ctx


async def test_the_services_resolve_by_name() -> None:
    ctx = await _composed()

    assert ctx.llm is not None
    assert ctx.sessions is not None
    assert ctx.telemetry is not None


async def test_an_adapter_plugin_waits_for_the_llm_service() -> None:
    """Reactive dependency, doing its job across package boundaries."""
    ctx = Context()

    adapter_fiber = await ctx.plugin(mock_adapter_plugin, {"script": []})
    assert adapter_fiber.state is FiberState.PENDING

    await ctx.plugin(llm_plugin)
    assert adapter_fiber.state is FiberState.ACTIVE


async def test_a_mounted_adapter_serves_requests() -> None:
    ctx = await _composed()
    await ctx.plugin(mock_adapter_plugin, {"script": [{"text": "hello", "stop_reason": "stop"}]})

    message = await collect(
        ctx.llm.stream(Request(model=ModelId("mock", "mock-1"), system="", messages=()))
    )

    assert message.content[0].text == "hello"


async def test_unmounting_the_adapter_withdraws_its_models() -> None:
    ctx = await _composed()
    fiber = await ctx.plugin(mock_adapter_plugin, {"script": []})
    assert ctx.llm.models()

    await ctx.plugins.unmount(fiber)

    assert ctx.llm.models() == frozenset()


async def test_sessions_are_created_and_retrieved_by_id() -> None:
    ctx = await _composed()

    created = ctx.sessions.create("s1")

    assert ctx.sessions.get("s1") is created
    assert ctx.sessions.get("missing") is None


async def test_the_session_service_forks_by_id() -> None:
    ctx = await _composed()
    ctx.sessions.create("s1")

    child = ctx.sessions.fork("s1", "s2")

    assert ctx.sessions.get("s2") is child
    assert child.ancestor is ctx.sessions.get("s1")


async def test_the_session_service_shares_one_artifact_store() -> None:
    """Content addressing only pays off if the store is shared."""
    ctx = await _composed()
    ctx.sessions.create("s1")
    ctx.sessions.create("s2")

    first = ctx.sessions.artifacts.put("shared block")
    second = ctx.sessions.artifacts.put("shared block")

    assert first == second
    assert len(ctx.sessions.artifacts) == 1


async def test_telemetry_records_by_default() -> None:
    ctx = await _composed()

    ctx.telemetry.emit(Span(kind=SpanKind.STEP, name="step"))

    assert ctx.telemetry.recording is not None
    assert len(ctx.telemetry.recording.spans) == 1
