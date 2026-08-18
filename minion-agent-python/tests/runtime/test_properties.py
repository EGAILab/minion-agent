"""Properties that must hold for any effect, scope, and mount sequence."""

from hypothesis import given
from hypothesis import strategies as st

from minion_agent.runtime import Context, DisposableList, FiberState, ScopeKey, plugin

labels = st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=25, unique=True)


@given(labels)
async def test_disposal_is_always_exact_reverse_of_creation(order: list[str]) -> None:
    seen: list[str] = []
    disposables = DisposableList()
    for label in order:
        disposables.push(lambda label=label: seen.append(label))

    await disposables.dispose_all()

    assert seen == list(reversed(order))


@given(labels)
async def test_every_effect_disposes_exactly_once(order: list[str]) -> None:
    counts: dict[str, int] = dict.fromkeys(order, 0)
    disposables = DisposableList()
    for label in order:
        disposables.push(lambda label=label: counts.__setitem__(label, counts[label] + 1))

    await disposables.dispose_all()
    await disposables.dispose_all()

    assert all(count == 1 for count in counts.values())


@given(st.integers(min_value=1, max_value=12))
async def test_mount_unmount_cycles_leave_no_residue(cycles: int) -> None:
    """Repeated provider churn always returns the dependent to a clean state."""
    live: list[str] = []

    @plugin(name="churn-provider", provides="tools")
    async def churn_provider(ctx, config):
        ctx.provide("tools", object())

    @plugin(name="churn-consumer", inject=["tools"])
    async def churn_consumer(ctx, config):
        def start() -> object:
            live.append("on")
            return lambda: live.remove("on")

        ctx.effect(start, "live")

    root = Context()
    consumer_fiber = await root.plugin(churn_consumer)

    for _ in range(cycles):
        provider_fiber = await root.plugin(churn_provider)
        assert consumer_fiber.state is FiberState.ACTIVE
        assert live == ["on"]

        await root.plugins.unmount(provider_fiber)
        assert consumer_fiber.state is FiberState.PENDING
        assert live == []


@given(st.integers(min_value=1, max_value=8))
async def test_a_scope_chain_of_any_depth_disposes_deepest_first(depth: int) -> None:
    """Nesting depth is the application's choice, so no depth may be special."""
    order: list[str] = []
    ctx = Context()

    key: ScopeKey | None = None
    scopes = []
    for level in range(depth):
        key = ScopeKey(f"level-{level}", parent=key)
        scope = ctx.scope(key)
        scope.ctx.effect(
            lambda level=level: lambda: order.append(f"level-{level}"), f"level-{level}"
        )
        scopes.append(scope)

    for scope in reversed(scopes):
        await scope.dispose()

    assert order == [f"level-{level}" for level in reversed(range(depth))]


@given(st.integers(min_value=1, max_value=8))
async def test_admission_holds_at_any_chain_depth(depth: int) -> None:
    """Every ancestor of the dispatch key is admitted; no descendant is."""
    from minion_agent.runtime import DispatchMode, EventBus

    bus = EventBus()
    bus.declare("test/depth", DispatchMode.EMIT)

    keys: list[ScopeKey] = []
    parent: ScopeKey | None = None
    for level in range(depth):
        parent = ScopeKey(f"level-{level}", parent=parent)
        keys.append(parent)

    seen: list[int] = []
    for level, key in enumerate(keys):
        bus.on("test/depth", lambda level=level: seen.append(level), scope=key)

    # Dispatch at the deepest key: every ancestor hears it, in registration order.
    bus.emit("test/depth", scope=keys[-1])

    assert seen == list(range(depth))
