"""The runtime package exposes a curated public surface."""

import minion_agent.runtime as runtime

EXPECTED = {
    "Context",
    "DisposableList",
    "DispatchMode",
    "Disposer",
    "EventBus",
    "EventModeError",
    "Fiber",
    "FiberState",
    "Impl",
    "InactiveFiberError",
    "PluginRegistry",
    "PluginSpec",
    "RuntimeError_",
    "Scope",
    "ScopeKey",
    "ScopedRegistry",
    "ServiceConflictError",
    "ServiceNotFoundError",
    "ServiceRegistry",
    "WaterfallError",
    "plugin",
    "scope_of",
    "spec_of",
}


def test_all_matches_expected_surface() -> None:
    assert set(runtime.__all__) == EXPECTED


def test_every_exported_name_resolves() -> None:
    for name in runtime.__all__:
        assert getattr(runtime, name) is not None


def test_no_cordis_in_public_identifiers() -> None:
    """Cordis is design lineage, not API vocabulary. See spec section 3."""
    for name in runtime.__all__:
        assert "cordis" not in name.lower()


def test_runtime_does_not_import_higher_layers() -> None:
    """Layer purity: the runtime knows nothing of tools, agents, or sessions.

    Checked rather than assumed, because the one time this rule was broken it
    was broken in prose that read perfectly reasonably.
    """
    from pathlib import Path

    package = Path(runtime.__file__).parent
    forbidden = ("minion_agent.tools", "minion_agent.agent", "minion_agent.session")

    offenders: list[str] = []
    for module in sorted(package.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        offenders.extend(f"{module.name} imports {name}" for name in forbidden if name in source)

    assert not offenders, "; ".join(offenders)
