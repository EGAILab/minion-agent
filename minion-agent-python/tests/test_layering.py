"""Layer purity, checked rather than assumed.

Checks real imports via the AST rather than grepping source text: a docstring
may legitimately *mention* a higher layer to explain a contract, and a rule
about dependencies should not be tripped by prose.
"""

import ast
from pathlib import Path

import minion_agent

FORBIDDEN = {
    "runtime": ("llm", "session", "telemetry", "agent", "tools"),
    "llm": ("session", "agent", "tools"),
    "session": ("agent", "tools"),
    "telemetry": ("session", "agent", "tools"),
}

ROOT = Path(minion_agent.__file__).parent


def _imported_packages(module: Path) -> set[str]:
    """Every `minion_agent.<package>` this module imports, absolute or relative."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[:1] == ["minion_agent"] and len(parts) > 1:
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                # `from ..llm.content import X` inside a package module.
                found.add(node.module.split(".")[0])
            elif node.module:
                parts = node.module.split(".")
                if parts[:1] == ["minion_agent"] and len(parts) > 1:
                    found.add(parts[1])

    return found


def test_no_package_imports_a_higher_layer() -> None:
    """Only upward dependencies are forbidden; importing `..runtime` is
    downward and expected of every plugin module."""
    offenders: list[str] = []

    for package, forbidden in FORBIDDEN.items():
        for module in sorted((ROOT / package).rglob("*.py")):
            imported = _imported_packages(module)
            offenders.extend(
                f"{package}/{module.name} imports {name}"
                for name in sorted(imported & set(forbidden))
            )

    assert not offenders, "; ".join(offenders)


def test_llm_does_not_depend_on_sessions() -> None:
    """Stated separately because it is the boundary most likely to erode:
    the session layer stores messages, so the pull is toward the reverse."""
    for module in sorted((ROOT / "llm").rglob("*.py")):
        assert "session" not in _imported_packages(module), module.name


def test_the_check_would_catch_a_real_violation() -> None:
    """A layering test that cannot fail is decoration, so prove it detects one."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        offender = Path(directory) / "offender.py"
        offender.write_text(
            "from ..session.log import SessionLog\nfrom minion_agent.tools import Registry\n",
            encoding="utf-8",
        )

        assert _imported_packages(offender) == {"session", "tools"}


def test_every_package_surface_resolves() -> None:
    import minion_agent.llm as llm
    import minion_agent.runtime as runtime
    import minion_agent.session as session
    import minion_agent.telemetry as telemetry

    for package in (runtime, llm, session, telemetry):
        for name in package.__all__:
            assert getattr(package, name) is not None
