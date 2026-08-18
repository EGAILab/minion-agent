"""The package imports and reports a version."""

import minion_agent


def test_package_exposes_version() -> None:
    assert isinstance(minion_agent.__version__, str)
    assert minion_agent.__version__
