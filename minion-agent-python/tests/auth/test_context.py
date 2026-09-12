"""`DefaultAuthContext` behavior (`PROV-006`)."""

import os
from pathlib import Path

from minion_agent.auth.context import DefaultAuthContext


async def test_env_returns_a_set_value() -> None:
    ctx = DefaultAuthContext()
    os.environ["MINION_AUTH_TEST_VAR"] = "value"
    try:
        assert await ctx.env("MINION_AUTH_TEST_VAR") == "value"
    finally:
        del os.environ["MINION_AUTH_TEST_VAR"]


async def test_env_treats_a_missing_variable_as_none() -> None:
    ctx = DefaultAuthContext()
    assert await ctx.env("MINION_AUTH_TEST_VAR_DOES_NOT_EXIST") is None


async def test_env_treats_a_whitespace_only_value_as_absent() -> None:
    """Matches Pi's own `defaultProviderAuthContext` exactly (`auth/context.ts:25-28`)."""
    ctx = DefaultAuthContext()
    os.environ["MINION_AUTH_TEST_BLANK"] = "   "
    try:
        assert await ctx.env("MINION_AUTH_TEST_BLANK") is None
    finally:
        del os.environ["MINION_AUTH_TEST_BLANK"]


async def test_file_exists_true_for_a_real_file(tmp_path: Path) -> None:
    ctx = DefaultAuthContext()
    real_file = tmp_path / "exists.txt"
    real_file.write_text("x", encoding="utf-8")
    assert await ctx.file_exists(str(real_file)) is True


async def test_file_exists_false_for_a_missing_file(tmp_path: Path) -> None:
    ctx = DefaultAuthContext()
    assert await ctx.file_exists(str(tmp_path / "does-not-exist.txt")) is False


async def test_file_exists_expands_a_leading_tilde() -> None:
    ctx = DefaultAuthContext()
    home_relative = os.path.expanduser("~")
    assert await ctx.file_exists(home_relative) is True
