"""`DefaultAuthContext` behavior (`PROV-006`)."""

import os
from pathlib import Path

from minion_agent.auth.context import DefaultAuthContext


class _BlankPassthroughAuthContext:
    """A conforming `AuthContext` that does NOT normalize a blank value to absent -- proving that
    behavior is `DefaultAuthContext`'s own, not a protocol-level requirement (`L11-R003`)."""

    async def env(self, name: str) -> str | None:
        return ""

    async def file_exists(self, path: str) -> bool:
        return False


async def test_a_custom_context_may_return_a_blank_value_unchanged() -> None:
    """`L11-R003`: Pi's own `AuthContext` interface permits ANY implementation to resolve a
    present-but-blank value as-is -- only `DefaultAuthContext` additionally normalizes it."""
    ctx = _BlankPassthroughAuthContext()
    assert await ctx.env("TOKEN") == ""


class _SecondNativeAuthContext:
    """A second, independently-written `AuthContext` implementation (deliberately NOT
    `DefaultAuthContext`) -- proving leading-`~` support is a PROTOCOL-level expectation every
    conforming implementation follows, not merely `DefaultAuthContext`'s own default behavior
    (`L11-R008`; Pi's own interface doc comment states this directly on `fileExists` itself,
    `types.ts:99-100`, unlike `env`'s blank-normalization, which is default-implementation-only)."""

    async def env(self, name: str) -> str | None:
        return os.environ.get(name)

    async def file_exists(self, path: str) -> bool:
        resolved = os.path.expanduser(path)
        return Path(resolved).exists()


async def test_w_r008_1_a_second_native_auth_context_also_supports_leading_tilde() -> None:
    """`L11-R008`: leading-`~` support belongs to the `AuthContext` protocol itself, so a SEPARATE,
    independently-written implementation must interpret it as the user's home directory too, not
    only `DefaultAuthContext` -- this witness is deliberately distinct from
    `test_file_exists_expands_a_leading_tilde` below, which only exercises the default."""
    ctx = _SecondNativeAuthContext()
    home_relative = os.path.expanduser("~")
    assert await ctx.file_exists(home_relative) is True


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
