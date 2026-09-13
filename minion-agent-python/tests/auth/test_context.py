"""`DefaultAuthContext` behavior (`PROV-006`)."""

import os
from pathlib import Path

import pytest

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


class _NonExpandingAuthContext:
    """A NEGATIVE CONTROL: a conforming-shaped `AuthContext` that does NOT expand a leading `~` --
    included only to prove the positive `W-R008-1` witness below actually discriminates (the
    targeted-closure review's own finding: a witness passing an ALREADY-EXPANDED path cannot tell
    a compliant implementation apart from this non-compliant one, since both would report the same
    -- correct-looking -- result for that input)."""

    async def env(self, name: str) -> str | None:
        return os.environ.get(name)

    async def file_exists(self, path: str) -> bool:
        return Path(path).exists()  # deliberately no expanduser() call


async def test_w_r008_1_a_second_native_auth_context_also_supports_leading_tilde() -> None:
    """`L11-R008` (targeted-closure finding): the witness must pass a LITERAL leading-`~` input
    (not a pre-expanded absolute path, which cannot discriminate expansion from a literal-path
    bug) -- a SEPARATE, independently-written implementation must still interpret it as the
    user's home directory, not only `DefaultAuthContext`."""
    ctx = _SecondNativeAuthContext()
    assert await ctx.file_exists("~") is True


async def test_w_r008_1_negative_control_a_non_expanding_context_fails_the_same_literal_input() -> (
    None
):
    """Proves the witness above actually discriminates: an implementation that does NOT expand a
    leading `~` reports the literal path `"~"` as nonexistent -- exactly the contract violation
    `L11-R008` exists to catch, and exactly what the ORIGINAL (pre-expanded) witness could not
    detect."""
    ctx = _NonExpandingAuthContext()
    assert await ctx.file_exists("~") is False


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
    """Passes the LITERAL `"~"` string, not a pre-expanded absolute path -- an already-expanded
    input cannot discriminate real expansion from a literal-relative-path bug (`L11-R008`'s own
    finding against a similarly-shaped witness for `_SecondNativeAuthContext`, above)."""
    ctx = DefaultAuthContext()
    assert await ctx.file_exists("~") is True


async def test_file_exists_naive_leading_tilde_concatenation_for_a_nontrivial_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`L11-R013`: Pi's own tilde expansion is LITERAL STRING CONCATENATION (`homedir() +
    path.slice(1)`), not path-join or `~username`-lookup semantics -- `~suffix` (no separator
    after the `~`) must resolve to `<homedir>suffix` (the home directory's own string with
    `suffix` appended directly), never `<homedir>/suffix` or a `~username` home-directory lookup.
    An injected, known home directory proves this precisely -- `Path(path).expanduser()`'s own
    differing platform conventions (a POSIX `~username` lookup, or Windows' own differing
    interpretation) would not produce this exact result."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(fake_home) if p == "~" else p)

    concatenated_marker = Path(str(fake_home) + "-marker")
    concatenated_marker.mkdir()

    ctx = DefaultAuthContext()
    assert await ctx.file_exists("~-marker") is True


async def test_file_exists_returns_false_on_a_filesystem_access_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`L11-R013`: Pi's own `fileExists` wraps the ENTIRE operation -- module/path resolution and
    the filesystem access itself -- in one failure boundary that returns `False` on ANY error
    (`try { ... } catch { return false; }`), not only "the target does not exist." A permission
    error or other filesystem failure must report `False`, never propagate the exception."""

    def _raise(self: Path) -> bool:
        raise OSError("denied")

    monkeypatch.setattr(Path, "exists", _raise)

    ctx = DefaultAuthContext()
    assert await ctx.file_exists(str(tmp_path)) is False


async def test_file_exists_returns_false_on_a_home_resolution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`L11-R013` (second remediation, targeted-closure-round-2 review): Pi's own `try` encloses
    HOME-DIRECTORY RESOLUTION too, not only the filesystem access afterward -- a first remediation
    left `os.path.expanduser("~")` OUTSIDE its own `try` block, so a failure there still propagated
    uncaught instead of resolving `False`."""

    def _raise(path: str) -> str:
        raise OSError("home unavailable")

    monkeypatch.setattr(os.path, "expanduser", _raise)

    ctx = DefaultAuthContext()
    assert await ctx.file_exists("~") is False


async def test_file_exists_returns_false_on_a_non_os_error_during_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`L11-R013` (second remediation): Pi's own bare `catch {}` filters on nothing -- a first
    remediation narrowed the catch to `OSError` alone, so a non-`OSError` failure (e.g. a
    `RuntimeError`) still propagated uncaught instead of resolving `False`."""

    def _raise(self: Path) -> bool:
        raise RuntimeError("non-os failure")

    monkeypatch.setattr(Path, "exists", _raise)

    ctx = DefaultAuthContext()
    assert await ctx.file_exists(str(tmp_path)) is False
