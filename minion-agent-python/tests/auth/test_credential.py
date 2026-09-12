"""Structural-shape tests for the auth vocabulary (`PROV-006`)."""

import pytest

from minion_agent.auth.credential import (
    ApiKeyCredential,
    AuthCheck,
    AuthOperationOptions,
    AuthResult,
    CredentialInfo,
    ModelAuth,
    OAuthCredential,
)


def test_api_key_credential_defaults_to_no_key_or_env() -> None:
    credential = ApiKeyCredential()
    assert credential.key is None
    assert credential.env is None
    assert credential.type == "api_key"


def test_api_key_credential_carries_key_and_env() -> None:
    credential = ApiKeyCredential(key="sk-test", env={"CF_ACCOUNT_ID": "abc"})
    assert credential.key == "sk-test"
    assert credential.env == {"CF_ACCOUNT_ID": "abc"}


def test_oauth_credential_requires_access_refresh_expires() -> None:
    credential = OAuthCredential(access="a", refresh="r", expires=1234.0)
    assert credential.type == "oauth"
    assert credential.extra == {}


def test_api_key_credential_env_is_immutable_and_not_aliased_to_the_constructor_argument() -> None:
    """`L11-R006`: Pi's own in-memory store holds mutable objects and exposes live references from
    `read`/`modify`; this project makes `Credential` a fully immutable VALUE instead (intentional
    divergence) -- a plain dict passed to the constructor must be snapshotted, not aliased, and the
    stored mapping itself must reject item assignment."""
    original = {"CF_ACCOUNT_ID": "abc"}
    credential = ApiKeyCredential(key="sk-test", env=original)

    original["CF_ACCOUNT_ID"] = "mutated-after-construction"

    assert credential.env is not None
    assert credential.env["CF_ACCOUNT_ID"] == "abc"
    with pytest.raises(TypeError):
        credential.env["CF_ACCOUNT_ID"] = "z"  # type: ignore[index]


def test_oauth_credential_extra_is_immutable_and_not_aliased_to_the_constructor_argument() -> None:
    """`L11-R006`, `OAuthCredential`'s own `extra` field -- same value-semantics guarantee as
    `ApiKeyCredential.env` above."""
    original = {"accountId": "acc_1"}
    credential = OAuthCredential(access="a", refresh="r", expires=1234.0, extra=original)

    original["accountId"] = "mutated-after-construction"

    assert credential.extra["accountId"] == "acc_1"
    with pytest.raises(TypeError):
        credential.extra["accountId"] = "z"  # type: ignore[index]


def test_oauth_credential_extra_is_an_open_escape_hatch() -> None:
    """`extra` mirrors Pi's own open index signature -- a provider-specific field (e.g. Codex's
    own `accountId`) rides alongside the three required fields without widening the closed core
    shape."""
    credential = OAuthCredential(
        access="a", refresh="r", expires=1234.0, extra={"accountId": "acc_1"}
    )
    assert credential.extra["accountId"] == "acc_1"


def test_credential_info_carries_no_secret_fields() -> None:
    info = CredentialInfo(provider_id="openai", type="oauth")
    assert info.provider_id == "openai"
    assert info.type == "oauth"
    assert not hasattr(info, "key")
    assert not hasattr(info, "access")


def test_model_auth_is_closed_to_api_key_headers_base_url() -> None:
    auth = ModelAuth(api_key="sk-test", headers={"x-custom": "1"}, base_url="https://example.test")
    assert auth.api_key == "sk-test"
    assert auth.headers == {"x-custom": "1"}
    assert auth.base_url == "https://example.test"


def test_auth_result_carries_env_and_source() -> None:
    result = AuthResult(auth=ModelAuth(api_key="sk-test"), env={"FOO": "bar"}, source="OAuth")
    assert result.auth.api_key == "sk-test"
    assert result.env == {"FOO": "bar"}
    assert result.source == "OAuth"


def test_auth_check_reports_type_and_source_only() -> None:
    check = AuthCheck(type="api_key", source="OPENAI_API_KEY")
    assert check.type == "api_key"
    assert check.source == "OPENAI_API_KEY"


def test_auth_operation_options_default_signal_is_none() -> None:
    assert AuthOperationOptions().signal is None
