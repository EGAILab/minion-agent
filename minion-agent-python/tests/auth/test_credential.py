"""Structural-shape tests for the auth vocabulary (`PROV-006`)."""

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


def test_w_r006_1_mutating_the_original_env_mapping_after_construction_is_observable() -> None:
    """`L11-R006` (owner-decided: adopt pinned Pi's own live-reference semantics, no intentional
    divergence) + `L11-R010` (env's own domain is pinned Pi's flat `ProviderEnv = Record<string,
    string>`, `types.ts:113` -- never recursive JSON, unlike `OAuthCredential.extra`): mutating the
    ORIGINAL mapping passed to the constructor remains observable through the credential
    afterward, matching Pi's own plain, mutable, reference-shared object exactly."""
    original = {"CF_ACCOUNT_ID": "abc"}
    credential = ApiKeyCredential(key="sk-test", env=original)

    original["CF_ACCOUNT_ID"] = "mutated-after-construction"

    assert credential.env is not None
    assert credential.env["CF_ACCOUNT_ID"] == "mutated-after-construction"


def test_w_r006_2_assigning_a_new_top_level_key_on_env_persists() -> None:
    """`L11-R006` (targeted-closure finding): the review's own minimal witness -- a caller must be
    able to assign a BRAND-NEW top-level key through the returned, statically-typed `env` mapping
    itself (not merely mutate an existing value), and mypy must accept it. A prior revision's own
    `Mapping[str, str]` annotation rejected this at the type level even though the runtime `dict`
    underneath would have allowed it."""
    credential = ApiKeyCredential(env={})

    credential.env["NEW"] = "v"  # no type: ignore needed -- env is a genuine mutable dict

    assert credential.env == {"NEW": "v"}


def test_w_r006_3_oauth_credential_extra_open_json_domain_round_trips_unchanged() -> None:
    """`L11-R006`: no freezing/deep-copy mechanism narrows `extra`'s own open JSON domain --
    objects, arrays, strings, numbers, booleans, and null all round-trip exactly, and remain
    the SAME container objects (proving no defensive copy occurred)."""
    nested_object = {"inner": "value"}
    nested_array = [1, "two", None, True]
    extra = {
        "object": nested_object,
        "array": nested_array,
        "string": "s",
        "number": 3.5,
        "boolean": False,
        "null": None,
    }

    credential = OAuthCredential(access="a", refresh="r", expires=1234.0, extra=extra)

    assert credential.extra == extra
    assert credential.extra["object"] is nested_object
    assert credential.extra["array"] is nested_array


def test_w_r006_1_oauth_credential_extra_original_mapping_aliasing() -> None:
    """`L11-R006`, `OAuthCredential`'s own `extra` field -- same constructor-aliasing guarantee as
    `ApiKeyCredential.env` above, including a nested mutable value (`extra`'s own domain is full
    recursive JSON, unlike `env`'s flat `L11-R010` domain, so nested aliasing is in-scope here)."""
    original = {"accountId": "acc_1", "nested": {"value": "A"}}
    credential = OAuthCredential(access="a", refresh="r", expires=1234.0, extra=original)

    original["accountId"] = "mutated-after-construction"
    original["nested"]["value"] = "B"

    assert credential.extra["accountId"] == "mutated-after-construction"
    assert credential.extra["nested"] == {"value": "B"}


def test_w_r006_2_mutating_a_nested_value_reached_through_extra_persists() -> None:
    """`L11-R006`: mutating a nested dict/list reached through `credential.extra` itself succeeds
    and is observed by a later access -- `extra` is not defensively copied or frozen at any level.
    `env` has no equivalent nested case since its own domain is flat (`L11-R010`)."""
    credential = OAuthCredential(
        access="a", refresh="r", expires=1234.0, extra={"nested": {"value": "A"}, "items": ["A"]}
    )

    credential.extra["nested"]["value"] = "B"  # type: ignore[index]
    credential.extra["items"].append("B")  # type: ignore[union-attr]

    assert credential.extra["nested"] == {"value": "B"}
    assert credential.extra["items"] == ["A", "B"]


def test_w_r006_new_top_level_key_assignment_on_extra_persists() -> None:
    """`L11-R006` (targeted-closure finding): the same brand-new-top-level-key witness as `env`
    above, exercised on `extra`, which additionally permits a recursive JSON value as that key."""
    credential = OAuthCredential(access="a", refresh="r", expires=1234.0)

    credential.extra["nested"] = {"value": "A"}

    assert credential.extra == {"nested": {"value": "A"}}


def test_w_r009_api_key_credential_scalar_fields_are_reassignable() -> None:
    """`L11-R009`: Pi's own returned live credential permits mutating scalar fields directly (a
    plain, mutable JS object) -- this dataclass must not be frozen, matching that exactly. No
    intentional divergence for scalar fields was ever approved (the owner's own `L11-R006`
    decision covered `env`/`extra` value/reference semantics, and this project defaults to Pi
    parity absent an approved divergence)."""
    credential = ApiKeyCredential(key="A")

    credential.key = "B"

    assert credential.key == "B"


def test_w_r009_oauth_credential_scalar_fields_are_reassignable() -> None:
    """`L11-R009`, `OAuthCredential`'s own scalar fields -- same guarantee as `ApiKeyCredential`
    above, covering all three required fields."""
    credential = OAuthCredential(access="a1", refresh="r1", expires=100.0)

    credential.access = "a2"
    credential.refresh = "r2"
    credential.expires = 200.0

    assert credential.access == "a2"
    assert credential.refresh == "r2"
    assert credential.expires == 200.0


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
