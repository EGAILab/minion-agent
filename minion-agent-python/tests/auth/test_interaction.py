"""Provider-auth interaction/auth-method vocabulary (`PROV-014`) -- pure shape/construction,
no orchestration."""

import pytest

from minion_agent.auth.interaction import (
    ApiKeyAuth,
    AuthEventDeviceCode,
    AuthEventInfo,
    AuthEventProgress,
    AuthEventUrl,
    AuthInfoLink,
    AuthPromptManualCode,
    AuthPromptOption,
    AuthPromptSecret,
    AuthPromptSelect,
    AuthPromptText,
    OAuthAuth,
    ProviderAuth,
)


def test_auth_prompt_text_carries_message_placeholder_and_signal() -> None:
    prompt = AuthPromptText(message="Enter your key", placeholder="sk-...")
    assert prompt.message == "Enter your key"
    assert prompt.placeholder == "sk-..."
    assert prompt.signal is None


def test_auth_prompt_secret_has_the_same_shape_as_text() -> None:
    prompt = AuthPromptSecret(message="Enter your secret")
    assert prompt.message == "Enter your secret"
    assert prompt.placeholder is None


def test_auth_prompt_select_stores_options_with_distinct_id_and_label_fields() -> None:
    """Construction/shape only -- `AuthPromptOption` carries `id` and `label` as two DISTINCT
    fields (deliberately given different values below so a future accidental field swap would be
    caught). This does NOT prove that some `AuthInteraction.prompt()` implementation actually
    returns the `id`, not the `label`, when resolving a `select` prompt -- Slice B defines
    vocabulary only and implements no concrete `AuthInteraction`, so no such behavioral witness is
    possible yet. That behavioral claim belongs to whichever later slice first supplies a real
    interaction seam to test against (`L11-SB-R004`, independent review -- an earlier revision of
    this test's own manifest description overclaimed this)."""
    options = (
        AuthPromptOption(id="browser", label="Browser login (default)"),
        AuthPromptOption(id="device_code", label="Device code login (headless)"),
    )
    prompt = AuthPromptSelect(message="Select login method:", options=options)
    assert prompt.options == options
    assert prompt.options[0].id == "browser"
    assert prompt.options[0].label == "Browser login (default)"
    assert prompt.options[0].id != prompt.options[0].label


def test_auth_prompt_manual_code_carries_message_placeholder_and_signal() -> None:
    prompt = AuthPromptManualCode(
        message="Paste the authorization code / redirect URL here:",
        placeholder="http://localhost:1455/auth/callback",
    )
    assert prompt.message.startswith("Paste")
    assert prompt.placeholder is not None


def test_auth_event_info_links_default_to_none() -> None:
    event = AuthEventInfo(message="hello")
    assert event.links is None
    linked = AuthEventInfo(message="hello", links=(AuthInfoLink(url="https://example.com"),))
    assert linked.links is not None
    assert linked.links[0].url == "https://example.com"
    assert linked.links[0].label is None


def test_auth_event_url_carries_url_and_optional_instructions() -> None:
    event = AuthEventUrl(url="https://auth.openai.com/oauth/authorize?...", instructions="Open me")
    assert event.url.startswith("https://")
    assert event.instructions == "Open me"


def test_auth_event_device_code_carries_all_rfc8628_display_fields() -> None:
    event = AuthEventDeviceCode(
        user_code="ABCD-EFGH",
        verification_uri="https://auth.openai.com/codex/device",
        interval_seconds=5.0,
        expires_in_seconds=900.0,
    )
    assert event.user_code == "ABCD-EFGH"
    assert event.interval_seconds == 5.0


def test_auth_event_progress_is_a_bare_message() -> None:
    assert AuthEventProgress(message="Waiting for authorization...").message.startswith("Waiting")


def test_provider_auth_requires_at_least_one_of_api_key_or_oauth() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        ProviderAuth()


def test_provider_auth_accepts_api_key_only() -> None:
    async def resolve(ctx, credential, signal):  # type: ignore[no-untyped-def]
        return None

    auth = ProviderAuth(api_key=ApiKeyAuth(name="Test API key", resolve=resolve))
    assert auth.api_key is not None
    assert auth.oauth is None


def test_provider_auth_accepts_oauth_only() -> None:
    async def login(interaction):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def refresh(credential, signal):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def to_auth(credential):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    auth = ProviderAuth(
        oauth=OAuthAuth(name="Test OAuth", login=login, refresh=refresh, to_auth=to_auth)
    )
    assert auth.oauth is not None
    assert auth.api_key is None


def test_provider_auth_accepts_both() -> None:
    async def resolve(ctx, credential, signal):  # type: ignore[no-untyped-def]
        return None

    async def login(interaction):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def refresh(credential, signal):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def to_auth(credential):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    auth = ProviderAuth(
        api_key=ApiKeyAuth(name="Test API key", resolve=resolve),
        oauth=OAuthAuth(name="Test OAuth", login=login, refresh=refresh, to_auth=to_auth),
    )
    assert auth.api_key is not None
    assert auth.oauth is not None


def test_api_key_auth_login_and_check_default_to_none() -> None:
    async def resolve(ctx, credential, signal):  # type: ignore[no-untyped-def]
        return None

    auth = ApiKeyAuth(name="Ambient-only provider", resolve=resolve)
    assert auth.login is None
    assert auth.check is None


def test_oauth_auth_is_subscription_defaults_to_none_not_false() -> None:
    """`L11-SB-R001`: pinned Pi's own `isSubscription?: boolean` is genuinely OPTIONAL -- absent,
    explicit `false`, and explicit `true` are three distinct observable states. Defaulting the
    Python field to `False` would make "omitted" indistinguishable from "explicitly false,"
    silently narrowing Pi's own three-valued field to two. `login_label` defaults to `None`
    (unchanged -- always genuinely optional in Pi with no analogous collapse risk)."""

    async def login(interaction):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def refresh(credential, signal):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def to_auth(credential):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    auth = OAuthAuth(name="Test OAuth", login=login, refresh=refresh, to_auth=to_auth)
    assert auth.is_subscription is None
    assert auth.login_label is None


def test_oauth_auth_is_subscription_distinguishes_absent_false_and_true() -> None:
    """The permanent three-state witness `L11-SB-R001` requires: absent (`None`), explicit
    `False`, and explicit `True` must all be independently observable, not collapsed into two
    states by a bare-`bool`-with-`False`-default field."""

    async def login(interaction):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def refresh(credential, signal):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def to_auth(credential):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    absent = OAuthAuth(name="Absent", login=login, refresh=refresh, to_auth=to_auth)
    explicit_false = OAuthAuth(
        name="Explicit false", login=login, refresh=refresh, to_auth=to_auth, is_subscription=False
    )
    explicit_true = OAuthAuth(
        name="Explicit true", login=login, refresh=refresh, to_auth=to_auth, is_subscription=True
    )
    assert absent.is_subscription is None
    assert explicit_false.is_subscription is False
    assert explicit_true.is_subscription is True


def test_public_vocabulary_fields_are_mutable_matching_pi() -> None:
    """`L11-SB-R005`: pinned Pi's own public object/interface field shapes are NOT `readonly`
    (only the two collection fields, `AuthPromptSelect.options`/`AuthEventInfo.links`, are --
    unaffected here, and stay `tuple`s deliberately). Ordinary field reassignment must succeed,
    not raise `FrozenInstanceError`, for every adopted vocabulary type -- construct-then-mutate is
    the only faithful reproduction of Pi's own assignable-property semantics; a frozen dataclass
    would be an unapproved, undisclosed divergence (this project's own established precedent for
    the identical question on Layer-11 credentials, `PROV-006`/`L11-R006`/`L11-R009`, resolved it
    the same way: adopt Pi's assignable fields in full)."""
    prompt = AuthPromptText(message="original")
    prompt.message = "replaced"
    assert prompt.message == "replaced"

    option = AuthPromptOption(id="a", label="A")
    option.label = "A (renamed)"
    assert option.label == "A (renamed)"

    event = AuthEventProgress(message="original")
    event.message = "replaced"
    assert event.message == "replaced"

    async def login(interaction):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def refresh(credential, signal):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def to_auth(credential):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    auth = OAuthAuth(name="original", login=login, refresh=refresh, to_auth=to_auth)
    auth.name = "renamed"
    auth.is_subscription = True
    assert auth.name == "renamed"
    assert auth.is_subscription is True
