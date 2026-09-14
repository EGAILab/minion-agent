"""Permanent static-type evidence for `PROV-014` (provider-auth interaction/auth-method
vocabulary).

Not a pytest test: mypy checking this module IS the test. `tests/auth/test_interaction.py`'s own
inline callback functions are loosely typed (`# type: ignore[no-untyped-def]`) since pytest's own
default gate never type-checks `tests/`; this module instead proves, under full mypy strictness,
that a PROPERLY-typed provider implementation actually satisfies `AuthInteraction`/
`ProviderAuthInteraction`'s own structural `Protocol` shape and each `Callable` type alias --
exactly the kind of static guarantee runtime tests alone cannot pin.

Run explicitly (not part of the default `mypy` gate, which is scoped to `src/minion_agent` only):

    mypy src/minion_agent tests/typing/valid_interaction_vocabulary.py

Never imported or executed by pytest.
"""

from __future__ import annotations

from minion_agent.auth.credential import (
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthResult,
    ModelAuth,
    OAuthCredential,
)
from minion_agent.auth.interaction import (
    ApiKeyAuth,
    ApiKeyCheck,
    ApiKeyLogin,
    ApiKeyResolve,
    AuthEvent,
    AuthInteraction,
    AuthPrompt,
    OAuthAuth,
    OAuthLogin,
    OAuthRefresh,
    OAuthToAuth,
    ProviderAuth,
    ProviderAuthInteraction,
)
from minion_agent.auth.signal import Abortable


# Properly-typed implementations of the `AuthInteraction`/`ProviderAuthInteraction` callback shape
# a concrete login flow could receive -- assigning each to its own Protocol-typed variable below is
# the actual type-check under test; a signature mismatch here is a `mypy` error, not a runtime
# failure. TWO separate classes, not one: `signal`'s own declared type differs between the two
# Protocols (`Abortable | None` vs `Abortable`), and a mutable Protocol attribute is invariant, so
# no single class attribute type can satisfy both simultaneously (matching `interaction.py`'s own
# docstring on why `ProviderAuthInteraction` is a standalone Protocol, not a narrowing subclass).
class _FakeInteraction:
    signal: Abortable | None = None

    async def prompt(self, prompt: AuthPrompt) -> str:
        return "entered-value"

    def notify(self, event: AuthEvent) -> None:
        pass


class _FakeProviderInteraction:
    signal: Abortable

    def __init__(self, signal: Abortable) -> None:
        self.signal = signal

    async def prompt(self, prompt: AuthPrompt) -> str:
        return "entered-value"

    def notify(self, event: AuthEvent) -> None:
        pass


class _FakeSignal:
    @property
    def aborted(self) -> bool:
        return False


_interaction: AuthInteraction = _FakeInteraction()
_provider_interaction: ProviderAuthInteraction = _FakeProviderInteraction(_FakeSignal())


async def _api_key_login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
    return ApiKeyCredential(key="sk-test")


async def _api_key_check(
    ctx: AuthContext, credential: ApiKeyCredential | None, signal: Abortable
) -> AuthCheck | None:
    return None


async def _api_key_resolve(
    ctx: AuthContext, credential: ApiKeyCredential | None, signal: Abortable
) -> AuthResult | None:
    return None


_login: ApiKeyLogin = _api_key_login
_check: ApiKeyCheck = _api_key_check
_resolve: ApiKeyResolve = _api_key_resolve

_api_key_auth = ApiKeyAuth(name="Test", login=_login, check=_check, resolve=_resolve)


async def _oauth_login(interaction: ProviderAuthInteraction) -> OAuthCredential:
    return OAuthCredential(access="a", refresh="r", expires=0.0)


async def _oauth_refresh(credential: OAuthCredential, signal: Abortable) -> OAuthCredential:
    return credential


async def _oauth_to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


_oauth_login_fn: OAuthLogin = _oauth_login
_oauth_refresh_fn: OAuthRefresh = _oauth_refresh
_oauth_to_auth_fn: OAuthToAuth = _oauth_to_auth

_oauth_auth = OAuthAuth(
    name="Test", login=_oauth_login_fn, refresh=_oauth_refresh_fn, to_auth=_oauth_to_auth_fn
)

_provider_auth = ProviderAuth(api_key=_api_key_auth, oauth=_oauth_auth)
