"""Provider-auth interaction and auth-method vocabulary (`PROV-014`; Pi
`packages/ai/src/auth/types.ts:118-241`).

Pass-2 Slice B, contract-first: the login-interaction/prompt/notification vocabulary a provider's
own login flow uses (`AuthPrompt`/`AuthInfoLink`/`AuthEvent`/`AuthInteraction`/
`ProviderAuthInteraction`), and the per-provider auth-METHOD vocabulary a concrete provider
registers (`ApiKeyAuth`/`OAuthAuth`/`ProviderAuth`) -- split out of the previously-bundled
`PROV-013` row, which now covers ONLY the remaining `resolveProviderAuth`/`Models`-level
orchestration built on top of this vocabulary, still `deferred parity` pending a real
`LlmService`/provider-registry integration point Minion does not have yet.

VOCABULARY ONLY -- no orchestration, no `Models`-equivalent dispatcher, no `LlmService`
extension. A concrete provider's own login flow (e.g. `PROV-012`'s Codex OAuth integration, a
later Pass-2 slice) constructs and consumes these types directly; nothing here wires them to a
provider registry.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from .credential import (
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthResult,
    ModelAuth,
    OAuthCredential,
)
from .signal import Abortable

# --- AuthPrompt: the shapes a prompt shown to the user during login can take -----------------


@dataclass(frozen=True, slots=True)
class AuthPromptText:
    """A free-text prompt (Pi `AuthPrompt`'s own `{type: "text", ...}` variant, `types.ts:126`).
    `signal` lets the flow cancel THIS pending prompt when an out-of-band event resolves the step
    first (Pi's own doc comment names exactly this pattern, `types.ts:120-123`) -- the racing
    mechanics themselves belong to the concrete login flow that constructs this prompt, not to
    this vocabulary."""

    message: str
    placeholder: str | None = None
    signal: Abortable | None = None


@dataclass(frozen=True, slots=True)
class AuthPromptSecret:
    """A masked/secret-entry prompt (Pi `{type: "secret", ...}`, `types.ts:127`) -- the same shape
    as `AuthPromptText`, distinguished only by display treatment (never echoed/logged in plain
    text), a caller-side UI concern this type itself does not enforce."""

    message: str
    placeholder: str | None = None
    signal: Abortable | None = None


@dataclass(frozen=True, slots=True)
class AuthPromptOption:
    """One selectable option for `AuthPromptSelect` (Pi's own inline `{id, label, description?}`
    object literal, `types.ts:128`)."""

    id: str
    label: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class AuthPromptSelect:
    """A single-choice prompt from a fixed option set (Pi `{type: "select", ...}`, `types.ts:128`).
    `prompt()`'s own return value for THIS variant is the chosen option's own `id`, never its
    `label` -- Pi's own doc comment states this explicitly ("select returns the option id"),
    `types.ts:152`."""

    message: str
    options: tuple[AuthPromptOption, ...]
    signal: Abortable | None = None


@dataclass(frozen=True, slots=True)
class AuthPromptManualCode:
    """A manual-entry fallback prompt (Pi `{type: "manual_code", ...}`, `types.ts:129`) -- used
    when an interactive callback (e.g. a local OAuth server) is racing this same prompt; Pi's own
    doc comment names exactly this case (`types.ts:120-123`): "a `manual_code` prompt raced
    against a callback server, aborted when the callback wins." `signal` is what that race cancels
    THIS prompt through -- the racing mechanics themselves belong to the concrete login flow that
    constructs this prompt (`PROV-012`), not to this vocabulary."""

    message: str
    placeholder: str | None = None
    signal: Abortable | None = None


type AuthPrompt = AuthPromptText | AuthPromptSecret | AuthPromptSelect | AuthPromptManualCode
"""Every prompt shape shown to the user during login (Pi `AuthPrompt`, `types.ts:125-130`)."""


# --- AuthEvent: the notifications a login flow may emit ---------------------------------------


@dataclass(frozen=True, slots=True)
class AuthInfoLink:
    """A link accompanying an `AuthEventInfo` notification (Pi `AuthInfoLink`,
    `types.ts:132-135`)."""

    url: str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class AuthEventInfo:
    """An informational notification, optionally with supporting links (Pi `{type: "info", ...}`,
    `types.ts:138`)."""

    message: str
    links: tuple[AuthInfoLink, ...] | None = None


@dataclass(frozen=True, slots=True)
class AuthEventUrl:
    """A URL the user should open to continue login (Pi `{type: "auth_url", ...}`, `types.ts:139`)
    -- e.g. Codex's own browser-flow authorization URL (`PROV-012`). Emitting this event is NOT
    the same as opening a browser: browser LAUNCHING remains architecturally out of scope for this
    layer (confirmed during the Pass-2 restart audit -- Pi's own equivalent lives entirely in a
    CLI-layer utility, `packages/coding-agent/src/utils/open-browser.ts`, outside `packages/ai`
    entirely); this event only carries the URL a caller MAY act on."""

    url: str
    instructions: str | None = None


@dataclass(frozen=True, slots=True)
class AuthEventDeviceCode:
    """RFC 8628 device-code details the user must act on (Pi `{type: "device_code", ...}`,
    `types.ts:140-146`) -- the notification counterpart to the already-certified `PROV-010` poll
    state machine's own outcomes; this event is how a login flow tells its OWN caller what to
    display, independent of how the poll loop itself is driven."""

    user_code: str
    verification_uri: str
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AuthEventProgress:
    """A free-text progress update with no further structure (Pi `{type: "progress", ...}`,
    `types.ts:147`)."""

    message: str


type AuthEvent = AuthEventInfo | AuthEventUrl | AuthEventDeviceCode | AuthEventProgress
"""Every notification a login flow may emit (Pi `AuthEvent`, `types.ts:137-147`)."""


# --- AuthInteraction / ProviderAuthInteraction: the callback surface a login flow receives ----


class AuthInteraction(Protocol):
    """The login-interaction callback surface serving BOTH api-key and OAuth flows (Pi
    `AuthInteraction`, `types.ts:156-161`). `prompt()` returns the entered/selected string
    (`select` returns the chosen option's own `id`); raises/rejects on cancel/abort. `signal`
    cancels the WHOLE login flow; per-prompt cancellation instead uses that specific
    `AuthPrompt`'s own `signal` field."""

    signal: Abortable | None

    async def prompt(self, prompt: AuthPrompt) -> str: ...
    def notify(self, event: AuthEvent) -> None: ...


class ProviderAuthInteraction(Protocol):
    """The NORMALIZED interaction passed to a concrete provider's own login implementation (Pi
    `ProviderAuthInteraction`, `types.ts:164`) -- identical to `AuthInteraction` except `signal`
    is REQUIRED, not optional: by the time a provider's own `login()` callable is invoked, the
    caller has already normalized an absent top-level signal into a real, always-present one.
    Declared as a SEPARATE, standalone `Protocol` rather than narrowing `AuthInteraction.signal`
    through inheritance -- `Protocol`'s own structural typing means any object whose `prompt`/
    `notify`/`signal` shape matches BOTH protocols conforms to both simultaneously regardless of
    declared inheritance, and narrowing a mutable Protocol attribute through a subclass runs into
    ordinary invariance rules for no real benefit here."""

    signal: Abortable

    async def prompt(self, prompt: AuthPrompt) -> str: ...
    def notify(self, event: AuthEvent) -> None: ...


# --- ApiKeyAuth / OAuthAuth / ProviderAuth: the per-provider auth-method vocabulary ------------

type ApiKeyLogin = Callable[[ProviderAuthInteraction], Awaitable[ApiKeyCredential]]
"""Interactive api-key setup (Pi `ApiKeyAuth.login?`, `types.ts:175`). Absent means ambient-only
(no interactive setup; the provider relies solely on `resolve`'s own ambient-source fallback)."""

type ApiKeyCheck = Callable[
    [AuthContext, ApiKeyCredential | None, Abortable], Awaitable[AuthCheck | None]
]
"""Optional side-effect-free availability check (Pi `ApiKeyAuth.check?`, `types.ts:182-186`) --
use when `resolve()` may itself execute commands or perform other request-time work; absent means
availability is instead checked by resolving auth directly."""

type ApiKeyResolve = Callable[
    [AuthContext, ApiKeyCredential | None, Abortable], Awaitable[AuthResult | None]
]
"""Resolve auth from the stored credential and/or ambient sources (Pi `ApiKeyAuth.resolve`,
`types.ts:194-198`), merging per field. `None` means not configured."""


@dataclass(frozen=True, slots=True)
class ApiKeyAuth:
    """Api-key auth-method vocabulary a concrete provider registers (Pi `ApiKeyAuth`,
    `types.ts:170-199`): stored key/provider env plus ambient sources (env vars, AWS profiles, ADC
    files). An ambient-only provider omits `login`."""

    name: str
    """Display name, e.g. "Anthropic API key" (Pi `types.ts:172`)."""
    resolve: ApiKeyResolve
    login: ApiKeyLogin | None = None
    check: ApiKeyCheck | None = None


type OAuthLogin = Callable[[ProviderAuthInteraction], Awaitable[OAuthCredential]]
"""Interactive OAuth login (Pi `OAuthAuth.login`, `types.ts:216`) -- unlike `ApiKeyAuth.login`,
REQUIRED: every OAuth auth method has an interactive setup flow."""

type OAuthRefresh = Callable[[OAuthCredential, Abortable], Awaitable[OAuthCredential]]
"""Exchange the refresh token (Pi `OAuthAuth.refresh`, `types.ts:222`) -- a network call; raises
on failure (invalid_grant etc.). Already-certified `PROV-008`'s own `refresh_if_expiring_at` runs
this under the credential store's own lock, matching Pi's own `Models`-level locked-refresh
design."""

type OAuthToAuth = Callable[[OAuthCredential], Awaitable[ModelAuth]]
"""Side-effect-free derivation of request auth from a valid credential (Pi `OAuthAuth.toAuth`,
`types.ts:229`) -- covers per-credential `base_url` (e.g. GitHub Copilot); async so a lazy wrapper
can load its own implementation on first use."""


@dataclass(frozen=True, slots=True)
class OAuthAuth:
    """OAuth auth-method vocabulary a concrete provider registers (Pi `OAuthAuth`,
    `types.ts:206-230`). The `refresh`/`to_auth` split lets a future orchestration layer own the
    locked-refresh pattern: `refresh` produces a credential, `to_auth` derives request auth from
    whatever credential ends up stored -- already-certified `PROV-011`'s own Codex
    `credentials_from_token`/`to_auth` are a concrete instance of exactly this split."""

    name: str
    """Display name, e.g. "OpenAI (ChatGPT Plus/Pro)" (Pi `types.ts:208`)."""
    login: OAuthLogin
    refresh: OAuthRefresh
    to_auth: OAuthToAuth
    is_subscription: bool = False
    """Whether access through this auth method is backed by a provider subscription (Pi
    `isSubscription?`, `types.ts:211`)."""
    login_label: str | None = None
    """Selector label for the OAuth login option, e.g. "Sign in with SuperGrok or X Premium" (Pi
    `loginLabel?`, `types.ts:214`)."""


@dataclass(frozen=True, slots=True)
class ProviderAuth:
    """Per-provider auth-method registration (Pi `ProviderAuth`, `types.ts:237-240`). At least one
    of `api_key`/`oauth` MUST be present -- Pi's own doc comment states this as a real constraint,
    not merely a convention: "even ambient-credential providers and keyless local servers provide
    `apiKey` auth whose `resolve()` reports whether the provider is configured." Enforced here at
    construction time, matching that binding requirement rather than leaving it an unchecked
    convention."""

    api_key: ApiKeyAuth | None = None
    oauth: OAuthAuth | None = None

    def __post_init__(self) -> None:
        if self.api_key is None and self.oauth is None:
            raise ValueError("ProviderAuth requires at least one of api_key or oauth")
