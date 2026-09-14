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

Every dataclass below is genuinely MUTABLE (`slots=True`, deliberately NOT `frozen=True`),
matching pinned Pi's own public object/interface field shapes, none of which are `readonly`
(`L11-SB-R005`, independent review; only the two COLLECTION fields, `AuthPromptSelect.options`/
`AuthEventInfo.links`, are `readonly` in Pi, and stay `tuple`s here for exactly that reason -- an
ordinary field being freely reassignable is a DIFFERENT question from a collection's own elements
being replaceable one at a time, and only the latter is restricted in Pi). This project already
resolved the identical "should an adopted public value be frozen despite Pi's own assignable
fields" question for Layer-11 credentials (`PROV-006`, `L11-R006`/`L11-R009`, owner-decided:
adopt Pi's assignable fields in full, no intentional divergence approved) -- freezing this newly
adopted vocabulary without an equivalent, separately-recorded owner approval would silently
reopen that same resolved question for a new type family.
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


@dataclass(slots=True)
class AuthPromptText:
    """A free-text prompt (Pi `AuthPrompt`'s own `{type: "text", ...}` variant, `types.ts:126`).
    `signal` lets the flow cancel THIS pending prompt when an out-of-band event resolves the step
    first (Pi's own doc comment names exactly this pattern, `types.ts:120-123`) -- the racing
    mechanics themselves belong to the concrete login flow that constructs this prompt, not to
    this vocabulary."""

    message: str
    placeholder: str | None = None
    signal: Abortable | None = None


@dataclass(slots=True)
class AuthPromptSecret:
    """A masked/secret-entry prompt (Pi `{type: "secret", ...}`, `types.ts:127`) -- the same shape
    as `AuthPromptText`, distinguished only by display treatment (never echoed/logged in plain
    text), a caller-side UI concern this type itself does not enforce."""

    message: str
    placeholder: str | None = None
    signal: Abortable | None = None


@dataclass(slots=True)
class AuthPromptOption:
    """One selectable option for `AuthPromptSelect` (Pi's own inline `{id, label, description?}`
    object literal, `types.ts:128`)."""

    id: str
    label: str
    description: str | None = None


@dataclass(slots=True)
class AuthPromptSelect:
    """A single-choice prompt from a fixed option set (Pi `{type: "select", ...}`, `types.ts:128`).
    `prompt()`'s own return value for THIS variant is the chosen option's own `id`, never its
    `label` -- Pi's own doc comment states this explicitly ("select returns the option id"),
    `types.ts:152`."""

    message: str
    options: tuple[AuthPromptOption, ...]
    signal: Abortable | None = None


@dataclass(slots=True)
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


@dataclass(slots=True)
class AuthInfoLink:
    """A link accompanying an `AuthEventInfo` notification (Pi `AuthInfoLink`,
    `types.ts:132-135`)."""

    url: str
    label: str | None = None


@dataclass(slots=True)
class AuthEventInfo:
    """An informational notification, optionally with supporting links (Pi `{type: "info", ...}`,
    `types.ts:138`)."""

    message: str
    links: tuple[AuthInfoLink, ...] | None = None


@dataclass(slots=True)
class AuthEventUrl:
    """A URL the user should open to continue login (Pi `{type: "auth_url", ...}`, `types.ts:139`)
    -- e.g. Codex's own browser-flow authorization URL (`PROV-012`). Emitting this event is NOT
    the same as opening a browser: browser LAUNCHING remains architecturally out of scope for this
    layer (confirmed during the Pass-2 restart audit -- Pi's own equivalent lives entirely in a
    CLI-layer utility, `packages/coding-agent/src/utils/open-browser.ts`, outside `packages/ai`
    entirely); this event only carries the URL a caller MAY act on."""

    url: str
    instructions: str | None = None


@dataclass(slots=True)
class AuthEventDeviceCode:
    """RFC 8628 device-code details the user must act on (Pi `{type: "device_code", ...}`,
    `types.ts:140-146`) -- the notification counterpart to the already-certified `PROV-010` poll
    state machine's own outcomes; this event is how a login flow tells its OWN caller what to
    display, independent of how the poll loop itself is driven."""

    user_code: str
    verification_uri: str
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


@dataclass(slots=True)
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
    `AuthPrompt`'s own `signal` field.

    `notify()` is a DIRECT SYNCHRONOUS call, not fire-and-forget/detached delivery (`L11-SB-R003`,
    second independent review, CORRECTING an earlier revision's own ambiguous "sync,
    fire-and-forget" phrasing, which could be misread as "delivery failures are swallowed"):
    confirmed directly against pinned Pi's own real call sites
    (`packages/ai/src/auth/oauth/openai-codex.ts:429`, `:456`) --
    `interaction.notify({...})` is an ordinary, un-awaited, un-wrapped statement in the middle of
    an `async` function's own body, with NO enclosing `try`/`catch` at either call site and no
    detachment mechanism (no `setTimeout`, no `.catch()`, no fire-and-forget queuing) -- a
    synchronous throw from `notify()` propagates directly out of the caller exactly like any other
    synchronous statement would, becoming a rejected promise for the whole `login()` call. This
    project's own bare `def notify(...)` (not `async def`) already models this correctly (a
    synchronous Python exception propagates the same way); this note exists because the PROSE
    describing it was previously ambiguous, not because the TYPE signature was wrong.

    `signal` is declared as a READ-ONLY `@property`, not a plain mutable instance attribute
    (`L11-SB-R002`, independent review): pinned Pi's own `ProviderAuthInteraction = AuthInteraction
    & { signal: AbortSignal }` is a TypeScript intersection type, under which a normalized
    interaction (required `signal`) remains a valid `AuthInteraction` (optional `signal`) --
    ordinary structural subtyping, since "always present with this type" trivially satisfies
    "optionally present with this type." A mutable Protocol attribute is INVARIANT under Python's
    own static-typing rules (a consumer could otherwise assign an incompatible value through the
    narrower reference), which silently broke this exact subtype relationship in an earlier
    revision: `ProviderAuthInteraction` could not be used anywhere `AuthInteraction` was expected,
    contradicting Pi's own intersection-type semantics. A read-only property is instead
    COVARIANT -- `ProviderAuthInteraction`'s own narrower `Abortable` return type correctly
    satisfies `AuthInteraction`'s own wider `Abortable | None` requirement, matching Pi exactly. A
    concrete implementation is unaffected: an ordinary mutable instance attribute (not itself a
    `@property`) still satisfies a Protocol's own read-only property requirement, since Protocol
    matching only checks the READ side.

    INTENTIONAL, NARROW, OWNER-APPROVED LANGUAGE-BINDING DIVERGENCE (`L11-SB-R006`,
    `GOVERNANCE_SOURCE`: owner decision recorded verbatim at
    `https://github.com/EGAILab/minion-agent/issues/29#issuecomment-5664609556`, per
    `agent-workflow.md` §11.10): pinned Pi's own TypeScript type system additionally permits
    ASSIGNING through an `AuthInteraction`-typed reference (`signal` is a plain, writable property
    there, not `readonly`) -- a capability this read-only `@property` design does NOT reproduce.
    This is a GENUINE, UNAVOIDABLE trade-off, not an oversight: TypeScript's own structural typing
    for mutable object properties is well-documented to be UNSOUND for exactly this combination
    (a required-property type narrowing a wider optional-property type, both independently
    writable) -- Python's own sound type system cannot safely replicate that unsoundness no matter
    how `signal` is modeled (a full read/write `@property` with a matching `.setter` was tried and
    independently confirmed to reintroduce the ORIGINAL `L11-SB-R002` subtyping failure instead,
    since a writable property is invariant for the identical reason a plain attribute is). Owner
    governance explicitly chose to preserve the SUBTYPING relationship and the provider-login
    guaranteed-`signal` invariant over widened-reference assignability, having confirmed (by
    grepping the entire pinned Pi source tree) that NO actual Pi call site ever reassigns an
    interaction's own `signal` after construction -- the sacrificed capability is a static
    permission Pi's own real code never exercises. This divergence is SCOPED EXCLUSIVELY to
    assignment through a value statically typed as `AuthInteraction`/`ProviderAuthInteraction`;
    it does NOT require a concrete implementation's own runtime object to be immutable -- a
    concrete class is free to expose its own mutable `signal` attribute or setter through its OWN
    concrete type, matching Pi's own real object behavior exactly, as long as this Protocol's own
    read-only view remains what generic vocabulary-consuming code sees. Tracked as a separate
    manifest subject (`PROV-015`), NOT mixed into this row's own `adopted` disposition."""

    @property
    def signal(self) -> Abortable | None: ...
    async def prompt(self, prompt: AuthPrompt) -> str: ...
    def notify(self, event: AuthEvent) -> None: ...


class ProviderAuthInteraction(Protocol):
    """The NORMALIZED interaction passed to a concrete provider's own login implementation (Pi
    `ProviderAuthInteraction`, `types.ts:164`) -- identical to `AuthInteraction` except `signal`
    is REQUIRED, not optional: by the time a provider's own `login()` callable is invoked, the
    caller has already normalized an absent top-level signal into a real, always-present one. See
    `AuthInteraction`'s own docstring for why `signal` is a read-only `@property` here (covariant,
    correctly subtyping `AuthInteraction`, `L11-SB-R002`) and for the owner-approved, narrow
    divergence this entails for widened-reference assignment specifically (`L11-SB-R006`,
    tracked at `PROV-015`)."""

    @property
    def signal(self) -> Abortable: ...
    async def prompt(self, prompt: AuthPrompt) -> str: ...
    def notify(self, event: AuthEvent) -> None: ...


# --- ApiKeyAuth / OAuthAuth / ProviderAuth: the per-provider auth-method vocabulary ------------

type ApiKeyLogin = Callable[[ProviderAuthInteraction], Awaitable[ApiKeyCredential]]
"""Interactive api-key setup (Pi `ApiKeyAuth.login?`, `types.ts:175`). ONE required positional
parameter (the normalized interaction); returns the newly-obtained credential, or raises/rejects
on failure/cancellation -- Pi's own signature has no separate error channel. Absent means
ambient-only (no interactive setup; the provider relies solely on `resolve`'s own ambient-source
fallback).

UNWRAPPED AT PI'S OWN REAL CALL SITE (`L11-SB-R007`, third independent review): unlike
`ApiKeyCheck`/`ApiKeyResolve`/`OAuthToAuth` above, confirmed directly against pinned Pi that
`Models.login()` (`models.ts:565-575`) calls `method.login({...interaction, signal})` and awaits
the result through `raceWithAbortSignal(loginOperation, signal)` directly, with NO enclosing
`try`/`catch` around that call -- a rejection propagates straight out of `Models.login()` itself.
(`Models.login()` has a LATER `try`/`catch`, `models.ts:591-613`, but that covers only the
subsequent credential-store mutation step, not the login call.) A future orchestration layer
consuming this callable therefore has no existing Pi wrapping convention to mirror for login
failures specifically, unlike the four callables above."""

type ApiKeyCheck = Callable[
    [AuthContext, ApiKeyCredential | None, Abortable], Awaitable[AuthCheck | None]
]
"""Optional side-effect-free availability check (Pi `ApiKeyAuth.check?`, `types.ts:182-186`) --
use when `resolve()` may itself execute commands or perform other request-time work; absent means
availability is instead checked by resolving auth directly.

DISCLOSED MAPPING, not a Pi-visible semantic change (`L11-SB-R003`, independent review): pinned
Pi's own signature takes ONE structured input object, `{ctx, credential?, signal}` -- `credential`
OMITTED when none is stored, `ctx`/`signal` always present. This project represents that SAME
three-field bundle as three ordinary POSITIONAL parameters instead (`ctx, credential, signal`, with
`credential` typed `ApiKeyCredential | None` -- Python has no direct equivalent of "the whole
object argument is required, but one of its OWN member fields may be omitted," so an always-present
parameter that may be `None` is the faithful rendering, not an optional parameter position, which
would instead model an OMITTED ARGUMENT -- a different thing Pi's own signature does not have
here). This is a LANGUAGE MAPPING for how the SAME three logical values are passed, matching every
other injected-callable convention already established in this codebase (e.g. `refresh.py`'s own
`RefreshOperation`) -- it changes no observable input value, requiredness, or behavior; a caller
supplies the identical three pieces of information either way. `credential`/`signal`'s own
requiredness is unchanged from Pi (`credential` may be absent/`None`; `ctx`/`signal` are always
present); the return type (`AuthCheck | None`, async) is unchanged from Pi (`None` means not
configured, exactly matching `ApiKeyResolve` below).

MAY RAISE/REJECT (`L11-SB-R003`, second independent review): confirmed directly against pinned Pi
(`models.ts:495-504`) -- `Models.checkProviderAuth` `await`s `apiKey.check(...)` inside its own
`try`/`catch`, wrapping a rejection as an auth-check failure. `ApiKeyCheck` itself carries NO
error-suppression contract of its own; a caller CONSUMING this callable (a future orchestration
layer, `PROV-013`, not this row) owns deciding how a raised exception is wrapped/reported --
`PROV-014` states only that raising is a valid, expected outcome, matching Pi's own `catch`-wrapped
call site."""

type ApiKeyResolve = Callable[
    [AuthContext, ApiKeyCredential | None, Abortable], Awaitable[AuthResult | None]
]
"""Resolve auth from the stored credential and/or ambient sources (Pi `ApiKeyAuth.resolve`,
`types.ts:194-198`), merging per field. `None` means not configured. Same three-positional-
parameter mapping from Pi's own structured `{ctx, credential?, signal}` input as `ApiKeyCheck`
above -- see that type's own docstring for why this is a disclosed language mapping, not an
observable semantic change.

MAY RAISE/REJECT (`L11-SB-R003`, second independent review): confirmed directly against pinned Pi
(`resolve.ts:188-192`, `resolveApiKey`) -- `await`ed inside a `try`/`catch`, wrapping a rejection
as an auth failure. Same "raising is valid, wrapping is the future orchestration owner's own
concern" contract as `ApiKeyCheck` above."""


@dataclass(slots=True)
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
REQUIRED: every OAuth auth method has an interactive setup flow. ONE required positional parameter
(the normalized interaction, unbundled 1:1 from Pi's own single-argument signature -- no grouping
ambiguity here, unlike `ApiKeyCheck`/`ApiKeyResolve` above); returns the newly-obtained credential,
or raises/rejects on failure/cancellation.

UNWRAPPED AT PI'S OWN REAL CALL SITE (`L11-SB-R007`, third independent review): the SAME
`Models.login()` call site as `ApiKeyLogin` above (both auth-method variants share it,
`models.ts:565-575`) -- see `ApiKeyLogin`'s own docstring for the exact citation and reasoning.
A rejection here propagates unwrapped, unlike `OAuthRefresh`/`OAuthToAuth` below."""

type OAuthRefresh = Callable[[OAuthCredential, Abortable], Awaitable[OAuthCredential]]
"""Exchange the refresh token (Pi `OAuthAuth.refresh`, `types.ts:222`) -- a network call; TWO
required positional parameters (unbundled 1:1 from Pi's own two-argument signature), raises on
failure (invalid_grant etc.), no separate error channel. Already-certified `PROV-008`'s own
`refresh_if_expiring_at` runs this under the credential store's own lock, matching Pi's own
`Models`-level locked-refresh design."""

type OAuthToAuth = Callable[[OAuthCredential], Awaitable[ModelAuth]]
"""Side-effect-free derivation of request auth from a valid credential (Pi `OAuthAuth.toAuth`,
`types.ts:229`) -- covers per-credential `base_url` (e.g. GitHub Copilot). ONE required positional
parameter (unbundled 1:1 from Pi's own single-argument signature); async so a lazy wrapper can
load its own implementation on first use.

MAY RAISE/REJECT (`L11-SB-R003`, second independent review, CORRECTING an earlier revision's own
"not expected to raise" claim): confirmed directly against pinned Pi (`resolve.ts:174-178`,
`resolveStoredOAuth`) -- `await oauth.toAuth(credential)` runs inside a `try`/`catch`, wrapping a
rejection as an OAuth-derivation failure. Pi's own real call site treats this exactly like
`ApiKeyCheck`/`ApiKeyResolve` above (raising is a valid, expected outcome the future orchestration
owner wraps), NOT as a function that is side-effect-free THEREFORE never fails -- those are
independent properties; "side-effect-free" describes what `to_auth` does to the WORLD, not whether
it can fail on an unexpected/malformed credential."""


@dataclass(slots=True)
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
    is_subscription: bool | None = None
    """Whether access through this auth method is backed by a provider subscription (Pi
    `isSubscription?`, `types.ts:211`). GENUINELY three-valued, not two (`L11-SB-R001`,
    independent review): pinned Pi's own field is OPTIONAL (`boolean | undefined`), and an absent
    value is observably distinct from an explicit `false` -- collapsing "omitted" into a `bool`
    field defaulting to `False` would make those two states indistinguishable, silently narrowing
    Pi's own optional field. `None` here means "not stated," matching Pi's own absent/`undefined`
    exactly; only an explicit `True`/`False` means the auth method actually asserts a value either
    way."""
    login_label: str | None = None
    """Selector label for the OAuth login option, e.g. "Sign in with SuperGrok or X Premium" (Pi
    `loginLabel?`, `types.ts:214`)."""


@dataclass(slots=True)
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
