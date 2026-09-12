"""Auth vocabulary: stored credentials, resolved request auth, and the auth-context seam
(`PROV-006`; Pi `packages/ai/src/auth/types.ts`, `auth/context.ts`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..runtime.signal import RunSignal

type JsonValue = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

AuthType = Literal["api_key", "oauth"]
"""The closed set of credential kinds (Pi `AuthType`, `auth/types.ts:117`)."""


@dataclass(slots=True)
class ApiKeyCredential:
    """Stored api-key credential (Pi `ApiKeyCredential`, `auth/types.ts:17-21`). `env` carries
    provider-scoped environment/config values (e.g. Cloudflare account/gateway ids) -- not
    request-time headers, which live on `ModelAuth` once auth has been resolved. `env`'s own
    domain is pinned Pi's own `ProviderEnv = Record<string, string>` (`types.ts:113`) -- a FLAT
    string-to-string mapping, never recursive JSON (`L11-R010`; recursive values belong to
    `OAuthCredential.extra`'s own open domain below, not here).

    This dataclass is NOT frozen, and `env` is a genuinely MUTABLE `dict` (`L11-R006`/`L11-R009`,
    resolved by explicit owner governance decision, `agent-workflow.md` §11.7/§11.8: adopt pinned
    Pi's own live-reference/shared-mutation behavior in full, no intentional divergence approved --
    including ordinary SCALAR field reassignment, not only nested-value mutation). Pi's own
    `InMemoryCredentialStore` holds and returns live references to ordinary mutable objects, so:
    reassigning `credential.key` directly, mutating the ORIGINAL mapping passed to this
    constructor, and assigning a brand-new top-level key on `credential.env` itself, all remain
    observable through this credential afterward, including through a later `CredentialStore.
    read()` for the same provider id. `CredentialStore.modify()` remains the documented, INTENDED
    sole mutation path (`PROV-007`) -- this is unaffected by and does not depend on this aliasing
    behavior; a caller that instead mutates a retained reference directly bypasses that
    convention, exactly as Pi's own plain-object credential type permits (see `PROV-007`'s own
    manifest row)."""

    key: str | None = None
    env: dict[str, str] | None = None
    type: Literal["api_key"] = "api_key"


@dataclass(slots=True)
class OAuthCredential:
    """Stored canonical OAuth credential (Pi `OAuthCredential`, `auth/types.ts:24-34`).

    `expires` is a Unix-epoch-MILLISECONDS timestamp (matching Pi's own `Date.now() +
    expires_in * 1000`), not a duration and not seconds -- a credential is "expiring soon" when
    `now_ms + minimum_validity_ms >= expires`, never a raw comparison against a duration.

    `extra` is an OPEN escape hatch, not a closed field set: it mirrors Pi's own
    `OAuthCredentials`'s index signature (`[key: string]: unknown`), which lets a provider's own
    login flow attach additional fields alongside the three required ones -- for example Codex's
    own `accountId`, extracted from the access token's JWT payload (`PROV-011`, deferred in this
    pass). `extra` is empty until a provider-specific flow populates it; this row does not invent
    a closed shape Pi itself leaves open. Unlike `ApiKeyCredential.env`, `extra`'s own domain is
    full recursive JSON (`L11-R010`): pinned Pi's own index signature is `[key: string]: unknown`,
    not `ProviderEnv`'s flat string-to-string shape.

    This dataclass is NOT frozen, and `extra` is a genuinely MUTABLE `dict`, for the exact same
    owner-decided Pi-parity reason `ApiKeyCredential` is not frozen (`L11-R006`/`L11-R009`, see its
    own docstring) -- scalar fields (`access`, `refresh`, `expires`) are reassignable directly, and
    `extra`'s own nested values, and brand-new top-level keys on `extra` itself, all remain mutable
    and observable through this credential afterward, recursively, including through a later
    `CredentialStore.read()` for the same provider id.
    """

    access: str
    refresh: str
    expires: float
    extra: dict[str, JsonValue] = field(default_factory=dict)
    type: Literal["oauth"] = "oauth"


Credential = ApiKeyCredential | OAuthCredential
"""One type-tagged credential per provider (Pi `Credential`, `auth/types.ts:37`) -- the shape a
`CredentialStore` persists, one entry per provider id."""


@dataclass(frozen=True, slots=True)
class CredentialInfo:
    """Non-secret credential metadata for account/status enumeration (Pi `CredentialInfo`,
    `auth/types.ts:40-43`) -- never carries `key`/`access`/`refresh`/`extra`."""

    provider_id: str
    type: AuthType


@dataclass(frozen=True, slots=True)
class AuthOperationOptions:
    """Optional cancellation for public auth and credential operations (Pi
    `AuthOperationOptions`, `auth/types.ts:46-48`). `signal` is Minion's own established
    `RunSignal` (Layer 09) -- poll-based by certified design, never a push notification; callers
    that need it checked promptly during a long wait see `abortable_sleep`'s own short-interval
    polling (`device_code.py`), not a redesign of `RunSignal` itself."""

    signal: RunSignal | None = None


@dataclass(frozen=True, slots=True)
class ModelAuth:
    """Resolved request auth for one model request (Pi `ModelAuth`, `auth/types.ts:7-11`). If a
    value cannot be expressed as `api_key`/`headers`/`base_url`, it is provider CONFIG, not auth --
    this type is deliberately closed to those three fields, matching Pi's own comment exactly."""

    api_key: str | None = None
    headers: Mapping[str, str] | None = None
    base_url: str | None = None


@dataclass(frozen=True, slots=True)
class AuthResult:
    """Result of resolving auth for a model (Pi `AuthResult`, `auth/types.ts:104-110`). `source`
    is a human-readable status-UI label ("ANTHROPIC_API_KEY", "OAuth", "~/.aws/credentials"), not
    a machine-discriminated enum -- Pi's own comment states this explicitly."""

    auth: ModelAuth
    env: Mapping[str, str] | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class AuthCheck:
    """Side-effect-free availability-check result (Pi `AuthCheck`, `auth/types.ts:112-115`) --
    distinct from `AuthResult`: a `check` may run before request-time work an actual `resolve()`
    would perform, so it reports only enough to describe availability, never full request auth."""

    type: AuthType
    source: str | None = None


class AuthContext(Protocol):
    """Environment access for auth resolution, injectable for tests (Pi `AuthContext`,
    `auth/types.ts:97-101`).

    `env(name)` returns the named environment value, or `None` if genuinely absent. Pi's own
    interface (`types.ts:97-100`) permits ANY implementation to resolve a present-but-blank value
    (e.g. `""`) unchanged -- blank-to-absent normalization is NOT part of this protocol's own
    contract (`L11-R003`: an earlier revision of this docstring incorrectly claimed it was). Only
    `DefaultAuthContext` (this module's own default implementation, matching Pi's own
    `defaultProviderAuthContext`, `auth/context.ts:25-28`) treats a whitespace-only value as
    absent -- a property of THAT implementation, not a requirement every `AuthContext` must
    satisfy. A caller-supplied test context is free to return `""` for a name it considers
    "present but empty," and that is conforming, not a bug.

    `file_exists(path)` reports whether `path` exists, with a leading `~` expanded to the user's
    home directory -- UNLIKE `env`'s own blank-normalization, tilde support IS part of this
    PROTOCOL's own contract, not merely `DefaultAuthContext`'s concrete behavior (`L11-R008`: an
    earlier revision of this docstring incorrectly over-generalized the `L11-R003` fix to cover
    `file_exists` too). Pi's own interface doc comment states this directly on the interface
    method itself (`types.ts:99-100`: "Check whether a file exists. Supports a leading `~`. Always
    false in browsers."), unlike `env`, which carries no interface-level comment about blank
    values at all -- only `defaultProviderAuthContext`'s own doc comment (`auth/context.ts`)
    mentions that. ANY conforming `AuthContext` implementation -- not only `DefaultAuthContext` --
    must interpret a leading `~` as the user's home directory, not a literal relative path
    component. Pi's own "always false in browsers" clause is architecturally inapplicable here:
    this project has no browser runtime target, so no implementation needs a browser-specific
    branch to satisfy this contract.
    """

    async def env(self, name: str) -> str | None: ...

    async def file_exists(self, path: str) -> bool: ...
