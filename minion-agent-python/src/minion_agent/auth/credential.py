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


@dataclass(frozen=True, slots=True)
class ApiKeyCredential:
    """Stored api-key credential (Pi `ApiKeyCredential`, `auth/types.ts:17-21`). `env` carries
    provider-scoped environment/config values (e.g. Cloudflare account/gateway ids) -- not
    request-time headers, which live on `ModelAuth` once auth has been resolved."""

    key: str | None = None
    env: Mapping[str, str] | None = None
    type: Literal["api_key"] = "api_key"


@dataclass(frozen=True, slots=True)
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
    a closed shape Pi itself leaves open.
    """

    access: str
    refresh: str
    expires: float
    extra: Mapping[str, JsonValue] = field(default_factory=dict)
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

    `env(name)` returns the named environment value, or `None` if absent OR present-but-blank --
    Pi's own `defaultProviderAuthContext` treats whitespace-only values as absent
    (`auth/context.ts:25-28`), and this protocol's own contract preserves that, not merely its
    default implementation. `file_exists(path)` reports whether `path` exists, with a leading `~`
    expanded to the user's home directory (`auth/context.ts:30-43`).
    """

    async def env(self, name: str) -> str | None: ...

    async def file_exists(self, path: str) -> bool: ...
