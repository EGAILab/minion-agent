"""OpenAI Codex (ChatGPT OAuth) account-id projection (`PROV-011`; Pi
`packages/ai/src/auth/oauth/openai-codex.ts`).

Pass-2 Slice A: the pure, non-network half of Codex's own OAuth module -- JWT payload decode for
`chatgpt_account_id` (unverified claim extraction, no signature validation) and the OAuth
access-token -> bearer `ModelAuth` projection. The network half (browser/device-code login, token
exchange/refresh against `auth.openai.com`) is `PROV-012`, a separate Pass-2 slice.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import cast

from .credential import JsonValue, ModelAuth, OAuthCredential

JWT_CLAIM_PATH = "https://api.openai.com/auth"
"""The JWT payload key namespacing Codex's own custom claims (Pi `openai-codex.ts:39`)."""


def decode_jwt(token: str) -> JsonValue:
    """Decode a JWT's payload segment WITHOUT verifying its signature (Pi `decodeJwt`,
    `openai-codex.ts:103-113`) -- UNVERIFIED claim extraction only, matching Pi's own comment and
    this project's own security constraint against treating an unverified claim as authenticated
    identity.

    Returns whatever `json.loads` produces for the decoded payload segment -- NOT necessarily a
    dict. Pi's own implementation never checks the parsed shape either (`JSON.parse(decoded) as
    JwtPayload` is a compile-time-only assertion with no runtime check); a non-object result is
    tolerated by `get_account_id`'s own graceful lookup below, exactly where Pi's own optional
    chaining (`payload?.[JWT_CLAIM_PATH]`) does the same tolerance, not here. A payload segment
    that decodes to the JSON literal `null` is therefore indistinguishable from a malformed
    token -- a faithfully-reproduced Pi ambiguity (`JSON.parse("null") === null` in Pi too), not
    an accident of this port.

    Faithfully reproduces two Pi/JS-specific `atob` quirks, independently confirmed against a
    live Node process (v22) before implementing, not inferred from documentation:

    1. `atob` decodes STANDARD base64 only -- it REJECTS base64url's `-`/`_` alphabet characters
       outright (`DOMException: Invalid character`), while ACCEPTING missing padding. Python's own
       `base64.b64decode(..., validate=True)` matches the first half (rejects any non-standard-
       alphabet character) but, unlike `atob`, raises on missing padding rather than tolerating
       it -- so this function pads the input to a multiple of 4 before decoding, to match `atob`'s
       own more lenient padding behavior while still rejecting `-`/`_`.
    2. `atob`'s return value is a LATIN-1 binary string, NEVER UTF-8-decoded -- each output byte
       becomes one JS UTF-16 code unit 0-255, not a decoded Unicode codepoint. A JWT payload
       containing a non-ASCII claim VALUE (encoded as UTF-8 bytes before base64, the universal way
       JWTs are built) therefore decodes to MOJIBAKE under `JSON.parse(atob(...))`, not the
       original character -- e.g. UTF-8 `é` (bytes `0xC3 0xA9`) becomes the two separate Latin-1
       characters `Ã©`, not `é`. This is faithfully reproduced (`bytes.decode("latin-1")` before
       `json.loads`), not "fixed": Pi's own real, unverified-claim-extraction behavior would
       silently mangle a non-ASCII claim the same way. A realistic `chatgpt_account_id` value is
       ASCII and unaffected either way, but the WHOLE payload must still parse successfully even
       when some OTHER claim value is non-ASCII, exactly as Pi's own `JSON.parse` on the Latin-1
       string does -- JSON's own structural syntax is pure ASCII, so only string CONTENT
       mojibakes, never the parse itself.

    Returns `None` for a malformed token (not exactly three `.`-separated segments, invalid
    base64, or invalid JSON) -- Pi's own bare `try { ... } catch { return null; }` around the
    whole operation, filtering no particular error type."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    try:
        padded = payload + "=" * (-len(payload) % 4)
        raw = base64.b64decode(padded, validate=True)
        decoded = raw.decode("latin-1")
        return cast(JsonValue, json.loads(decoded))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None


def get_account_id(access_token: str) -> str | None:
    """Extract `chatgpt_account_id` from an access token's own JWT payload (Pi `getAccountId`,
    `openai-codex.ts:396-401`) -- `None` if the token is malformed, `decode_jwt`'s own result is
    not an object, the claim namespace (`JWT_CLAIM_PATH`) is missing or not an object, or the
    claim itself is not a non-empty string. Mirrors Pi's own chain of optional-chaining lookups
    (`payload?.[JWT_CLAIM_PATH]`, `auth?.chatgpt_account_id`) one graceful `None`-check at a
    time, rather than requiring `decode_jwt` itself to pre-validate the parsed shape."""
    payload = decode_jwt(access_token)
    if not isinstance(payload, dict):
        return None
    auth_claims = payload.get(JWT_CLAIM_PATH)
    if not isinstance(auth_claims, dict):
        return None
    account_id = auth_claims.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and len(account_id) > 0 else None


def credentials_from_token(access: str, refresh: str, expires: float) -> OAuthCredential:
    """Build the stored `OAuthCredential` for a freshly-obtained or refreshed Codex token (Pi
    `credentialsFromToken`, `openai-codex.ts:403-416`).

    MINION-SPECIFIC MAPPING, not direct Pi parity (`PROV-006`'s own docstring already anticipates
    this exact case): pinned Pi's own `OAuthCredential` type carries `accountId` as a literal
    top-level field. This project's own `OAuthCredential` (`PROV-006`) instead keeps
    provider-specific attachments under the OPEN `extra` escape hatch -- `extra["account_id"]`
    here -- rather than widening the shared credential shape every OTHER provider's own OAuth flow
    also uses, for one Codex-specific field. This is a disclosed architectural mapping, not a
    claimed direct-parity field placement.

    Raises `ValueError` if the token's own JWT payload does not carry a usable
    `chatgpt_account_id` -- matching Pi's own `throw new Error("Failed to extract accountId from
    token")`. Takes `access`/`refresh`/`expires` as plain scalars rather than a Pi-shaped
    `OAuthToken` object: that type belongs to `PROV-012`'s own network/token-exchange layer (a
    separate Pass-2 slice not yet implemented), and this function does not presuppose its shape."""
    account_id = get_account_id(access)
    if account_id is None:
        raise ValueError("Failed to extract accountId from token")
    return OAuthCredential(
        access=access, refresh=refresh, expires=expires, extra={"account_id": account_id}
    )


def to_auth(credential: OAuthCredential) -> ModelAuth:
    """Project a stored Codex OAuth credential to provider-facing bearer auth (Pi `toAuth`,
    `openai-codex.ts:541-543`) -- a trivial, side-effect-free `{apiKey: credential.access}`
    projection; Codex's own request auth is the raw access token used as a bearer/API key."""
    return ModelAuth(api_key=credential.access)
