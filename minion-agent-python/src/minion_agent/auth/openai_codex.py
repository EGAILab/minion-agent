"""OpenAI Codex (ChatGPT OAuth) account-id projection, provider-specific and non-network
(`PROV-011`; Pi `packages/ai/src/auth/oauth/openai-codex.ts`).

This module owns ONLY the pure, non-network half of Codex's own OAuth flow: extracting
`chatgpt_account_id` from an access token's JWT payload, and projecting a stored Codex credential
to the bearer-style request auth every model request needs. The network half -- the browser/
local-callback-server login flow, the device-code endpoint integration, and the token exchange/
refresh HTTP calls -- is `PROV-012`'s own scope, a separate Layer 11 Pass 2 slice.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import cast

from .credential import JsonValue, ModelAuth, OAuthCredential

JWT_CLAIM_PATH = "https://api.openai.com/auth"
"""The namespaced claim key pinned Pi's own JWT payload nests `chatgpt_account_id` under (Pi
`JWT_CLAIM_PATH`, `openai-codex.ts:39`) -- a full URL string used as an object key, not a
path/pointer to traverse."""


def _decode_jwt_payload(token: str) -> JsonValue:
    """Decode a JWT's own middle (payload) segment, matching Pi's exact `atob(payload)` +
    `JSON.parse(...)` behavior (`openai-codex.ts::decodeJwt`) -- not a general-purpose or
    signature-validating JWT decoder. Returns `None` on any malformed input, exactly like Pi's own
    bare `try { ... } catch { return null; }`.

    Three JS/Python behaviors reproduced exactly, independently confirmed against a real Node
    process before implementing (not merely inferred from documentation):

    1. `token.split(".")` must yield EXACTLY 3 parts (`parts.length !== 3 -> null`) -- a malformed
       token with too few/many `.`-separated segments is rejected outright, before any decode is
       attempted.
    2. `atob` decodes STANDARD base64 only, not base64url -- it accepts a MISSING padding suffix
       (auto-pads internally) but REJECTS the base64url-specific `-`/`_` characters outright
       (confirmed live: `atob("-_-_")` throws `"Invalid character"`; a length whose remainder mod
       4 is exactly `1` is also always invalid, no amount of padding can repair it, and `atob`
       throws for that too). A genuinely base64url-encoded real-world JWT payload (one whose
       underlying bytes happen to need a `-`/`_` replacement character) therefore FAILS to decode
       through this exact function, matching Pi's own real, unfixed behavior -- this is Pi's own
       observable contract, not a Python-side bug to silently correct.
    3. `atob`'s own return value is a BINARY STRING (one JS char per decoded BYTE, i.e. Latin-1/
       ISO-8859-1 semantics), never a UTF-8 decode -- Pi's own code hands that binary string
       straight to `JSON.parse` with no intermediate UTF-8 re-decoding step. A JWT claim containing
       a genuine non-ASCII character (e.g. `"café"`, UTF-8 bytes `0xC3 0xA9` for `é`) therefore
       decodes to MOJIBAKE (`"cafÃ©"`, two separate Latin-1 codepoints `U+00C3`/`U+00A9`), not the
       original character -- independently confirmed byte-for-byte identical codepoints
       (`[0x63, 0x61, 0x66, 0xc3, 0xa9]`) between a live Node `atob`+`JSON.parse` run and this
       function's own `bytes.decode("latin-1")` + `json.loads`. This is Pi's own real, observable
       behavior for any non-ASCII claim value, not a Python-specific encoding bug -- faithfully
       reproducing it (rather than "fixing" it with a UTF-8 decode Pi itself never performs) is
       what Pi parity requires here.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    segment = parts[1]
    padded = segment + "=" * ((4 - len(segment) % 4) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        return cast(JsonValue, json.loads(raw.decode("latin-1")))
    except json.JSONDecodeError:
        return None


def get_account_id(access_token: str) -> str | None:
    """Extract `chatgpt_account_id` from an access token's JWT payload (Pi `getAccountId`,
    `openai-codex.ts::396-401`) -- UNVERIFIED CLAIM EXTRACTION ONLY, never cryptographic signature
    validation; this decodes what the token CLAIMS, it does not authenticate the token itself
    (Pi's own code performs no signature check anywhere in this flow either). Returns `None` for a
    malformed token, a payload missing the `JWT_CLAIM_PATH` claim, a non-string
    `chatgpt_account_id`, or an empty-string one -- matching Pi's own `typeof accountId ===
    "string" && accountId.length > 0` guard exactly, not merely "falsy"."""
    payload = _decode_jwt_payload(access_token)
    if not isinstance(payload, dict):
        return None
    auth = payload.get(JWT_CLAIM_PATH)
    if not isinstance(auth, dict):
        return None
    account_id = auth.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and len(account_id) > 0 else None


def credential_from_token(*, access: str, refresh: str, expires: float) -> OAuthCredential:
    """Project a raw OAuth token triple into a stored Codex credential (Pi `credentialsFromToken`,
    `openai-codex.ts::403-416`). The extracted account id rides on `OAuthCredential.extra`
    (`PROV-006`) under the `"accountId"` key -- Pi's own object literal spreads `accountId` as a
    sibling field on the credential itself, but this project's own `OAuthCredential` type keeps
    every provider-specific extension inside its single open `extra` escape hatch rather than
    growing a new named field per provider (see `PROV-006`'s own manifest row), so `accountId`
    lands there, not as a new dataclass field.

    Raises `ValueError` when no account id can be extracted -- matching Pi's own `throw new
    Error("Failed to extract accountId from token")` exactly: a token that does not carry a usable
    `chatgpt_account_id` claim is a login-flow failure, not a partially-successful credential."""
    account_id = get_account_id(access)
    if not account_id:
        raise ValueError("Failed to extract accountId from token")
    return OAuthCredential(
        access=access, refresh=refresh, expires=expires, extra={"accountId": account_id}
    )


def to_auth(credential: OAuthCredential) -> ModelAuth:
    """Derive provider-facing request auth from a stored Codex credential (Pi `OpenAICodexOAuth.
    toAuth`, `openai-codex.ts::541-543`): a Codex access token IS the bearer token every model
    request needs, with no additional headers or base-url override -- `ModelAuth(api_key=...)`
    only, matching Pi's own `{ apiKey: credential.access }` exactly."""
    return ModelAuth(api_key=credential.access)
