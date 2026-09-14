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
import re
from typing import cast

from .credential import JsonValue, ModelAuth, OAuthCredential

JWT_CLAIM_PATH = "https://api.openai.com/auth"
"""The JWT payload key namespacing Codex's own custom claims (Pi `openai-codex.ts:39`)."""

_ASCII_WHITESPACE = re.compile(r"[\t\n\f\r ]")
"""The exact five code points the WHATWG "forgiving-base64 decode" algorithm strips before
decoding (`infra.spec.whatwg.org/#forgiving-base64-decode` step 1; `atob` implements this
algorithm) -- TAB, LF, FF, CR, SPACE. Not general Unicode whitespace."""


def _forgiving_base64_decode(data: str) -> bytes:
    """A faithful port of the WHATWG "forgiving-base64 decode" algorithm `atob` actually
    implements (`L11-SA-R001`) -- NOT `base64.b64decode`'s own, materially different, padding and
    character-validity rules. Independently reverse-engineered empirically against a live Node v22
    process (both the exact stripping/padding behavior below and the error/success boundary for
    every case), not assumed from a half-remembered spec summary:

    1. Remove EVERY ASCII whitespace character (`_ASCII_WHITESPACE`) anywhere in the string,
       regardless of position -- leading, trailing, or interior (`atob("  MTIzNA==  ")` and
       `atob("MTIz\\tNA==")` both succeed, decoding identically to the unspaced form).
    2. If the (whitespace-stripped) length is a multiple of 4 AND the string ends with EXACTLY one
       or two `=` characters, strip that trailing padding before further validation. This is the
       ONLY circumstance under which a `=` is ever accepted -- unlike `base64.b64decode`'s own
       tolerant/padding-optional behavior, a `=` that does not satisfy this exact condition (wrong
       position, wrong count, or a length that was never a multiple of 4 to begin with) makes the
       WHOLE input invalid, not merely un-paddable. `atob("MTIzNA=")` (length 7, not a multiple of
       4) THROWS, even though Python's own `base64.b64decode` would happily re-pad and decode it
       to `"1234"` -- silently repairing bona fide malformed padding Pi itself rejects.
    3. If the remaining length is a multiple of 4 leaving remainder 1, the input is invalid
       (`atob("MTIzN")` throws) -- no valid base64 grouping can end in exactly one leftover
       character.
    4. Every remaining character must be in the standard base64 alphabet (`A-Za-z0-9+/`) -- a
       stray `=` anywhere else (not satisfying step 2's exact trailing condition) fails this check,
       exactly like any other invalid character; base64url's `-`/`_` are likewise rejected here,
       already covered by this same step (no separate check needed).

    Raises `ValueError` for any input that fails these rules; the caller (`decode_jwt`) treats that
    identically to every other decode failure."""
    stripped = _ASCII_WHITESPACE.sub("", data)
    trailing_equals = len(stripped) - len(stripped.rstrip("="))
    if len(stripped) % 4 == 0 and trailing_equals in (1, 2):
        content = stripped[: len(stripped) - trailing_equals]
    else:
        content = stripped
    remainder = len(content) % 4
    if remainder == 1:
        raise ValueError("forgiving-base64 decode: invalid length (remainder 1)")
    if not re.fullmatch(r"[A-Za-z0-9+/]*", content):
        raise ValueError("forgiving-base64 decode: invalid character")
    repadded = content + "=" * (-len(content) % 4)
    return base64.b64decode(repadded, validate=True)


def _reject_js_incompatible_constant(token: str) -> float:
    """Python's own `json.loads` accepts the bare tokens `NaN`/`Infinity`/`-Infinity` as an
    extension beyond the JSON grammar (`parse_constant`'s own default); JavaScript's `JSON.parse`
    does NOT -- independently confirmed live (`JSON.parse("NaN")`/`JSON.parse("Infinity")` both
    throw `"... is not valid JSON"`). Wiring this as `json.loads`'s own `parse_constant` hook makes
    Python raise for exactly the same three tokens Pi's `JSON.parse` rejects (`L11-SA-R001`)."""
    raise ValueError(f"not valid JSON: unexpected token {token!r}")


def decode_jwt(token: str) -> JsonValue:
    """Decode a JWT's payload segment WITHOUT verifying its signature (Pi `decodeJwt`,
    `openai-codex.ts:103-113`) -- UNVERIFIED claim extraction only, matching Pi's own comment and
    this project's own security constraint against treating an unverified claim as authenticated
    identity.

    Returns whatever the JS-`JSON.parse`-faithful decode below produces for the decoded payload
    segment -- NOT necessarily a dict. Pi's own implementation never checks the parsed shape
    either (`JSON.parse(decoded) as JwtPayload` is a compile-time-only assertion with no runtime
    check); a non-object result is tolerated by `get_account_id`'s own graceful lookup below,
    exactly where Pi's own optional chaining (`payload?.[JWT_CLAIM_PATH]`) does the same
    tolerance, not here. A payload segment that decodes to the JSON literal `null` is therefore
    indistinguishable from a malformed token -- a faithfully-reproduced Pi ambiguity
    (`JSON.parse("null") === null` in Pi too), not an accident of this port.

    Faithfully reproduces FOUR independently-live-Node-cross-checked (v22) `atob`/`JSON.parse`
    fidelity gaps a naive `base64.b64decode`/`json.loads` port misses (`L11-SA-R001`, found by
    independent review after an initial candidate covered only the two below marked *):

    1. *`atob` decodes via the WHATWG "forgiving-base64" algorithm, NOT `base64.b64decode`'s own
       rules -- see `_forgiving_base64_decode`'s own docstring for the exact, empirically-verified
       grammar (whitespace stripping, and the narrow padding-strip condition that makes genuinely
       malformed padding -- e.g. a single `=` where two are required -- an error, not something to
       silently repair).
    2. *`atob`'s return value is a LATIN-1 binary string, NEVER UTF-8-decoded -- each output byte
       becomes one JS UTF-16 code unit 0-255, not a decoded Unicode codepoint. A JWT payload
       containing a non-ASCII claim VALUE (encoded as UTF-8 bytes before base64, the universal way
       JWTs are built) therefore decodes to MOJIBAKE under `JSON.parse(atob(...))`, not the
       original character -- e.g. UTF-8 `é` (bytes `0xC3 0xA9`) becomes the two separate Latin-1
       characters `Ã©`, not `é`. Faithfully reproduced (`bytes.decode("latin-1")` before parsing),
       not "fixed": Pi's own real, unverified-claim-extraction behavior mangles a non-ASCII claim
       the same way.
    3. JS `JSON.parse` REJECTS the bare tokens `NaN`/`Infinity`/`-Infinity` as invalid JSON, while
       Python's own `json.loads` accepts them by default as a non-standard extension
       (`parse_constant`) -- rejected here via `_reject_js_incompatible_constant` so a payload
       containing one of these tokens anywhere fails to decode, exactly like Pi.
    4. JS `JSON.parse` parses EVERY number (including integer literals) as an IEEE-754 double,
       losing precision beyond 2**53 and preserving signed zero (`JSON.parse("9007199254740993")
       === 9007199254740992`; `JSON.parse("-0")` is `-0`, distinct from `+0` under `Object.is`).
       Python's own `json.loads` instead parses an integer literal to an exact, arbitrary-precision
       `int`, silently preserving precision JS would lose and losing the sign of zero JS would
       keep. Matched here via `parse_int=float`, which round-trips every integer literal through
       the SAME IEEE-754-double coercion (`float("9007199254740993") ==
       9007199254740992.0`; `float("-0") == -0.0`, sign-preserving) JS's own single number type
       already performs unconditionally.

    Returns `None` for a malformed token (not exactly three `.`-separated segments, invalid
    base64, or invalid JSON) -- Pi's own bare `try { ... } catch { return null; }` around the
    whole operation, filtering no particular error type."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    try:
        raw = _forgiving_base64_decode(payload)
        decoded = raw.decode("latin-1")
        return cast(
            JsonValue,
            json.loads(decoded, parse_int=float, parse_constant=_reject_js_incompatible_constant),
        )
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
