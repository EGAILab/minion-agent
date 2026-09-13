"""PKCE (RFC 7636) code verifier/challenge generation, provider-neutral (`PROV-009`; Pi
`generatePKCE`, `auth/oauth/pkce.ts`).
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass

VERIFIER_ENTROPY_BYTES = 32
"""Pi's own choice (`auth/oauth/pkce.ts::generatePKCE`): 32 random bytes, base64url-encoded,
yields a 43-character verifier -- the RFC-7636 MINIMUM allowed length (section 4.1), not an
arbitrary round number."""


@dataclass(frozen=True, slots=True)
class PkcePair:
    """A generated PKCE verifier/challenge pair."""

    verifier: str
    challenge: str


def _base64url_encode(data: bytes) -> str:
    """Base64url without padding -- matches Pi's own `btoa(...).replace(/\\+/g, "-")
    .replace(/\\//g, "_").replace(/=/g, "")` chain, and stays within RFC 7636's own allowed
    verifier/challenge character set (`[A-Za-z0-9-._~]`), of which base64url's alphabet is a
    subset."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def derive_pkce_challenge(verifier: str) -> str:
    """The deterministic half of PKCE: `challenge = base64url(SHA-256(verifier))` (RFC 7636
    section 4.2, `code_challenge_method=S256`) -- the SAME derivation `generate_pkce()` uses
    internally, exposed separately so it is directly testable against a KNOWN verifier without
    depending on random generation."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _base64url_encode(digest)


def generate_pkce() -> PkcePair:
    """Generate a fresh PKCE verifier/challenge pair (Pi `generatePKCE`): a
    `VERIFIER_ENTROPY_BYTES`-byte cryptographically random verifier, base64url-encoded, and its
    SHA-256 challenge."""
    verifier = _base64url_encode(secrets.token_bytes(VERIFIER_ENTROPY_BYTES))
    return PkcePair(verifier=verifier, challenge=derive_pkce_challenge(verifier))
