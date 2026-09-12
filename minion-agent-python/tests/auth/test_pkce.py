"""PKCE (RFC 7636) generation/derivation (`PROV-009`)."""

import base64
import re

from minion_agent.auth.pkce import VERIFIER_ENTROPY_BYTES, derive_pkce_challenge, generate_pkce

RFC7636_APPENDIX_B_VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
RFC7636_APPENDIX_B_CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

UNRESERVED_VERIFIER_CHARACTERS = re.compile(r"^[A-Za-z0-9\-._~]+$")


def test_derive_pkce_challenge_matches_the_rfc7636_known_vector() -> None:
    """RFC 7636 Appendix B's own worked example -- independently recomputed and confirmed
    (`hashlib.sha256` + base64url) before being pinned here as a permanent regression witness."""
    assert derive_pkce_challenge(RFC7636_APPENDIX_B_VERIFIER) == RFC7636_APPENDIX_B_CHALLENGE


def test_derive_pkce_challenge_is_deterministic() -> None:
    assert derive_pkce_challenge("same-verifier") == derive_pkce_challenge("same-verifier")


def test_derive_pkce_challenge_differs_for_different_verifiers() -> None:
    assert derive_pkce_challenge("verifier-a") != derive_pkce_challenge("verifier-b")


def test_generate_pkce_produces_a_matching_verifier_and_challenge() -> None:
    pair = generate_pkce()
    assert pair.challenge == derive_pkce_challenge(pair.verifier)


def test_generate_pkce_produces_two_different_verifiers_across_calls() -> None:
    first = generate_pkce()
    second = generate_pkce()
    assert first.verifier != second.verifier


def test_generate_pkce_verifier_meets_rfc7636_length_and_charset_constraints() -> None:
    """RFC 7636 section 4.1: the verifier must be 43-128 characters from
    `[A-Za-z0-9-._~]`. Pi's own choice of `VERIFIER_ENTROPY_BYTES` bytes yields exactly the
    MINIMUM length after base64url encoding -- this pins that exact length, not merely
    "within range", since a length regression would silently still pass a wider bound."""
    pair = generate_pkce()
    expected_length = -(-VERIFIER_ENTROPY_BYTES * 8 // 6)  # ceil(bytes * 8 bits / 6 bits-per-char)
    assert len(pair.verifier) == expected_length
    assert 43 <= len(pair.verifier) <= 128
    assert UNRESERVED_VERIFIER_CHARACTERS.match(pair.verifier)


def test_generate_pkce_challenge_is_base64url_without_padding() -> None:
    pair = generate_pkce()
    assert "=" not in pair.challenge
    assert "+" not in pair.challenge
    assert "/" not in pair.challenge
    # A valid base64url string decodes cleanly once re-padded.
    padding = "=" * (-len(pair.challenge) % 4)
    decoded = base64.urlsafe_b64decode(pair.challenge + padding)
    assert len(decoded) == 32  # SHA-256 digest size
