"""OpenAI Codex account-id projection (`PROV-011`) -- synthetic JWT-shaped fixtures only, never a
real account/token, per this project's own security constraint against live secrets in tests.
"""

from __future__ import annotations

import base64
import json

import pytest

from minion_agent.auth.credential import ModelAuth
from minion_agent.auth.openai_codex import (
    JWT_CLAIM_PATH,
    credential_from_token,
    get_account_id,
    to_auth,
)


def _json_bytes(payload: dict[str, object]) -> bytes:
    """`ensure_ascii=False`, matching JS's own `JSON.stringify` -- which does NOT escape
    non-ASCII characters by default, unlike `json.dumps`'s own default. Using the default here
    would silently escape a non-ASCII fixture value (e.g. `"café"` -> the ASCII text `caf\\u00e9`)
    before it ever reached UTF-8 encoding, masking the exact mojibake behavior this module's own
    tests need to pin."""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _b64url_segment(payload: dict[str, object]) -> str:
    """A JWT-shaped base64URL payload segment (unpadded) -- what a REAL Codex token's own middle
    segment looks like. `get_account_id` must still resolve it, since `_decode_jwt_payload`'s own
    padding-leniency covers the missing `=` suffix, even though it only accepts the STANDARD
    (non-url-safe) alphabet -- these synthetic fixtures are deliberately built to avoid any
    `-`/`_` character so they decode successfully, matching realistic Codex claim shapes."""
    return base64.urlsafe_b64encode(_json_bytes(payload)).rstrip(b"=").decode("ascii")


def _b64_segment(payload: dict[str, object]) -> str:
    """A JWT-shaped STANDARD (non-url-safe) base64 payload segment, padded -- the exact alphabet
    `atob` actually accepts."""
    return base64.b64encode(_json_bytes(payload)).decode("ascii")


def _token(payload: dict[str, object], *, url_safe: bool = False) -> str:
    segment = _b64url_segment(payload) if url_safe else _b64_segment(payload)
    return f"header.{segment}.signature"


def test_get_account_id_extracts_the_namespaced_claim() -> None:
    token = _token({JWT_CLAIM_PATH: {"chatgpt_account_id": "acct_123"}})
    assert get_account_id(token) == "acct_123"


def test_get_account_id_accepts_unpadded_base64url_payload() -> None:
    """`_decode_jwt_payload`'s own padding leniency: a real Codex token's payload segment has no
    `=` padding at all -- this must still decode, matching `atob`'s own auto-padding behavior."""
    token = _token({JWT_CLAIM_PATH: {"chatgpt_account_id": "acct_456"}}, url_safe=True)
    assert get_account_id(token) == "acct_456"


def test_get_account_id_returns_none_for_a_token_without_three_segments() -> None:
    assert get_account_id("only.two") is None
    assert get_account_id("one") is None
    assert get_account_id("a.b.c.d") is None


def test_get_account_id_returns_none_for_base64url_specific_characters() -> None:
    """`atob` decodes STANDARD base64 only: `-`/`_` (base64url's own replacements for `+`/`/`)
    are rejected outright, independently confirmed live against a real Node process
    (`atob("-_-_")` throws `"Invalid character"`) before pinning this witness. `-_-_` happens to
    decode to an empty payload either way once non-alphabet characters are discarded, so this
    witness alone does not distinguish strict rejection (`base64.b64decode(..., validate=True)`)
    from Python's own default lenient discard-and-decode -- both return `None` for THIS input, one
    by outright rejection, the other because the resulting empty string fails JSON parsing.
    `validate=True` remains the semantically exact match for `atob`'s own documented behavior
    regardless; a corrupted (but non-empty) base64 segment realistically also fails JSON parsing
    after lenient discarding, since discarding shifts every subsequent 4-character bit grouping,
    which is why a byte-for-byte discriminating witness for this specific axis was not practical to
    construct."""
    assert get_account_id("header.-_-_.signature") is None


def test_get_account_id_returns_none_for_an_invalid_base64_length() -> None:
    """A payload segment length whose remainder mod 4 is exactly 1 is never valid base64, no
    amount of padding can repair it -- independently confirmed live against a real Node process
    (`atob` throws `"The string to be decoded is not correctly encoded."`) before pinning this
    witness."""
    assert get_account_id("header.eyJhIjoxf.signature") is None


def test_get_account_id_returns_none_when_the_payload_is_not_valid_json() -> None:
    garbage = base64.b64encode(b"not json").decode("ascii")
    assert get_account_id(f"header.{garbage}.signature") is None


def test_get_account_id_returns_none_when_the_claim_path_is_absent() -> None:
    token = _token({"unrelated": "claim"})
    assert get_account_id(token) is None


def test_get_account_id_returns_none_when_chatgpt_account_id_is_not_a_string() -> None:
    token = _token({JWT_CLAIM_PATH: {"chatgpt_account_id": 12345}})
    assert get_account_id(token) is None


def test_get_account_id_returns_none_for_an_empty_string_account_id() -> None:
    """Matches Pi's own `typeof accountId === "string" && accountId.length > 0` guard exactly --
    an empty string is a STRING, but still rejected, not merely "falsy" in the Python sense."""
    token = _token({JWT_CLAIM_PATH: {"chatgpt_account_id": ""}})
    assert get_account_id(token) is None


def test_get_account_id_reproduces_pis_own_mojibake_for_non_ascii_claims() -> None:
    """`atob`'s own return value is a Latin-1 binary string, never UTF-8-decoded, so a genuine
    non-ASCII claim value decodes to MOJIBAKE, not the original character -- independently
    confirmed byte-for-byte identical codepoints (`[0x63, 0x61, 0x66, 0xc3, 0xa9]`, i.e.
    `"caf" + U+00C3 + U+00A9`) between a live Node `atob`+`JSON.parse` run and this project's own
    `bytes.decode("latin-1")` + `json.loads`, before pinning this as a permanent witness. This is
    Pi's own real, observable behavior for a non-ASCII claim -- not a bug this port should "fix"
    with a UTF-8 decode Pi itself never performs."""
    token = _token({JWT_CLAIM_PATH: {"chatgpt_account_id": "café"}})
    account_id = get_account_id(token)
    assert account_id is not None
    assert [hex(ord(char)) for char in account_id] == ["0x63", "0x61", "0x66", "0xc3", "0xa9"]


def test_credential_from_token_stores_the_account_id_on_extra() -> None:
    access = _token({JWT_CLAIM_PATH: {"chatgpt_account_id": "acct_789"}})
    credential = credential_from_token(access=access, refresh="refresh-token", expires=1_000.0)
    assert credential.access == access
    assert credential.refresh == "refresh-token"
    assert credential.expires == 1_000.0
    assert credential.extra == {"accountId": "acct_789"}


def test_credential_from_token_raises_when_no_account_id_can_be_extracted() -> None:
    access = _token({"unrelated": "claim"})
    with pytest.raises(ValueError, match="Failed to extract accountId from token"):
        credential_from_token(access=access, refresh="refresh-token", expires=1_000.0)


def test_to_auth_projects_the_access_token_as_a_bearer_api_key() -> None:
    access = _token({JWT_CLAIM_PATH: {"chatgpt_account_id": "acct_bearer"}})
    credential = credential_from_token(access=access, refresh="refresh-token", expires=1_000.0)
    assert to_auth(credential) == ModelAuth(api_key=access)
