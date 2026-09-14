"""Codex (ChatGPT OAuth) account-id projection (`PROV-011`) -- pure JWT decode, no network."""

import pytest

from minion_agent.auth.credential import ModelAuth, OAuthCredential
from minion_agent.auth.openai_codex import (
    JWT_CLAIM_PATH,
    credentials_from_token,
    decode_jwt,
    get_account_id,
    to_auth,
)

# Cross-checked directly against a live Node v22 process (`atob` + `JSON.parse`), not inferred
# from documentation. Header `{"alg":"none","typ":"JWT"}`; payload `{"https://api.openai.com/
# auth":{"chatgpt_account_id":"acct_synthetic_test_123"},"email":"héllo@example.com"}` (the email
# claim's own non-ASCII character is the witness for the Latin-1/mojibake decode quirk below);
# signature segment is the arbitrary literal `sig` (never verified, matching Pi's own unverified
# extraction). No real account, token, or secret -- entirely synthetic.
VALID_TOKEN = (
    "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0=."
    "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF9zeW50aGV0aWNf"
    "dGVzdF8xMjMifSwiZW1haWwiOiJow6lsbG9AZXhhbXBsZS5jb20ifQ==."
    "sig"
)
VALID_TOKEN_ACCOUNT_ID = "acct_synthetic_test_123"
# The exact mojibake Node produced for the UTF-8 `é` (bytes 0xC3 0xA9) claim value, decoded as
# Latin-1 instead of UTF-8: two separate characters, not the original `é`.
VALID_TOKEN_MOJIBAKE_EMAIL = "hÃ©llo@example.com"


def test_decode_jwt_returns_the_full_parsed_payload() -> None:
    payload = decode_jwt(VALID_TOKEN)
    assert isinstance(payload, dict)
    assert payload[JWT_CLAIM_PATH] == {"chatgpt_account_id": VALID_TOKEN_ACCOUNT_ID}


def test_decode_jwt_latin1_mojibake_matches_live_node_cross_check() -> None:
    """`L11-R011-DECODE`: `atob`'s return value is a LATIN-1 binary string, never UTF-8-decoded --
    a non-ASCII claim value decodes to mojibake, not the original character. Pinned to the exact
    codepoints independently cross-checked against a live Node process, not merely "some
    corruption occurs"."""
    payload = decode_jwt(VALID_TOKEN)
    assert isinstance(payload, dict)
    assert payload["email"] == VALID_TOKEN_MOJIBAKE_EMAIL


def test_decode_jwt_mojibake_is_genuinely_from_latin1_not_utf8() -> None:
    """Revert-and-confirm witness pinned directly in the test: decoding the SAME raw bytes as
    UTF-8 instead of Latin-1 recovers the original, non-mojibake character -- proving the mojibake
    above is a real consequence of the Latin-1 choice, not an unrelated encoding artifact."""
    import base64

    parts = VALID_TOKEN.split(".")
    raw = base64.b64decode(parts[1] + "=" * (-len(parts[1]) % 4), validate=True)
    assert raw.decode("utf-8") != raw.decode("latin-1")
    assert '"email":"héllo@example.com"' in raw.decode("utf-8")


def test_decode_jwt_accepts_missing_base64_padding() -> None:
    unpadded_payload = VALID_TOKEN.split(".")[1].rstrip("=")
    token = f"{VALID_TOKEN.split('.')[0]}.{unpadded_payload}.sig"
    assert decode_jwt(token) == decode_jwt(VALID_TOKEN)


def test_decode_jwt_rejects_base64url_alphabet_characters() -> None:
    """`atob` REJECTS `-`/`_` outright (`DOMException: Invalid character`), independently
    confirmed against a live Node process: `atob("-_-_")` throws, while the equivalent standard
    base64 `atob("+/+/")` succeeds. A base64url-encoded payload segment must therefore fail to
    decode, not be silently accepted as if it were standard base64.

    The fixture below is constructed so a NON-validating decoder would still succeed (a lone `-`
    inserted into an otherwise-valid, standard base64 encoding of `{"a":1}` -- Python's own
    `base64.b64decode` without `validate=True` silently DISCARDS the `-` rather than rejecting it,
    and would decode the remaining characters to the exact same valid `{"a":1}` JSON). A witness
    using an all-`-_` payload cannot discriminate a real validation defect from an unrelated empty-
    payload parse failure, since stripping every character leaves nothing to decode either way."""
    token = "header.eyJh-IjoxfQ==.sig"
    assert decode_jwt(token) is None


def test_decode_jwt_returns_none_for_a_token_that_is_not_three_segments() -> None:
    assert decode_jwt("only.two") is None
    assert decode_jwt("one") is None
    assert decode_jwt("a.b.c.d") is None


def test_decode_jwt_returns_none_for_invalid_json() -> None:
    import base64

    not_json = base64.b64encode(b"not json at all").decode("ascii")
    assert decode_jwt(f"header.{not_json}.sig") is None


def test_decode_jwt_null_literal_is_indistinguishable_from_malformed() -> None:
    """A payload segment that decodes to the JSON literal `null` returns `None` too -- a
    faithfully-reproduced Pi ambiguity (`JSON.parse("null") === null` in Pi as well), not an
    accident of this port."""
    import base64

    null_payload = base64.b64encode(b"null").decode("ascii")
    assert decode_jwt(f"header.{null_payload}.sig") is None


def test_get_account_id_extracts_from_a_valid_token() -> None:
    assert get_account_id(VALID_TOKEN) == VALID_TOKEN_ACCOUNT_ID


def test_get_account_id_returns_none_for_a_malformed_token() -> None:
    assert get_account_id("not-a-jwt") is None


def test_get_account_id_returns_none_when_decoded_payload_is_not_an_object() -> None:
    """Pi's own `decodeJwt` never checks the parsed shape -- if the payload segment decodes to a
    JSON array or scalar, `getAccountId`'s own optional chaining (`payload?.[JWT_CLAIM_PATH]`)
    gracefully returns `undefined`/`null` rather than crashing. This must not raise in Python
    either (a naive `payload[JWT_CLAIM_PATH]` on a non-dict would raise `TypeError`)."""
    import base64

    array_payload = base64.b64encode(b"[1,2,3]").decode("ascii")
    assert get_account_id(f"header.{array_payload}.sig") is None
    string_payload = base64.b64encode(b'"just a string"').decode("ascii")
    assert get_account_id(f"header.{string_payload}.sig") is None


def test_get_account_id_returns_none_when_claim_namespace_is_missing() -> None:
    import base64
    import json

    payload = base64.b64encode(json.dumps({"unrelated": "claim"}).encode()).decode("ascii")
    assert get_account_id(f"header.{payload}.sig") is None


def test_get_account_id_returns_none_when_claim_namespace_is_not_an_object() -> None:
    import base64
    import json

    payload = base64.b64encode(json.dumps({JWT_CLAIM_PATH: "not an object"}).encode()).decode(
        "ascii"
    )
    assert get_account_id(f"header.{payload}.sig") is None


def test_get_account_id_returns_none_when_account_id_is_not_a_non_empty_string() -> None:
    import base64
    import json

    for bad_value in (None, "", 123, ["a"]):
        payload = base64.b64encode(
            json.dumps({JWT_CLAIM_PATH: {"chatgpt_account_id": bad_value}}).encode()
        ).decode("ascii")
        assert get_account_id(f"header.{payload}.sig") is None


def test_credentials_from_token_builds_the_oauth_credential_with_account_id_under_extra() -> None:
    credential = credentials_from_token(
        access=VALID_TOKEN, refresh="refresh-token-value", expires=1234567890.0
    )
    assert credential == OAuthCredential(
        access=VALID_TOKEN,
        refresh="refresh-token-value",
        expires=1234567890.0,
        extra={"account_id": VALID_TOKEN_ACCOUNT_ID},
    )


def test_credentials_from_token_raises_when_account_id_cannot_be_extracted() -> None:
    with pytest.raises(ValueError, match="Failed to extract accountId from token"):
        credentials_from_token(access="not-a-jwt", refresh="r", expires=0.0)


def test_to_auth_projects_the_access_token_as_a_bearer_api_key() -> None:
    credential = OAuthCredential(
        access=VALID_TOKEN, refresh="r", expires=0.0, extra={"account_id": "x"}
    )
    assert to_auth(credential) == ModelAuth(api_key=VALID_TOKEN)
