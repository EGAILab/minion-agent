"""Permanent static-type evidence for `L11-R006`/`L11-R009` (owner-decided Pi-parity credential
mutation, `agent-workflow.md` §11.7/§11.8).

Not a pytest test: mypy checking this module IS the test. The targeted-closure review's own
finding: an ad-hoc, un-executed mypy probe is not a permanent gate -- reverting `env`/`extra` back
to a read-only `Mapping`, or re-adding `frozen=True` to either credential dataclass, would leave
every OTHER configured gate (pytest, the default `mypy` gate scoped to `src/minion_agent`) green,
silently reopening `L11-R006`/`L11-R009` with no regression signal at all. This module's only job
is to fail `mypy` if any of these owner-decided, Pi-parity mutations ever stop type-checking.

Run explicitly (not part of the default `mypy` gate, which is scoped to `src/minion_agent` only):

    mypy src/minion_agent tests/typing/valid_auth_credential_mutation.py

Never imported or executed by pytest.
"""

from __future__ import annotations

from minion_agent.auth.credential import ApiKeyCredential, OAuthCredential

# ApiKeyCredential.env: a brand-new top-level key assignment through the returned, statically
# typed mapping itself must type-check with no `# type: ignore` (`L11-R006`'s own minimal witness).
# `env`'s own declared type is `dict[str, str] | None` (optional at the FIELD level, matching Pi's
# own `env?: ProviderEnv`); the `is not None` narrowing below is the ordinary way a caller proves
# it holds an actual mapping before indexing it -- the point under test is that, once narrowed,
# `dict[str, str]` itself supports plain item ASSIGNMENT, unlike the rejected read-only `Mapping`.
_api_key_credential: ApiKeyCredential = ApiKeyCredential(env={})
assert _api_key_credential.env is not None
_api_key_credential.env["NEW"] = "v"

# OAuthCredential.extra: the same brand-new top-level key assignment, on the OPEN recursive-JSON
# field this time -- both a flat string value and a nested JSON value must type-check.
_oauth_credential: OAuthCredential = OAuthCredential(access="a", refresh="r", expires=1.0)
_oauth_credential.extra["accountId"] = "acc_1"
_oauth_credential.extra["nested"] = {"value": "A"}

# ApiKeyCredential scalar-field reassignment (`L11-R009`): the dataclass is not frozen, matching
# Pi's own plain, mutable credential object -- `credential.key = "B"` must type-check directly.
_api_key_credential.key = "B"

# OAuthCredential scalar-field reassignment (`L11-R009`): all three required fields.
_oauth_credential.access = "a2"
_oauth_credential.refresh = "r2"
_oauth_credential.expires = 2.0
