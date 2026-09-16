"""Layer 11 (Real providers): the auth foundation and the Codex OAuth network integration.

Pass 1 established the provider-agnostic foundation that does not itself require a live network
call: credential vocabulary, the `CredentialStore` seam, PKCE, and RFC 8628 device-code polling
(`PROV-008`/`PROV-009`/`PROV-010`). Pass 2 builds real network-facing Codex support on top of
that foundation: the account-id projection (`PROV-011`), the login-interaction/auth-method
vocabulary (`PROV-014`), and the full Codex OAuth network integration itself (`PROV-012`) --
real HTTP calls to the OAuth token/device endpoints, the browser/local-callback-server login
flow, and the device-code endpoint integration (`openai_codex_oauth.py`) -- per `spec/auth.md`;
pinned Pi `packages/ai/src/auth/**`.

Deliberately excluded, Layer-11-owned but not yet implemented: any Codex-CLI-specific
credential-file loader, and generic Models-level provider/auth orchestration (`PROV-013`,
deferred).
"""
