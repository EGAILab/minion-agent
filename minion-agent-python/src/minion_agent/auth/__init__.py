"""Layer 11 (Real providers), Pass 1: the auth foundation.

Provider-agnostic credential vocabulary, the `CredentialStore` seam, PKCE, and RFC 8628
device-code polling -- everything a real provider's own login/refresh flow needs that does not
itself require a live network call (`spec/auth.md`; pinned Pi `packages/ai/src/auth/**`).

Deliberately excluded from this pass (Layer-11-owned, not yet implemented): real HTTP calls to
any OAuth endpoint, the Codex-specific browser/local-callback-server login flow, the Codex
device-code endpoint integration, and any Codex-CLI-specific credential-file loader.
"""
