# Rust Layer 04 XFORM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement certified Layer-04 XFORM through one typed Rust library seam and make all 13 XFORM plus all 20 Session scenarios executable.

**Architecture:** Add `llm/transform.rs` for the typed stages 2–4 and a narrow library-owned dynamic compatibility boundary for stage 1. Reuse existing typed messages; canonical runners only parse, call, and normalize.

**Tech Stack:** Rust 2024, serde/serde_json/serde_yaml, existing minion-agent crate.

**Spec:** `docs/superpowers/specs/2026-08-24-rust-layer-04-xform-design.md`; authoritative contract `minion-agent-docs/spec/target-model-transformation.md` at `0b02b9b`.

## Global Constraints

- Preserve provider + api + model_id identity and typed image capability.
- Normalizer receives the original source AssistantMessage and runs cross-model only.
- Dynamic legacy input exists only at the compatibility/conformance boundary.
- No runner-owned XFORM semantics and no Layer-05/provider-wire work.
- Pipeline order is legacy normalization → image downgrade → content/signature/ID pass → filtering/orphan pass.

---

### Task 1: Typed target and string-preserving seam

**Files:**
- Create: `minion-agent-rust/crates/minion-agent/src/llm/transform.rs`
- Modify: `minion-agent-rust/crates/minion-agent/src/llm/mod.rs`
- Create: `minion-agent-rust/crates/minion-agent/tests/llm_transform.rs`

**Interfaces:**
- Produces: `TransformTarget`, `ToolCallIdNormalizer`, and `transform_messages`.

- [ ] Write failing compile/runtime tests for target construction, full identity comparison, source cloning, and string preservation for vision/non-vision/empty/whitespace/literal inputs.
- [ ] Run `cargo test -p minion-agent --test llm_transform` and verify failure because the API is absent.
- [ ] Implement the minimum public target/seam and unchanged-message behavior.
- [ ] Re-run the focused test and verify green.
- [ ] Commit the typed seam.

### Task 2: Image downgrade and placeholder mechanics

**Files:**
- Modify: `minion-agent-rust/crates/minion-agent/src/llm/transform.rs`
- Modify: `minion-agent-rust/crates/minion-agent/tests/llm_transform.rs`

**Interfaces:**
- Consumes: typed seam from Task 1.
- Produces: role-specific image transformation with exact placeholders.

- [ ] Add failing tests for vision preservation, user/tool-result downgrade, adjacent runs, three images, text breaks, literal suppression, and per-message scope.
- [ ] Run the focused tests and verify observable failures.
- [ ] Implement role-specific block helpers without generic text deduplication.
- [ ] Re-run focused tests and verify green.
- [ ] Commit image behavior.

### Task 3: Thinking and signature compatibility

**Files:**
- Modify: `minion-agent-rust/crates/minion-agent/src/llm/transform.rs`
- Modify: `minion-agent-rust/crates/minion-agent/tests/llm_transform.rs`

**Interfaces:**
- Produces: full same/cross-model thinking matrix and text/tool signature behavior.

- [ ] Add failing tests for redacted, signed empty/non-empty, unsigned blank/nonblank, each identity-component mismatch, text_signature stripping, thought_signature truthiness, and namespace preservation.
- [ ] Run focused tests and confirm semantic failures.
- [ ] Implement transcript content transformation while preserving unrelated message fields.
- [ ] Re-run focused tests and verify green.
- [ ] Commit thinking/signature behavior.

### Task 4: Injected ID orchestration

**Files:**
- Modify: `minion-agent-rust/crates/minion-agent/src/llm/transform.rs`
- Modify: `minion-agent-rust/crates/minion-agent/tests/llm_transform.rs`

**Interfaces:**
- Consumes: `ToolCallIdNormalizer`.
- Produces: original-ID map and matching ToolResult rewrites.

- [ ] Add failing tests for same-model non-invocation, cross-model invocation order, original source assistant input, matching/unrelated results, multiple calls, and rich ToolResult preservation.
- [ ] Run focused tests and verify failures.
- [ ] Implement the single forward-pass map orchestration.
- [ ] Re-run focused tests and verify green.
- [ ] Commit ID orchestration.

### Task 5: Filtering and orphan synthesis

**Files:**
- Modify: `minion-agent-rust/crates/minion-agent/src/llm/transform.rs`
- Modify: `minion-agent-rust/crates/minion-agent/tests/llm_transform.rs`

**Interfaces:**
- Produces: final filtering/orphan pass with wall-clock synthetic timestamps.

- [ ] Add failing tests for error/aborted-only exclusion, excluded-call non-synthesis, interruption/end flushing, multiple calls, resolved calls, normalized synthetic IDs, required tool_name, and source ordering.
- [ ] Run focused tests and verify failures.
- [ ] Implement one final forward pass with pending-call tracking and exact synthetic values.
- [ ] Re-run focused tests and verify green.
- [ ] Commit final typed-core behavior.

### Task 6: Library-owned legacy-null boundary

**Files:**
- Create: `minion-agent-rust/crates/minion-agent/src/llm/transform_compat.rs`
- Modify: `minion-agent-rust/crates/minion-agent/src/llm/mod.rs`
- Modify: `minion-agent-rust/crates/minion-agent/tests/llm_transform.rs`

**Interfaces:**
- Produces: narrow dynamic decoder/normalizer returning typed messages and delegating to `transform_messages`.

- [ ] Add a failing test feeding null content for each role through library compatibility code.
- [ ] Run focused tests and confirm the boundary is absent.
- [ ] Implement schema-shaped legacy decoding, null-to-role-valid-empty normalization, and delegation to the typed seam.
- [ ] Re-run focused tests and verify green without weakening modern message types.
- [ ] Commit compatibility boundary.

### Task 7: Canonical XFORM adapter

**Files:**
- Create: `minion-agent-rust/crates/minion-agent/tests/xform_conformance.rs`

**Interfaces:**
- Consumes: shared `conformance/agent/*.yaml`, compatibility boundary, target, and normalizer seam.
- Produces: 13 real-seam canonical executions.

- [ ] Write the dynamic discovery test and thin parser/normalizer; run it red against incomplete adapter behavior.
- [ ] Add only parsing/serialization needed to call the real library seam.
- [ ] Verify 13 discovered, 13 executed, 13 passed, zero deferred.
- [ ] Audit the runner for forbidden semantic helpers.
- [ ] Commit canonical XFORM evidence.

### Task 8: Activate Session→XFORM composition

**Files:**
- Modify: `minion-agent-rust/crates/minion-agent/tests/session_conformance.rs`

**Interfaces:**
- Consumes: real Session `derive_messages` and real `transform_messages`.
- Produces: 20 discovered/executed Session cases.

- [ ] Remove the deferment and add failing handling for `transform_target`/`expect_transformed_messages` through real seams.
- [ ] Run Session conformance and confirm the integration scenario fails before adapter wiring.
- [ ] Implement target construction, call real XFORM, and normalize only the projected observation.
- [ ] Verify 20 discovered, 20 executed, 20 passed, zero deferred and unchanged persisted source output.
- [ ] Commit Session integration evidence.

### Task 9: Assurance, verification, and integration

**Files:**
- Create centrally after implementation: `minion-agent-docs/assurance/layers/04-target-model-transformation-rust-implementation.md`

**Interfaces:**
- Produces: reviewed implementation evidence and PR/merge record.

- [ ] Run fmt, strict Clippy, full tests, rustdoc, and xtask conformance.
- [ ] Perform independent code review covering typed architecture, runner thinness, semantics, and Phase-5 exclusions.
- [ ] Record exact counts and findings in the assurance artifact without touching the unrelated Phase-5 edit.
- [ ] Open the dedicated Layer-04 PR with certified SHAs and evidence.
- [ ] Merge only after implementation review APPROVED, rerun post-merge gates, and record final SHAs.
