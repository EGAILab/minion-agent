# Rust Layer 04 XFORM Implementation Design

**Status:** APPROVED
**Certified contract:** `minion-agent-docs@0b02b9bcdef31ff9b23da7e7eeea48a13a732681`
**Implementation baseline:** `minion-agent@65569ba7079995e6e8dc717652ce17152fe08b78`

## Architecture

`minion-agent-rust/crates/minion-agent/src/llm/transform.rs` is the single generic XFORM semantic
implementation. It reuses the certified Layer-02 `Message` vocabulary and exposes a synchronous,
typed `transform_messages` seam over source messages, a narrow `TransformTarget`, and an optional
injected tool-call-ID normalizer. No provider framework or provider algorithm is introduced.

`TransformTarget` contains only a validated `ModelIdentity` and `supports_images`. Same-model
compatibility compares the complete provider + api + model_id triple.

The typed core performs the certified stages 2–4 in order: image downgrade, transcript-ordered
content/signature/ID transformation, then error/aborted filtering with orphan synthesis. The ID
normalizer receives the original source `AssistantMessage`, not a transformed copy; the library
owns mapping and later result rewrites.

## Legacy compatibility

Modern Rust messages cannot contain null content. A narrow library-owned compatibility module or
section accepts dynamic legacy values only at serialization/conformance boundaries, normalizes
null content into role-valid typed empty content, and then calls the typed `transform_messages`.
Raw JSON is not a normal runtime representation or a peer public semantic API.

## Conformance

`tests/xform_conformance.rs` parses the shared schema/scenarios, constructs typed inputs or invokes
the library-owned legacy boundary, supplies only a scripted normalization policy, calls the real
transform seam, and normalizes actual output. It contains no transformation semantics.

The existing Session conformance adapter activates
`request-reconstruction-after-target-transform.yaml` by calling real Session derivation followed
by real XFORM. It neither transforms messages nor persists transformed output.

## Test strategy

TDD proceeds through target/string behavior, image/dedup behavior, thinking/signatures, ID
orchestration, filtering/orphans, legacy null, canonical XFORM, and Session composition. Direct
tests pin rich-field preservation, source value immutability, original-assistant callback input,
and ordering interactions. Final counts must be 13/13 XFORM and 20/20 Session, with no deferments.

## Exclusions

No provider wire encoding, Responses replay, concrete provider ID normalization, HTTP integration,
Agent behavior, asynchronous machinery, or Layer-05 work is included. `AI-013` remains Phase-5
provider-wire work.
