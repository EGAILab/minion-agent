# Minion Agent

A Python agent runtime that preserves the functional scope and semantics of Pi
`agent-core`, rebuilt on a Cordis-inspired plugin architecture: plugin
lifecycle, service dependency, typed events, and reversible effects.

**Status:** design complete, implementation not yet started.

## Design

The design specification lives in the companion documentation repository under
`minion-agent-docs/design/`.

## Architecture in brief

- **Plugin runtime** (`minion_agent.runtime`) — Cordis-semantic context,
  fibers, service resolution, four event dispatch modes, and reversible
  effects with reactive dependency.
- **Log as truth** — an append-only session log is the sole source of
  model-visible context; requests are derived from it and independently
  verified by runtime invariants.
- **Capability seams** — `ctx.llm`, `ctx.fs`, `ctx.shell`, and `ctx.tools` are
  swappable services, each declared and withdrawn as an effect.

## Validation

Development is contract-driven. Behavior is validated in three tiers:

| Tier | Location | Purpose |
|---|---|---|
| Conformance | `conformance/` | Language-neutral executable compatibility cases, so a future Rust implementation runs the identical files |
| Python tests | `tests/` | Implementation-specific behavior and property tests |
| Model-backed evals | `evals/` | End-to-end agent quality against real providers |

Every externally meaningful runtime behavior change must be represented in
`conformance/`.

## License

TBD
