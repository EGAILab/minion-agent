# Minion Agent — Python

The **Python implementation** of Minion Agent: an agent runtime that preserves
the functional scope and semantics of Pi `agent-core`, rebuilt on a
Cordis-inspired plugin architecture — plugin lifecycle, service dependency,
typed events, and reversible effects.

The repository name says which language; the package says which project. The
importable package is `minion_agent`, and a second-language implementation
would expose the same name against the same specification.

**Status:** plugin runtime, LLM vocabulary, session log, and telemetry are
implemented and passing conformance. The agent loop and tool subsystem follow.

## Design and specification

The design specification lives in the companion documentation repository under
`minion-agent-docs/`. It is deliberately language-neutral: `conformance/` in
this repository is pure data, so another implementation can vendor or
submodule it without depending on Python.

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
