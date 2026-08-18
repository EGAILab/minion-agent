# Conformance runner contract

These scenarios are the executable half of the Minion Agent specification.
They are language-neutral data: any implementation ships a thin runner that
feeds them to its own code and diffs the observed trace.

## Families

| Family | Asserts |
|---|---|
| `runtime/` | A generic lifecycle and effect trace — mount, unmount, and scope operations in; ordered fiber-state transitions, effect disposals, and scope disposals out. |
| `agent/` | The session-log projection and the derived Pi event stream. |
| `session/` | Derivation after log operations, driven directly with no model in play. |

## Rules for every runner

1. **Assert on observable output only.** The trace, the log projection, and the
   derived messages are the surface. A runner that inspects implementation
   internals is not conformant, and its scenarios will not port.
2. **Order is significant** unless a scenario marks an entry otherwise.
   Reverse-order effect disposal and source-order tool results are behavior,
   not implementation detail.
3. **No wall-clock time.** Scenarios settle on state transitions and logical
   ticks. A runner that sleeps is wrong even when it passes.
4. **Unknown keys are errors.** Every schema sets `additionalProperties: false`
   so a scenario using a feature a runner has not implemented fails loudly
   rather than silently skipping.
5. **Trace comparison is exact.** Extra observed entries fail the case, as do
   missing ones. A scenario that needs to permit variation says so explicitly.
6. **Reconstruction, never storage.** Where a scenario concerns request state,
   it asserts that the reconstructed model input matches what was dispatched.
   Hashing scheme and artifact layout are implementation choices.

## Normativity

For behavior a scenario covers, the executable result is the compatibility
oracle. The prose spec (`minion-agent-docs/spec/`) defines the general
semantic rule and the behavior that cannot be exhaustively enumerated.
Behavior is not "unspecified" merely because no scenario encodes it yet.
