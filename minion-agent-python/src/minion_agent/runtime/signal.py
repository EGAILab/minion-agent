"""A run-scoped cancellation flag (Layer 09, `L09-C001`/`L09-C002`/`L09-C003`, `L09-R004`).

Pinned Pi's own `AbortController`/`AbortSignal` (`packages/agent/src/agent.ts`) split authority:
`AbortController` is PRIVATE to `Agent.runWithLifecycle`; only `Agent.abort()` may ever call its
own `.abort()`. Every consumer -- lifecycle listeners, tool hooks, `execute()`, the provider
stream, `prepareNextTurn`/`shouldStopAfterTurn` -- receives `Agent.signal`'s own value instead: an
`AbortSignal`, observational and read-only, with no `.abort()` of its own at all. Handing a
consumer the controller itself, or letting `Agent.signal` be reassigned mid-run, are both things
Pi's own type system structurally forbids -- an independent Rust contract review caught an earlier
Python revision doing both (`L09-R004`): every consumer received the SAME mutable object tools and
adapters could call `.abort()` on, and `AgentInstance.signal` was a plain public attribute any
listener could overwrite mid-run, redirecting later requests to a caller-supplied replacement.

`RunAbortController`/`RunSignal` reproduce the same split. Both are cooperative and poll-based,
deliberately NOT built on `asyncio.Task.cancel()`/`CancelledError`: task cancellation would
forcibly interrupt cooperative code at whatever `await` point it happens to be at, where Pi's own
design lets that same code run to completion unless it chooses to check the signal (see
`assurance/layers/09-active-abort-contract-checkpoint.md`). This also matches certified Rust
Layer 06's own already-reserved, poll-based `ToolExecutionSignal` trait
(`fn is_cancelled(&self) -> bool`) rather than introducing a push/interrupt mechanism Rust never
committed to either.

Lives in `runtime/` (Layer 05, the lowest already-certified, most-depended-upon package) because
both Layer 02 (LLM) and Layer 06 (tools) need the same concrete `RunSignal` type without depending
on each other or on Layer 07/08 (`agent`/`agent_loop`), which are themselves layered above both.
This is an additive new module -- no existing certified `runtime/` file is touched -- the same kind
of narrow, additive extension Layer 08 PASS 9 already used for `EventBus.serial`'s own
`yield_after_each` parameter.
"""

from __future__ import annotations

import asyncio


class RunSignal:
    """The READ-ONLY view every consumer actually receives -- pinned Pi's own `AbortSignal`.
    Observational only: `aborted` polls the underlying flag; there is no mutator here at all, by
    construction (`L09-R004`) -- a tool, hook, or adapter holding a `RunSignal` cannot itself
    trigger cancellation, matching Pi's own type-level guarantee exactly. Never constructed
    directly by application code; obtained only via `RunAbortController.signal` or
    `AgentInstance.signal`."""

    __slots__ = ("_controller",)

    def __init__(self, controller: RunAbortController) -> None:
        self._controller = controller

    @property
    def aborted(self) -> bool:
        """Poll-based, matching pinned Pi's own `signal.aborted` and certified Rust's own
        `ToolExecutionSignal::is_cancelled()` -- never a push notification."""
        return self._controller._aborted


class RunAbortController:
    """The PRIVATE per-run mutator -- pinned Pi's own `AbortController`. Held only by
    `AgentInstance`/`AgentLoop` (Layer 07/08); never handed to a tool, hook, adapter, or
    lifecycle listener directly -- those all receive `.signal` (a `RunSignal`) instead. A NEW
    instance per run (matching pinned Pi's own `new AbortController()` per `runWithLifecycle`
    call, `agent.ts:491`) -- never reused across runs, and never mutated back to "not aborted"
    once set."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.signal = RunSignal(self)
        """The one read-only view for this controller's entire lifetime -- created once, not
        per-access, so every consumer observing `instance.signal` across a run's own duration
        sees the SAME object (`L09-R004`'s own "stable per-run identity" requirement)."""

    @property
    def _aborted(self) -> bool:
        return self._event.is_set()

    def abort(self) -> None:
        """Idempotent: aborting an already-aborted signal is a no-op, matching pinned Pi's own
        `AbortController.abort()`, which has no "already aborted" error state."""
        self._event.set()
