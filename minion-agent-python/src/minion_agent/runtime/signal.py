"""A run-scoped cancellation flag (Layer 09, `L09-C001`/`L09-C002`/`L09-C003`).

Pinned Pi's own `AbortSignal`/`AbortController` (`packages/agent/src/agent.ts`) are cooperative
and poll-based: every consumer in `agent-loop.ts` either reads `signal?.aborted` directly or
passes the same `signal` object on to a hook/tool that may poll it itself. Pi's own code never
forcibly interrupts in-flight synchronous work because a signal became aborted.

`RunSignal` is the same shape, deliberately NOT built on `asyncio.Task.cancel()`/
`CancelledError`: task cancellation would forcibly interrupt cooperative code at whatever
`await` point it happens to be at, where Pi's own design lets that same code run to completion
unless it chooses to check the signal (see `assurance/layers/09-active-abort-contract-
checkpoint.md`). It also matches certified Rust Layer 06's own already-reserved, poll-based
`ToolExecutionSignal` trait (`fn is_cancelled(&self) -> bool`) rather than introducing a
push/interrupt mechanism Rust never committed to either.

Lives in `runtime/` (Layer 05, the lowest already-certified, most-depended-upon package) because
both Layer 02 (LLM) and Layer 06 (tools) need the same concrete type without depending on each
other or on Layer 07/08 (`agent`/`agent_loop`), which are themselves layered above both. This is
an additive new module -- no existing certified `runtime/` file is touched -- the same kind of
narrow, additive extension Layer 08 PASS 9 already used for `EventBus.serial`'s own
`yield_after_each` parameter.
"""

from __future__ import annotations

import asyncio


class RunSignal:
    """One run's cancellation flag. A NEW instance per run (matching pinned Pi's own
    `new AbortController()` per `runWithLifecycle` call, `agent.ts:491`) -- never reused
    across runs, and never mutated back to "not aborted" once set."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        """Poll-based, matching pinned Pi's own `signal.aborted` and certified Rust's own
        `ToolExecutionSignal::is_cancelled()` -- never a push notification."""
        return self._event.is_set()

    def abort(self) -> None:
        """Idempotent: aborting an already-aborted signal is a no-op, matching pinned Pi's own
        `AbortController.abort()`, which has no "already aborted" error state."""
        self._event.set()
