"""`Result[T, E]`: the never-raise boundary at the three execution capability seams
(Layer 12 WP-12.1, `EXEC-001`, spec/execution.md section 2).

Operational failures at `ctx.fs`/`ctx.shell`/`ctx.subprocess` are values, not exceptions.
Everywhere else in this codebase, ordinary exceptions remain the convention
(`runtime/errors.py`) -- this type is deliberately scoped to these three seams, not a
general-purpose replacement for exceptions across the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard


@dataclass(frozen=True, slots=True)
class Ok[T]:
    """A successful outcome carrying `value`."""

    value: T


@dataclass(frozen=True, slots=True)
class Err[E]:
    """A failed outcome carrying `error` -- never a raised exception."""

    error: E


type Result[T, E] = Ok[T] | Err[E]


def is_ok[T, E](result: Result[T, E]) -> TypeGuard[Ok[T]]:
    """Narrow `result` to `Ok[T]` for static type checkers."""
    return isinstance(result, Ok)


def is_err[T, E](result: Result[T, E]) -> TypeGuard[Err[E]]:
    """Narrow `result` to `Err[E]` for static type checkers."""
    return isinstance(result, Err)
