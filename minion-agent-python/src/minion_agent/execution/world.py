"""Execution-world identity and compatibility (`EXEC-006`, spec/execution.md section 7).
`MINION_EXTENSION` -- no Pi source at all.

A consumer needing multiple capabilities to address the SAME resource validates their
execution-world identities at its own activation; mounting incompatible capabilities that no
consumer ever asks to be validated together remains legal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .result import Err, Ok, Result


@dataclass(frozen=True, slots=True)
class ExecutionWorldIdentity:
    """Opaque, provider-declared value. Comparable only for equality -- `compatible()` below is
    the ONE relation this contract defines, and it is exactly equality (`L12-R013`)."""

    value: str


def compatible(a: ExecutionWorldIdentity, b: ExecutionWorldIdentity) -> bool:
    """`L12-R013`: EQUALITY-ONLY. No broader/declarable relation -- that earlier design left
    `validate()` observably order-dependent (an earlier revision permitted a provider to declare
    itself compatible with specific other identities without a symmetry rule); equality is
    inherently symmetric and order-independent by construction, so this needs no separate
    declaration model to specify or get wrong."""
    return a == b


@dataclass(frozen=True, slots=True)
class IncompatiblePair:
    """One entry in `ExecutionWorldError.incompatible_pairs` (`L12-R019`)."""

    left: str
    right: str


@dataclass(frozen=True, slots=True)
class ExecutionWorldError:
    """`L12-R019`'s own concrete payload: an ordered `incompatible_pairs` list, enumerated by
    input index `i < j` (never reversed, never re-sorted) -- the sole normative field. Any
    human-readable message a caller attaches separately is non-normative."""

    incompatible_pairs: tuple[IncompatiblePair, ...]


def validate(
    providers: Sequence[tuple[str, ExecutionWorldIdentity]],
) -> Result[None, ExecutionWorldError]:
    """Called by a CONSUMER (never the runtime) over the specific providers it needs to address
    the same resource through. Pairwise-checks every `i < j` combination in INPUT order; returns
    `Err` naming every incompatible pair if any exist, `Ok(None)` otherwise.

    Caller-supplied labels (the `name` half of each tuple) MUST be unique within one call --
    a duplicate label is an explicit caller-precondition violation (undefined behavior for this
    primitive), not a case this function special-cases or validates against.
    """
    pairs: list[IncompatiblePair] = []
    for i in range(len(providers)):
        name_i, identity_i = providers[i]
        for j in range(i + 1, len(providers)):
            name_j, identity_j = providers[j]
            if not compatible(identity_i, identity_j):
                pairs.append(IncompatiblePair(left=name_i, right=name_j))
    if pairs:
        return Err(ExecutionWorldError(incompatible_pairs=tuple(pairs)))
    return Ok(None)
