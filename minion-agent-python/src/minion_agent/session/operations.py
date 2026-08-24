"""Session operations, defined as log events rather than mutations.

Under an append-only log none of these can be a method that edits history.
Each appends an event, and derivation reads that event — which is why the
effect on derivation is stated here alongside the operation.
"""

from __future__ import annotations

from .events import EventKind, SessionEvent
from .log import SessionLog


class InvalidForkBoundaryError(ValueError):
    """A fork boundary beyond the ancestor's committed tip.

    A boundary that does not exist yet cannot fix what the fork sees: later
    ancestor writes that happen to land at or before that not-yet-committed
    seq would leak into the child once appended, breaking "later writes to
    either side stay private to that side" (delta finding D).
    """


def reset(log: SessionLog) -> SessionEvent:
    """Exclude everything so far from future derivation.

    Session identity is preserved: a session id is a durable external handle
    that applications bind conversations to, so "start over" is a derivation
    change rather than a new conversation. History remains fully readable for
    search and audit.
    """
    return log.append(EventKind.SESSION_RESET, {})


def compact(log: SessionLog, summary: str, keep: int) -> SessionEvent:
    """Replace the current surface with a summary plus the last `keep` entries.

    Records three things, because repeated and nested compaction otherwise has
    no deterministic projection: the superseded range, the replacement content,
    and the retained tail's provenance — the exact seqs carried forward.

    Provenance is what prevents double projection. The retained entries are
    named by sequence rather than copied, so derivation can never emit both an
    original and a copy of it.
    """
    from .derive import effective_surface

    surface = effective_surface(log)
    retained = surface[len(surface) - keep :] if keep else ()
    return log.append(
        EventKind.COMPACTION,
        {
            "summary": summary,
            "superseded_through": surface[-1].seq if surface else 0,
            "retained": [event.seq for event in retained],
        },
    )


def fork(source: SessionLog, session_id: str, at: int | None = None) -> SessionLog:
    """Branch a new session from `source` at `at` (default: its head).

    The fork *references* its ancestor rather than copying it, for the same
    reason the log never deletes: copying would duplicate model-visible content
    and create two places for one truth. Derivation walks the ancestry chain.

    The boundary is fixed here, so later writes to either side stay private to
    that side. A boundary beyond the source's current tip is rejected rather
    than silently accepted: it does not name a committed point yet, so it
    cannot fix what the child sees (delta finding D).
    """
    tip = len(source)
    boundary = tip if at is None else at
    if boundary > tip:
        raise InvalidForkBoundaryError(
            f"fork boundary {boundary} is beyond committed tip {tip}"
        )
    child = SessionLog(
        session_id,
        ancestor=source,
        boundary=boundary,
        surface_kinds=source.surface_kinds,
    )
    child.append(EventKind.SESSION_FORKED, {"source": source.session_id, "boundary": boundary})
    return child
