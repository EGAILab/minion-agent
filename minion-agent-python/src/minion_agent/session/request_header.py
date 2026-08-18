"""Request headers as compositions of content-addressed components.

    Reconstruct request state from content-addressed components, never from
    repeated monolithic snapshots.  — design spec section 5

Component *names* are open: an application composes whatever sections it has.
The mechanism is not: every component is stored by hash, the header logs only
references, and reconstruction resolves them.

`assemble_system` is the canonical join. Dispatch and reconstruction must both
use it, or "the model saw what the log says" degrades to "the header we
recorded matches what we recorded".
"""

from __future__ import annotations

from .artifacts import ArtifactStore
from .events import EventKind, SessionEvent
from .log import SessionLog


def assemble_system(components: dict[str, str]) -> str:
    """Join components into the system prompt actually dispatched.

    Sorted by component name so the result depends on content alone, never on
    the order a caller happened to build the mapping.
    """
    return "\n\n".join(components[name] for name in sorted(components))


def record_header(
    log: SessionLog,
    store: ArtifactStore,
    components: dict[str, str],
    model: str,
) -> SessionEvent:
    """Store each component by hash and log the composition."""
    references = {name: store.put(text) for name, text in components.items()}
    return log.append(EventKind.REQUEST_HEADER, {"model": model, "components": references})


def reconstruct_header(event: SessionEvent, store: ArtifactStore) -> dict[str, str]:
    """Resolve a logged header's references back to their content."""
    references: dict[str, str] = event.data["components"]
    return {name: store.get(ref).decode("utf-8") for name, ref in references.items()}
