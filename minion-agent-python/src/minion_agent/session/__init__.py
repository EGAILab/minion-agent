"""The append-only session log: the system's semantic truth."""

from .artifacts import ArtifactStore, MissingArtifactError
from .derive import (
    decode_message,
    derive_messages,
    effective_surface,
    encode_message,
    messages_from,
)
from .events import (
    CORE_SURFACE_KINDS,
    SURFACE_KINDS,
    EventKind,
    EventName,
    InvalidEventNameError,
    SessionEvent,
    is_surface,
    validate_event_name,
)
from .log import NotJsonSafeError, SessionLog
from .operations import compact, fork, reset
from .request_header import (
    assemble_system,
    reconstruct_header,
    reconstruct_tools,
    record_header,
)
from .service import SessionService

__all__ = [
    "CORE_SURFACE_KINDS",
    "SURFACE_KINDS",
    "ArtifactStore",
    "EventKind",
    "EventName",
    "InvalidEventNameError",
    "MissingArtifactError",
    "NotJsonSafeError",
    "SessionEvent",
    "SessionLog",
    "SessionService",
    "assemble_system",
    "compact",
    "decode_message",
    "derive_messages",
    "effective_surface",
    "encode_message",
    "fork",
    "is_surface",
    "messages_from",
    "reconstruct_header",
    "reconstruct_tools",
    "record_header",
    "reset",
    "validate_event_name",
]
