"""The append-only session log: the system's semantic truth."""

from .artifacts import ArtifactStore, MissingArtifactError
from .derive import (
    decode_message,
    derive_messages,
    effective_surface,
    encode_message,
    messages_from,
)
from .events import SURFACE_KINDS, EventKind, SessionEvent, is_surface
from .log import NotJsonSafeError, SessionLog
from .operations import compact, fork, reset
from .request_header import assemble_system, reconstruct_header, record_header
from .service import SessionService

__all__ = [
    "SURFACE_KINDS",
    "ArtifactStore",
    "EventKind",
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
    "record_header",
    "reset",
]
