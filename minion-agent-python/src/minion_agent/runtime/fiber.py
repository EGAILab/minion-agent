"""Fibers: one loaded plugin instance, its lifecycle, config, and effects."""

from __future__ import annotations

from enum import StrEnum


class FiberState(StrEnum):
    """Lifecycle state of one plugin instance."""

    PENDING = "pending"
    """Mounted but not satisfied: at least one injected service is missing."""

    LOADING = "loading"
    """Dependencies satisfied; the plugin body is running."""

    ACTIVE = "active"
    """Loaded. Its services are visible and its effects are live."""

    FAILED = "failed"
    """The plugin body raised. Effects created before the failure are unwound."""

    UNLOADING = "unloading"
    """Effects are being disposed."""

    DISPOSED = "disposed"
    """Terminal. The fiber cannot be reused."""
