"""Mounting the telemetry seam on the runtime."""

from __future__ import annotations

from ..runtime import Context, plugin
from .service import RecordingSink, TelemetryService


@plugin(name="telemetry", provides="telemetry")
async def telemetry_plugin(ctx: Context, config: None) -> None:
    """Provide telemetry with a recording sink mounted.

    Recording by default keeps the seam useful in tests and development
    without a deployment having to configure anything. Production sinks land
    in a later phase.
    """
    service = TelemetryService()
    service.recording = RecordingSink()
    service.add_sink(service.recording)
    ctx.provide("telemetry", service)
