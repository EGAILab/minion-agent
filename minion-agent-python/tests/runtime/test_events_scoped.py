"""Event admission extends up the scope chain; untagged listeners hear all."""

from collections.abc import Awaitable, Callable
from typing import Any

from minion_agent.runtime.events import DispatchMode, EventBus
from minion_agent.runtime.scope import ScopeKey

DEFINITION = ScopeKey("definition")
INSTANCE = ScopeKey("instance", parent=DEFINITION)
TURN = ScopeKey("turn", parent=INSTANCE)
OTHER = ScopeKey("other", parent=DEFINITION)


def _bus() -> EventBus:
    bus = EventBus()
    bus.declare("test/emit", DispatchMode.EMIT)
    return bus


def test_untagged_listener_hears_every_dispatch() -> None:
    bus = _bus()
    seen: list[str] = []
    bus.on("test/emit", lambda: seen.append("untagged"))

    bus.emit("test/emit", scope=TURN)
    bus.emit("test/emit")

    assert seen == ["untagged", "untagged"]


def test_ancestor_listener_hears_a_descendant_dispatch() -> None:
    bus = _bus()
    seen: list[str] = []
    bus.on("test/emit", lambda: seen.append("definition"), scope=DEFINITION)

    bus.emit("test/emit", scope=TURN)

    assert seen == ["definition"]


def test_descendant_listener_does_not_hear_an_ancestor_dispatch() -> None:
    bus = _bus()
    seen: list[str] = []
    bus.on("test/emit", lambda: seen.append("turn"), scope=TURN)

    bus.emit("test/emit", scope=DEFINITION)

    assert seen == []


def test_siblings_do_not_hear_each_other() -> None:
    bus = _bus()
    seen: list[str] = []
    bus.on("test/emit", lambda: seen.append("other"), scope=OTHER)

    bus.emit("test/emit", scope=INSTANCE)

    assert seen == []


def test_unscoped_dispatch_admits_only_untagged_listeners() -> None:
    bus = _bus()
    seen: list[str] = []
    bus.on("test/emit", lambda: seen.append("untagged"))
    bus.on("test/emit", lambda: seen.append("tagged"), scope=INSTANCE)

    bus.emit("test/emit")

    assert seen == ["untagged"]


def test_admission_order_follows_registration_order() -> None:
    bus = _bus()
    seen: list[str] = []
    bus.on("test/emit", lambda: seen.append("first"), scope=DEFINITION)
    bus.on("test/emit", lambda: seen.append("second"))
    bus.on("test/emit", lambda: seen.append("third"), scope=TURN)

    bus.emit("test/emit", scope=TURN)

    assert seen == ["first", "second", "third"]


async def test_waterfall_honours_admission() -> None:
    bus = EventBus()
    bus.declare("test/waterfall", DispatchMode.WATERFALL)

    async def excluded(next_: Callable[..., Awaitable[Any]]) -> Any:
        return "excluded"

    async def included(next_: Callable[..., Awaitable[Any]]) -> Any:
        return "included"

    bus.on("test/waterfall", excluded, scope=OTHER)
    bus.on("test/waterfall", included, scope=DEFINITION)

    assert await bus.waterfall("test/waterfall", scope=TURN) == "included"


async def test_serial_and_parallel_honour_admission() -> None:
    bus = EventBus()
    bus.declare("test/serial", DispatchMode.SERIAL)
    bus.declare("test/parallel", DispatchMode.PARALLEL)
    seen: list[str] = []

    bus.on("test/serial", lambda: seen.append("serial-other") or "other", scope=OTHER)
    bus.on("test/serial", lambda: seen.append("serial-def") or "def", scope=DEFINITION)
    bus.on("test/parallel", lambda: seen.append("parallel-other"), scope=OTHER)
    bus.on("test/parallel", lambda: seen.append("parallel-def"), scope=DEFINITION)

    assert await bus.serial("test/serial", scope=TURN) == "def"
    await bus.parallel("test/parallel", scope=TURN)

    assert seen == ["serial-def", "parallel-def"]
