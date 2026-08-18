"""Exception hierarchy for the plugin runtime.

These are programming and composition errors. Operational failures at the
capability seams are values, not exceptions — see the design spec, section 7.
"""


class RuntimeError_(Exception):
    """Base for every runtime error.

    Named with a trailing underscore to avoid shadowing the builtin while
    keeping the public alias `errors.RuntimeError_` unambiguous at call sites.
    """


class ServiceConflictError(RuntimeError_):
    """A second plugin tried to provide a service that is already provided."""


class InactiveFiberError(RuntimeError_):
    """An effect was created on a fiber or scope that is no longer active."""


class ServiceNotFoundError(RuntimeError_):
    """A service was accessed that no active provider supplies."""


class EventModeError(RuntimeError_):
    """An event was dispatched in a mode other than the one it declared."""


class WaterfallError(RuntimeError_):
    """A waterfall listener misused its `next` continuation."""
