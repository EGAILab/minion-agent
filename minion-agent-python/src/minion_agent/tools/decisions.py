"""Closed decision union for `tools/pre-execute`.

Identical in shape to `agent/pre-step` (design spec section 6): a listener with
no opinion delegates, a listener that owns the decision returns one without
delegating, and the first decision wins as a consequence of the waterfall's
short-circuit rule rather than a second mechanism.

Approval and sandboxing are plugins on this event, not built-ins -- which is
where pi places them too (section 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Block:
    """Do not run the call. The model receives `reason` as an error result.

    A blocked call still produces a result: the model must see one for every
    call it made. `terminate` additionally asks the loop to end the turn, and
    participates in the batch fold like any other terminating result.
    """

    reason: str
    terminate: bool = False


@dataclass(frozen=True, slots=True)
class Proceed:
    """Run the call with these arguments.

    Carrying the arguments rather than merely approving is what lets a
    sandboxing listener narrow them -- pinning a path, capping a limit --
    without the tool needing to know a policy exists.
    """

    arguments: dict[str, Any]


type PreExecuteDecision = Block | Proceed
