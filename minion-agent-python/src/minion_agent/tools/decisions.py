"""Closed decision/override types for `tools/pre-execute` and `tools/post-execute`.

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

from ..llm import ToolResultContentBlock, Usage


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


@dataclass(frozen=True, slots=True)
class AfterToolCallOverride:
    """Pinned Pi's `AfterToolCallResult` (`packages/agent/src/types.ts`) -- the ONLY fields an
    after-hook may override, exactly five, field-by-field: an omitted (`None`) field keeps the
    current value; a supplied field replaces it wholesale (no deep merge). `tool_call_id`,
    `tool_name`, and `added_tool_names` are execution identity/metadata, not part of Pi's
    override surface at all -- this type structurally cannot carry them, so a hook cannot
    replace them by construction, not merely by convention (`L06-R003`).

    An earlier, uncertified revision of this pipeline let an after-hook listener return/replace
    the entire `ToolResult`, which could observably rewrite identity fields Pi never allows a
    hook to touch -- a genuine `CONTRACT_ASSURANCE_DEFECT`. This type is the repair.
    """

    content: tuple[ToolResultContentBlock, ...] | None = None
    details: dict[str, Any] | None = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None
