"""An instance is one live execution identity."""

import pytest

from minion_agent.agent.envelope import InboxTarget
from minion_agent.agent.identity import AgentDefinition, AgentStatus, ThinkingLevel
from minion_agent.agent.instance import AgentActiveError, AgentInstance, instance_scope_key
from minion_agent.llm import ModelId, TextBlock, UserMessage
from minion_agent.runtime import Context, FiberState, scope_of
from minion_agent.session import SessionLog
from minion_agent.session.derive import encode_message
from minion_agent.session.events import EventKind
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.registry import ToolRegistry


class _FakeOwner:
    """A minimal, always-active service owner (mirrors `test_context_access.py`)."""

    name = "owner"
    state = FiberState.ACTIVE


def _with_tools(ctx: Context, registry: ToolRegistry | None = None) -> ToolRegistry:
    """Provide `registry` (or a fresh one) as the `ctx.tools` service and return it."""
    registry = registry or ToolRegistry()
    ctx.registry.provide("tools", registry, _FakeOwner())
    return registry


def _definition() -> AgentDefinition:
    return AgentDefinition(name="ada", model=ModelId("mock", "mock-1"))


def _instance(ctx: Context | None = None) -> AgentInstance:
    context = ctx or Context()
    return AgentInstance(
        instance_id="room-a",
        definition=_definition(),
        log=SessionLog("room-a"),
        ctx=context,
    )


def test_an_instance_starts_idle() -> None:
    assert _instance().status is AgentStatus.IDLE


def test_an_instance_owns_its_log_and_inbox() -> None:
    first, second = _instance(), _instance()

    assert first.inbox is not second.inbox
    assert first.log is not second.log


def test_instances_of_one_definition_share_its_configuration() -> None:
    first, second = _instance(), _instance()

    assert first.definition.name == second.definition.name


def test_the_instance_scope_is_a_child_of_the_definition_scope() -> None:
    """So definition-level registrations are visible to every instance, and
    instance-level ones are not visible to siblings."""
    key = instance_scope_key(_definition(), "room-a")

    assert key.name == "agent-instance:room-a"
    assert key.parent is not None
    assert key.parent.name == "agent-definition:ada"


def test_the_instance_context_carries_its_scope() -> None:
    instance = _instance()

    assert scope_of(instance.scope.ctx) == instance.scope.key


def test_status_changes_fire_the_hook() -> None:
    instance = _instance()
    seen: list[AgentStatus] = []
    instance.on_status_change = seen.append

    instance.set_status(AgentStatus.RUNNING)
    instance.set_status(AgentStatus.IDLE)

    assert seen == [AgentStatus.RUNNING, AgentStatus.IDLE]


def test_setting_the_same_status_twice_reports_once() -> None:
    """A transition signal must signal transitions, not assignments."""
    instance = _instance()
    seen: list[AgentStatus] = []
    instance.on_status_change = seen.append

    instance.set_status(AgentStatus.RUNNING)
    instance.set_status(AgentStatus.RUNNING)

    assert seen == [AgentStatus.RUNNING]


# -- AG-014: mutable per-instance current configuration ----------------------


def test_current_config_defaults_from_the_definition() -> None:
    """Pinned Pi's `createMutableAgentState`: `initialState?.systemPrompt ?? ""`,
    `initialState?.model ?? DEFAULT_MODEL` -- the instance's current value starts
    at the definition's own default."""
    instance = _instance()

    assert instance.system_prompt == ""
    assert instance.model == _definition().model
    assert instance.thinking_level is ThinkingLevel.OFF


def test_mutating_one_instances_config_does_not_affect_a_sibling_or_the_definition() -> None:
    """Pi's `agent.state.systemPrompt = "..."` mutates only that Agent's own
    state; the shared definition/default and any sibling instance are
    unaffected (`AG-014`, `L07-R001`)."""
    definition = _definition()
    context = Context()
    a = AgentInstance(
        instance_id="a", definition=definition, log=SessionLog("a"), ctx=context
    )
    b = AgentInstance(
        instance_id="b", definition=definition, log=SessionLog("b"), ctx=context
    )

    a.system_prompt = "custom"
    a.model = ModelId("mock", "mock-2")
    a.thinking_level = ThinkingLevel.HIGH

    assert a.system_prompt == "custom"
    assert a.model == ModelId("mock", "mock-2")
    assert a.thinking_level is ThinkingLevel.HIGH
    assert b.system_prompt == ""
    assert b.model == definition.model
    assert b.thinking_level is ThinkingLevel.OFF
    assert definition.system == ""
    assert definition.model == ModelId("mock", "mock-1")


# -- AG-015: runtime-state vocabulary (initial values only) ------------------


def test_runtime_fields_start_at_pis_own_initial_values() -> None:
    """Pinned Pi: `streamingMessage: undefined`, `pendingToolCalls: new Set()`,
    `errorMessage: undefined` (`AG-015`). Transition timing is Layer 08's;
    this only pins the initial vocabulary/value."""
    instance = _instance()

    assert instance.streaming_message is None
    assert instance.pending_tool_calls == frozenset()
    assert instance.error_message is None


def test_messages_projects_the_session_log() -> None:
    """Pi's `state.messages` read; Minion's authority is the SessionLog
    (Layer 03, certified) -- `AgentInstance.messages` is a fresh projection,
    never a live mutable reference (`AG-015`)."""
    instance = _instance()

    assert instance.messages == ()


def test_messages_reflects_events_actually_appended_to_the_log() -> None:
    """The positive counterpart to the empty-log case above: a real appended
    message actually surfaces through the projection, proving it reads live
    log content and is not just always empty."""
    instance = _instance()
    message = UserMessage(content=(TextBlock(text="hi"),), timestamp=1)
    instance.log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})

    assert instance.messages == (message,)


# -- AG-017: Agent-level tools projection -------------------------------------


def test_tools_projects_the_tool_registry_visible_from_this_instances_scope() -> None:
    """Pi's `state.tools` read; Minion's authority is the certified Layer-05
    `ToolRegistry` -- `AgentInstance.tools` is a fresh projection over
    `visible_from(scope)`, mirroring `messages` (`AG-017`, `L07-R005`)."""
    context = Context()
    registry = _with_tools(context)
    definition = ToolDefinition(
        name="echo",
        description="d",
        parameters={"type": "object", "properties": {}},
        execute=lambda args: "ok",
        label="echo",
    )
    instance = _instance(context)
    registry.register(definition, scope=instance.scope.key)

    assert instance.tools == (definition,)


def test_tools_reflects_registrations_made_after_construction() -> None:
    """Not a snapshot taken at `__init__` time -- a fresh read every access,
    exactly like `messages`."""
    context = Context()
    registry = _with_tools(context)
    instance = _instance(context)
    assert instance.tools == ()

    definition = ToolDefinition(
        name="echo",
        description="d",
        parameters={"type": "object", "properties": {}},
        execute=lambda args: "ok",
        label="echo",
    )
    registry.register(definition, scope=instance.scope.key)

    assert instance.tools == (definition,)


# -- AG-016: in-place reset ---------------------------------------------------


def test_reset_clears_runtime_state_and_both_queues() -> None:
    instance = _instance()
    instance.error_message = "boom"
    instance.streaming_message = UserMessage(content=(TextBlock(text="hi"),), timestamp=1)
    instance.pending_tool_calls = frozenset({"t1"})
    instance.inbox.followup(UserMessage(content=(TextBlock(text="turn"),), timestamp=1))
    instance.inbox.steer(UserMessage(content=(TextBlock(text="step"),), timestamp=1))

    instance.reset()

    assert instance.error_message is None
    assert instance.streaming_message is None
    assert instance.pending_tool_calls == frozenset()
    assert not instance.inbox.has_pending()


def test_reset_clears_messages_via_the_session_reset_marker() -> None:
    """Pinned Pi's `reset()` also clears `messages` (`this._state.messages = []`).
    Minion reproduces this through the already-certified Layer-03
    `session.reset(log)` marker, not by adding a new primitive to `SessionLog`
    (`L07-R003`, second independent Rust review): appending a `session/reset`
    event makes `derive_messages` stop projecting everything at or before it."""
    instance = _instance()
    message = UserMessage(content=(TextBlock(text="hi"),), timestamp=1)
    instance.log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})
    assert instance.messages == (message,)

    instance.reset()

    assert instance.messages == ()


def test_reset_preserves_full_history_for_audit_after_clearing_the_projection() -> None:
    """`session.reset()` appends a marker; it never truncates the log itself --
    history stays readable, only the model-facing projection changes."""
    instance = _instance()
    message = UserMessage(content=(TextBlock(text="hi"),), timestamp=1)
    instance.log.append(EventKind.USER_MESSAGE, {"message": encode_message(message)})

    instance.reset()

    assert len(instance.log) == 2


def test_reset_does_not_clear_a_pending_wake_signal() -> None:
    """Normative, not incidental (`L07-R003`, second independent Rust review):
    wake and queued content are orthogonal concerns, and pinned Pi has no wake
    concept to constrain this decision either way -- a wake that arrived before
    an idle instance was reset still describes a real, unconsumed signal."""
    instance = _instance()
    instance.inbox.followup(UserMessage(content=(TextBlock(text="hi"),), timestamp=1))
    assert instance.inbox.wake_requested

    instance.reset()

    assert instance.inbox.wake_requested


def test_reset_retains_identity_configuration_and_tool_relationship() -> None:
    instance = _instance()
    instance.system_prompt = "custom"
    instance.model = ModelId("mock", "mock-2")
    instance.thinking_level = ThinkingLevel.HIGH
    seen: list[AgentStatus] = []

    def on_change(status: AgentStatus) -> None:
        seen.append(status)

    instance.on_status_change = on_change

    instance.reset()

    assert instance.system_prompt == "custom"
    assert instance.model == ModelId("mock", "mock-2")
    assert instance.thinking_level is ThinkingLevel.HIGH
    assert instance.on_status_change is on_change
    assert instance.definition.name == "ada"
    assert instance.status is AgentStatus.IDLE


def test_reset_is_in_place_not_a_fresh_instance() -> None:
    """The independent Rust review's core objection: fresh-instance
    replacement is observably different from Pi's in-place `reset()` --
    identity must survive (`L07-R003`)."""
    instance = _instance()
    inbox_before = instance.inbox
    log_before = instance.log

    instance.reset()

    assert instance.inbox is inbox_before
    assert instance.log is log_before


def test_steer_and_follow_up_delegate_to_the_inbox_at_the_agent_surface() -> None:
    """Pinned Pi's `Agent.steer()`/`Agent.followUp()` are the public API, not
    an internal queue's own methods (`AG-011`)."""
    instance = _instance()
    turn_msg = UserMessage(content=(TextBlock(text="turn"),), timestamp=1)
    step_msg = UserMessage(content=(TextBlock(text="step"),), timestamp=1)

    instance.follow_up(turn_msg)
    instance.steer(step_msg)

    assert [e.message for e in instance.inbox.pending(InboxTarget.NEXT_TURN)] == [turn_msg]
    assert [e.message for e in instance.inbox.pending(InboxTarget.NEXT_STEP)] == [step_msg]


def test_inject_delegates_to_the_inbox_without_waking() -> None:
    instance = _instance()

    instance.inject(UserMessage(content=(TextBlock(text="ambient"),), timestamp=1))

    assert instance.has_queued_messages()
    assert not instance.inbox.wake_requested


def test_agent_level_queue_clearing_delegates_to_the_inbox() -> None:
    """Pinned Pi's `Agent.clearSteeringQueue()`/`clearFollowUpQueue()`/
    `clearAllQueues()`/`hasQueuedMessages()` (`AG-013`)."""
    instance = _instance()
    instance.follow_up(UserMessage(content=(TextBlock(text="turn"),), timestamp=1))
    instance.steer(UserMessage(content=(TextBlock(text="step"),), timestamp=1))

    instance.clear_steering_queue()
    assert instance.has_queued_messages()  # follow-up survives

    instance.steer(UserMessage(content=(TextBlock(text="step-2"),), timestamp=1))
    instance.clear_follow_up_queue()
    assert instance.has_queued_messages()  # steering survives

    instance.clear_all_queues()
    assert not instance.has_queued_messages()


def test_reset_while_running_is_rejected_atomically() -> None:
    """Pinned Pi: `if (this.activeRun) { throw new Error("Agent is already
    processing. Wait for completion before resetting.") }` -- exact text,
    and no partial mutation (`L07-R003`)."""
    instance = _instance()
    instance.error_message = "boom"
    instance.inbox.followup(UserMessage(content=(TextBlock(text="turn"),), timestamp=1))
    instance.set_status(AgentStatus.RUNNING)

    with pytest.raises(AgentActiveError, match="Wait for completion before resetting"):
        instance.reset()

    assert instance.error_message == "boom"
    assert instance.inbox.has_pending()
    assert instance.status is AgentStatus.RUNNING
