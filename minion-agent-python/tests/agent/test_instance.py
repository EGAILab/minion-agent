"""An instance is one live execution identity."""

from minion_agent.agent.identity import AgentDefinition, AgentStatus
from minion_agent.agent.instance import AgentInstance, instance_scope_key
from minion_agent.llm import ModelId
from minion_agent.runtime import Context, scope_of
from minion_agent.session import SessionLog


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
