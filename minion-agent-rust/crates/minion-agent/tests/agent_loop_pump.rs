use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use minion_agent::{
    DynPluginSpec, PluginInitError, PluginSpec, Runtime,
    agent::{AgentDefinition, AgentInstance, ClaimPolicy},
    agent_loop::{
        AgentEndReason, AgentEvent, AgentListenerError, AgentLoop, RunCause, TurnStopping,
        register_agent_listener, register_should_stop_after_turn_listener,
    },
    llm::{
        AssistantContentBlock, AssistantMessage, LlmService, Message, ModelIdentity, Script,
        ScriptItem, ScriptedAdapter, StopReason, StreamChunk, TextBlock, Usage, UserContent,
        UserMessage,
    },
    session::Session,
};
use parking_lot::Mutex;
use serde_json::{Value, json};

fn run(future: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap()
        .block_on(future);
}

fn identity() -> ModelIdentity {
    ModelIdentity::new("provider", "api", "model").unwrap()
}

fn user(text: &str) -> Message {
    Message::User(UserMessage::new(UserContent::Text(text.into()), 1.0))
}

fn text_turn(text: &str) -> Script {
    Script::new([ScriptItem::Chunk(Box::new(StreamChunk::Done {
        reason: minion_agent::llm::DoneReason::Stop,
        message: AssistantMessage::new(
            identity(),
            vec![AssistantContentBlock::Text(TextBlock::new(text))],
            Usage::default(),
            StopReason::Stop,
            2.0,
        ),
    }))])
}

fn setup(scripts: impl IntoIterator<Item = Script>) -> (Runtime, AgentLoop, Arc<AgentInstance>) {
    let runtime = Runtime::new();
    let llm = Arc::new(LlmService::new());
    llm.register(identity(), Arc::new(ScriptedAdapter::new(scripts)));
    let agent = Arc::new(AgentInstance::new(
        "room-a",
        AgentDefinition::new("ada", "system", identity()),
        Session::new("room-a", [] as [&str; 0]).unwrap(),
        Some(runtime.context()),
        None,
    ));
    let driver = AgentLoop::new(Arc::clone(&agent), runtime.context(), llm);
    (runtime, driver, agent)
}

type Listener = Arc<dyn Fn(AgentEvent) -> Result<(), AgentListenerError> + Send + Sync>;

fn listener_plugin(listener: Listener) -> DynPluginSpec {
    PluginSpec::<Value>::new(
        "pump-lifecycle-listener",
        vec![],
        || json!({}),
        move |context, _config| {
            let listener = Arc::clone(&listener);
            async move {
                register_agent_listener(&context, move |event| {
                    let listener = Arc::clone(&listener);
                    async move { listener(event) }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

fn stop_first_turn_plugin() -> DynPluginSpec {
    let first = Arc::new(AtomicBool::new(true));
    PluginSpec::<Value>::new(
        "stop-first-turn",
        vec![],
        || json!({}),
        move |context, _config| {
            let first = Arc::clone(&first);
            async move {
                register_should_stop_after_turn_listener(&context, move |_context| {
                    let first = Arc::clone(&first);
                    async move {
                        Ok(if first.swap(false, Ordering::SeqCst) {
                            TurnStopping::Stop
                        } else {
                            TurnStopping::Continue
                        })
                    }
                })
                .map_err(|error| PluginInitError::new(error.to_string()))?;
                Ok(())
            }
        },
    )
    .erase()
}

fn cause(id: &str, origin: Option<Value>) -> RunCause {
    RunCause {
        id: id.to_owned(),
        origin,
    }
}

#[test]
fn claim_all_pump_exposes_every_cause_and_consumes_wake_only_when_idle() {
    run(async {
        let (runtime, mut driver, agent) = setup([text_turn("answer")]);
        driver.set_next_turn_policy(ClaimPolicy::All);
        let events = Arc::new(Mutex::new(Vec::new()));
        runtime
            .mount(
                &listener_plugin({
                    let events = Arc::clone(&events);
                    Arc::new(move |event| {
                        events.lock().push(event);
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();
        let one = agent.follow_up(user("one"), Some(json!("a")));
        let two = agent.follow_up(user("two"), Some(json!({"nested": [1, null]})));
        let three = agent.follow_up(user("three"), Some(Value::Null));

        driver.run_until_idle().await.unwrap();

        let expected = vec![
            cause(&one.id, one.origin.clone()),
            cause(&two.id, two.origin.clone()),
            cause(&three.id, three.origin.clone()),
        ];
        let events = events.lock();
        assert!(matches!(&events[0], AgentEvent::AgentStart { causes } if causes == &expected));
        assert!(matches!(
            events.last().unwrap(),
            AgentEvent::AgentEnd { reason: AgentEndReason::Completed, causes, messages }
                if causes == &expected && messages.len() == 4
        ));
        assert!(!agent.inbox().wake_requested());
        assert!(!agent.inbox().has_pending());
    });
}

#[test]
fn one_at_a_time_followups_accumulate_causes_on_the_same_run() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("first"), text_turn("second")]);
        let events = Arc::new(Mutex::new(Vec::new()));
        runtime
            .mount(
                &listener_plugin({
                    let events = Arc::clone(&events);
                    Arc::new(move |event| {
                        events.lock().push(event);
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();
        let one = agent.follow_up(user("one"), Some(json!({"source": "first"})));
        let two = agent.follow_up(user("two"), Some(json!({"source": null})));

        driver.run_until_idle().await.unwrap();

        let events = events.lock();
        let starts = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::AgentStart { causes } => Some(causes.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let ends = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::AgentEnd { reason, causes, .. } => Some((*reason, causes.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(starts, vec![vec![cause(&one.id, one.origin.clone())]]);
        assert_eq!(
            ends,
            vec![(
                AgentEndReason::Completed,
                vec![
                    cause(&one.id, one.origin.clone()),
                    cause(&two.id, two.origin.clone()),
                ],
            )]
        );
    });
}

#[test]
fn pump_opens_a_second_run_when_should_stop_leaves_followup_pending() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("first"), text_turn("second")]);
        let events = Arc::new(Mutex::new(Vec::new()));
        runtime.mount(&stop_first_turn_plugin(), json!({})).unwrap();
        runtime
            .mount(
                &listener_plugin({
                    let events = Arc::clone(&events);
                    Arc::new(move |event| {
                        events.lock().push(event);
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();
        let one = agent.follow_up(user("one"), Some(json!("a")));
        let two = agent.follow_up(user("two"), Some(json!("b")));

        driver.run_until_idle().await.unwrap();

        let ends = events
            .lock()
            .iter()
            .filter_map(|event| match event {
                AgentEvent::AgentEnd { reason, causes, .. } => Some((*reason, causes.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            ends,
            vec![
                (
                    AgentEndReason::Stopped,
                    vec![cause(&one.id, one.origin.clone())],
                ),
                (
                    AgentEndReason::Completed,
                    vec![cause(&two.id, two.origin.clone())],
                ),
            ]
        );
        assert!(!agent.inbox().wake_requested());
    });
}

#[test]
fn assistant_last_continue_uses_the_claimed_envelopes_as_run_causes() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("continued")]);
        agent
            .session()
            .append_message(Message::Assistant(Box::new(AssistantMessage::new(
                identity(),
                vec![AssistantContentBlock::Text(TextBlock::new("history"))],
                Usage::default(),
                StopReason::Stop,
                1.0,
            ))))
            .unwrap();
        let envelope = agent.steer(user("steer"), Some(json!({"request": 7})));
        let starts = Arc::new(Mutex::new(Vec::new()));
        runtime
            .mount(
                &listener_plugin({
                    let starts = Arc::clone(&starts);
                    Arc::new(move |event| {
                        if let AgentEvent::AgentStart { causes } = event {
                            starts.lock().push(causes);
                        }
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();

        driver.continue_run().await.unwrap();

        assert_eq!(
            starts.lock().as_slice(),
            [vec![cause(&envelope.id, envelope.origin.clone())]]
        );
    });
}

#[test]
fn assistant_last_followup_continue_uses_that_exact_envelope_as_the_cause() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("continued")]);
        agent
            .session()
            .append_message(Message::Assistant(Box::new(AssistantMessage::new(
                identity(),
                vec![AssistantContentBlock::Text(TextBlock::new("history"))],
                Usage::default(),
                StopReason::Stop,
                1.0,
            ))))
            .unwrap();
        let envelope = agent.follow_up(user("follow"), Some(json!({"request": 8})));
        let starts = Arc::new(Mutex::new(Vec::new()));
        runtime
            .mount(
                &listener_plugin({
                    let starts = Arc::clone(&starts);
                    Arc::new(move |event| {
                        if let AgentEvent::AgentStart { causes } = event {
                            starts.lock().push(causes);
                        }
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();

        driver.continue_run().await.unwrap();

        assert_eq!(
            starts.lock().as_slice(),
            [vec![cause(&envelope.id, envelope.origin.clone())]]
        );
    });
}

#[test]
fn prompt_and_plain_continue_have_no_queued_input_causes() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("prompt reply"), text_turn("continued")]);
        let starts = Arc::new(Mutex::new(Vec::new()));
        runtime
            .mount(
                &listener_plugin({
                    let starts = Arc::clone(&starts);
                    Arc::new(move |event| {
                        if let AgentEvent::AgentStart { causes } = event {
                            starts.lock().push(causes);
                        }
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();

        driver
            .prompt(minion_agent::agent_loop::PromptInput::Message(user(
                "prompt",
            )))
            .await
            .unwrap();
        agent.session().append_message(user("new input")).unwrap();
        driver.continue_run().await.unwrap();

        assert_eq!(starts.lock().as_slice(), [Vec::new(), Vec::new()]);
    });
}

#[test]
fn pump_consumes_wake_without_treating_steering_as_next_turn_work() {
    run(async {
        let (_runtime, driver, agent) = setup([]);
        let steering = agent.steer(user("ambient"), Some(json!("step-only")));

        driver.run_until_idle().await.unwrap();

        assert!(!agent.inbox().wake_requested());
        assert_eq!(
            agent
                .inbox()
                .pending(minion_agent::agent::InboxTarget::Steering),
            vec![steering]
        );
        assert!(
            agent
                .inbox()
                .pending(minion_agent::agent::InboxTarget::FollowUp)
                .is_empty()
        );
    });
}

#[test]
fn failure_agent_end_retains_the_run_causes() {
    run(async {
        let (runtime, driver, agent) = setup([text_turn("first")]);
        let events = Arc::new(Mutex::new(Vec::new()));
        let turns = Arc::new(AtomicUsize::new(0));
        runtime
            .mount(
                &listener_plugin({
                    let events = Arc::clone(&events);
                    let turns = Arc::clone(&turns);
                    Arc::new(move |event| {
                        events.lock().push(event.clone());
                        if matches!(event, AgentEvent::TurnStart)
                            && turns.fetch_add(1, Ordering::SeqCst) == 1
                        {
                            return Err(AgentListenerError::new("failed"));
                        }
                        Ok(())
                    })
                }),
                json!({}),
            )
            .unwrap();
        runtime.reconcile().await.unwrap();
        let first = agent.follow_up(user("first"), Some(json!("scheduler-1")));
        let second = agent.follow_up(user("second"), Some(json!("scheduler-2")));

        driver.run_until_idle().await.unwrap();

        assert!(events.lock().iter().any(|event| matches!(
            event,
            AgentEvent::AgentEnd {
                reason: AgentEndReason::Failed,
                causes,
                messages,
            } if causes == &vec![
                    cause(&first.id, first.origin.clone()),
                    cause(&second.id, second.origin.clone()),
                ]
                && messages.len() == 1
        )));
    });
}
