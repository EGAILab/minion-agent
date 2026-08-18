use std::{
    cell::Cell,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
    thread,
    time::Duration,
};

use futures::{channel::oneshot, executor::block_on, future::poll_fn, task::noop_waker_ref};
use minion_agent::{
    DispatchMode, EffectStore, EventBus, EventError, EventListenerError, EventName, EventSpec,
    ScopeTree, WaterfallError,
};
use parking_lot::Mutex;

fn event(name: &str) -> EventName {
    EventName::new(name).unwrap()
}

#[test]
fn declarations_are_idempotent_only_for_the_same_mode_and_rust_contract() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let emit = EventSpec::<u32, ()>::new(event("test/contract"), DispatchMode::Emit, |_| ());

    assert_eq!(emit.name().as_str(), "test/contract");
    assert_eq!(emit.mode(), DispatchMode::Emit);
    assert_eq!(emit.clone().name(), emit.name());

    bus.declare(&emit).unwrap();
    bus.declare(&EventSpec::<u32, ()>::new(
        event("test/contract"),
        DispatchMode::Emit,
        |_| (),
    ))
    .unwrap();

    let mode_error = bus
        .declare(&EventSpec::<u32, ()>::new(
            event("test/contract"),
            DispatchMode::Serial,
            |_| (),
        ))
        .unwrap_err();
    assert!(matches!(
        mode_error,
        EventError::DeclarationModeMismatch {
            expected: DispatchMode::Emit,
            actual: DispatchMode::Serial,
            ..
        }
    ));

    let payload_error = bus
        .declare(&EventSpec::<String, ()>::new(
            event("test/contract"),
            DispatchMode::Emit,
            |_| (),
        ))
        .unwrap_err();
    assert!(matches!(
        payload_error,
        EventError::DeclarationTypeMismatch { .. }
    ));

    let result_error = bus
        .declare(&EventSpec::<u32, String>::new(
            event("test/contract"),
            DispatchMode::Emit,
            |_| String::new(),
        ))
        .unwrap_err();
    assert!(matches!(
        result_error,
        EventError::DeclarationTypeMismatch { .. }
    ));
}

#[test]
fn the_first_terminal_is_retained_when_a_contract_is_redeclared() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let first = EventSpec::new(
        event("test/terminal-contract"),
        DispatchMode::Waterfall,
        |_: &()| "first",
    );
    let second = EventSpec::new(
        event("test/terminal-contract"),
        DispatchMode::Waterfall,
        |_: &()| "second",
    );

    bus.declare(&first).unwrap();
    bus.declare(&second).unwrap();

    assert_eq!(block_on(bus.waterfall(&second, (), None)).unwrap(), "first");
}

#[test]
fn registration_and_dispatch_reject_undeclared_or_incompatible_specs() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let effects = EffectStore::new();
    let undeclared = EventSpec::new(event("test/undeclared"), DispatchMode::Emit, |_: &()| ());
    assert!(matches!(
        bus.on_emit(&undeclared, &effects, None, |_| {}),
        Err(EventError::Undeclared { .. })
    ));
    assert!(matches!(
        bus.emit(&undeclared, &(), None),
        Err(EventError::Undeclared { .. })
    ));

    let serial = EventSpec::new(event("test/wrong-mode"), DispatchMode::Serial, |_: &()| ());
    bus.declare(&serial).unwrap();
    assert!(matches!(
        bus.on_emit(&serial, &effects, None, |_| {}),
        Err(EventError::DispatchModeMismatch {
            expected: DispatchMode::Emit,
            actual: DispatchMode::Serial,
            ..
        })
    ));

    let fake_emit = EventSpec::new(event("test/wrong-mode"), DispatchMode::Emit, |_: &()| ());
    assert!(matches!(
        bus.emit(&fake_emit, &(), None),
        Err(EventError::DispatchModeMismatch {
            expected: DispatchMode::Emit,
            actual: DispatchMode::Serial,
            ..
        })
    ));

    let typed = EventSpec::new(
        event("test/typed-dispatch"),
        DispatchMode::Emit,
        |_: &u32| (),
    );
    bus.declare(&typed).unwrap();
    let wrong_payload = EventSpec::new(
        event("test/typed-dispatch"),
        DispatchMode::Emit,
        |_: &String| (),
    );
    assert!(matches!(
        bus.emit(&wrong_payload, &String::new(), None),
        Err(EventError::DeclarationTypeMismatch { .. })
    ));
}

#[test]
fn registration_and_dispatch_validate_scope_tree_and_activity() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let inactive = tree.create_child(&root).unwrap();
    let other_tree = ScopeTree::new();
    let foreign = other_tree.create_root();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/scope-errors"), DispatchMode::Emit, |_: &()| ());
    let effects = EffectStore::new();
    bus.declare(&spec).unwrap();

    assert!(matches!(
        bus.on_emit(&spec, &effects, Some(&foreign), |_| {}),
        Err(EventError::ForeignScope { .. })
    ));
    assert!(matches!(
        bus.emit(&spec, &(), Some(&foreign)),
        Err(EventError::ForeignScope { .. })
    ));

    block_on(inactive.dispose()).unwrap();
    assert!(matches!(
        bus.on_emit(&spec, &effects, Some(&inactive), |_| {}),
        Err(EventError::InactiveScope { .. })
    ));
}

#[test]
fn a_closed_lifecycle_owner_rejects_and_removes_a_listener_registration() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/closed-owner"), DispatchMode::Emit, |_: &()| ());
    let effects = EffectStore::new();
    let runs = Arc::new(AtomicUsize::new(0));
    bus.declare(&spec).unwrap();
    effects.close();

    let listener_runs = Arc::clone(&runs);
    assert!(matches!(
        bus.on_emit(&spec, &effects, None, move |_| {
            listener_runs.fetch_add(1, Ordering::SeqCst);
        }),
        Err(EventError::Lifecycle(_))
    ));
    bus.emit(&spec, &(), None).unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

#[test]
fn emit_is_synchronous_and_follows_global_registration_order() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let definition = tree.create_child(&root).unwrap();
    let turn = tree.create_child(&definition).unwrap();
    let sibling = tree.create_child(&root).unwrap();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/emit"), DispatchMode::Emit, |_: &&str| ());
    let effects = EffectStore::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    bus.declare(&spec).unwrap();

    for (label, scope) in [
        ("untagged", None),
        ("definition", Some(&definition)),
        ("sibling", Some(&sibling)),
        ("turn", Some(&turn)),
    ] {
        let seen = Arc::clone(&seen);
        bus.on_emit(&spec, &effects, scope, move |payload| {
            seen.lock().push(format!("{label}:{payload}"));
        })
        .unwrap();
    }

    bus.emit(&spec, &"payload", Some(&turn)).unwrap();
    assert_eq!(
        &*seen.lock(),
        &["untagged:payload", "definition:payload", "turn:payload"]
    );
}

#[test]
fn inactive_scope_tags_are_not_admitted_but_untagged_listeners_always_are() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let tagged_scope = tree.create_child(&root).unwrap();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/scope"), DispatchMode::Emit, |_: &()| ());
    let effects = EffectStore::new();
    let tagged_runs = Arc::new(AtomicUsize::new(0));
    let untagged_runs = Arc::new(AtomicUsize::new(0));
    bus.declare(&spec).unwrap();

    let runs = Arc::clone(&tagged_runs);
    bus.on_emit(&spec, &effects, Some(&tagged_scope), move |_| {
        runs.fetch_add(1, Ordering::SeqCst);
    })
    .unwrap();
    let runs = Arc::clone(&untagged_runs);
    bus.on_emit(&spec, &effects, None, move |_| {
        runs.fetch_add(1, Ordering::SeqCst);
    })
    .unwrap();

    block_on(tagged_scope.dispose()).unwrap();
    bus.emit(&spec, &(), Some(&tagged_scope)).unwrap();

    assert_eq!(tagged_runs.load(Ordering::SeqCst), 0);
    assert_eq!(untagged_runs.load(Ordering::SeqCst), 1);
}

#[test]
fn listener_registration_is_owned_by_its_effect_store() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/lifecycle"), DispatchMode::Emit, |_: &()| ());
    let effects = EffectStore::new();
    let runs = Arc::new(AtomicUsize::new(0));
    bus.declare(&spec).unwrap();

    let listener_runs = Arc::clone(&runs);
    let handle = bus
        .on_emit(&spec, &effects, None, move |_| {
            listener_runs.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
    block_on(effects.close_and_dispose()).unwrap();
    block_on(handle.dispose()).unwrap();
    bus.emit(&spec, &(), None).unwrap();

    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

#[test]
fn tagged_listener_cleanup_is_owned_by_its_scope_not_the_supplied_store() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let scope = tree.create_child(&root).unwrap();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/scoped-owner"), DispatchMode::Emit, |_: &()| ());
    let supplied_owner = EffectStore::new();
    let drops = Arc::new(AtomicUsize::new(0));
    bus.declare(&spec).unwrap();

    bus.on_emit(&spec, &supplied_owner, Some(&scope), {
        let registration = RegistrationDrop(Arc::clone(&drops));
        move |_| {
            let _ = &registration;
        }
    })
    .unwrap();

    block_on(scope.dispose()).unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    block_on(supplied_owner.close_and_dispose()).unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    let rejected_drops = Arc::new(AtomicUsize::new(0));
    let error = match bus.on_emit(&spec, &supplied_owner, Some(&scope), {
        let registration = RegistrationDrop(Arc::clone(&rejected_drops));
        move |_| {
            let _ = &registration;
        }
    }) {
        Ok(_) => panic!("inactive scope registration unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(error, EventError::InactiveScope { .. }));
    assert_eq!(rejected_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn listener_callback_is_dropped_after_releasing_the_event_mutex() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(
        event("test/reentrant-drop"),
        DispatchMode::Emit,
        |_: &()| (),
    );
    let effects = EffectStore::new();
    let (completed_sender, completed_receiver) = mpsc::channel();
    bus.declare(&spec).unwrap();

    let handle = bus
        .on_emit(&spec, &effects, None, {
            let reentrant = ReentrantRegistrationDrop {
                bus: bus.clone(),
                spec: spec.clone(),
                completed: Some(completed_sender),
            };
            move |_| {
                let _ = &reentrant;
            }
        })
        .unwrap();
    let disposal = thread::spawn(move || block_on(handle.dispose()));

    let reentrant_result = completed_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("listener callback drop deadlocked while re-entering the event bus");
    reentrant_result.unwrap();
    disposal.join().unwrap().unwrap();
}

#[test]
fn handle_and_effect_unwind_race_to_remove_a_listener_exactly_once() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/removal-race"), DispatchMode::Emit, |_: &()| ());
    let effects = Arc::new(EffectStore::new());
    let drops = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(3));
    bus.declare(&spec).unwrap();

    let handle = bus
        .on_emit(&spec, &effects, None, {
            let counter = RegistrationDrop(Arc::clone(&drops));
            move |_| {
                let _ = &counter;
            }
        })
        .unwrap();
    let handle_thread = thread::spawn({
        let barrier = Arc::clone(&barrier);
        move || {
            barrier.wait();
            block_on(handle.dispose()).unwrap();
        }
    });
    let unwind_thread = thread::spawn({
        let barrier = Arc::clone(&barrier);
        let effects = Arc::clone(&effects);
        move || {
            barrier.wait();
            block_on(effects.close_and_dispose()).unwrap();
        }
    });

    barrier.wait();
    handle_thread.join().unwrap();
    unwind_thread.join().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    bus.emit(&spec, &(), None).unwrap();
}

#[test]
fn parallel_callbacks_overlap_and_failures_keep_registration_order() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/parallel"), DispatchMode::Parallel, |_: &()| ());
    let effects = EffectStore::new();
    let (first_started_sender, first_started_receiver) = oneshot::channel();
    let (release_first, first_gate) = oneshot::channel();
    let first_started_sender = Arc::new(Mutex::new(Some(first_started_sender)));
    let first_gate = Arc::new(Mutex::new(Some(first_gate)));
    let first_started_receiver = Arc::new(Mutex::new(Some(first_started_receiver)));
    let release_first = Arc::new(Mutex::new(Some(release_first)));
    bus.declare(&spec).unwrap();

    bus.on_parallel(&spec, &effects, None, {
        let first_started_sender = Arc::clone(&first_started_sender);
        let first_gate = Arc::clone(&first_gate);
        move |_| {
            let started = first_started_sender.lock().take().unwrap();
            let gate = first_gate.lock().take().unwrap();
            async move {
                started.send(()).unwrap();
                gate.await.unwrap();
                Err(EventListenerError::new("first"))
            }
        }
    })
    .unwrap();
    bus.on_parallel(&spec, &effects, None, {
        let first_started_receiver = Arc::clone(&first_started_receiver);
        let release_first = Arc::clone(&release_first);
        move |_| {
            let started = first_started_receiver.lock().take().unwrap();
            let release = release_first.lock().take().unwrap();
            async move {
                started.await.unwrap();
                release.send(()).unwrap();
                Err(EventListenerError::new("second"))
            }
        }
    })
    .unwrap();

    let error = block_on(bus.parallel(&spec, (), None)).unwrap_err();
    let EventError::Parallel(errors) = error else {
        panic!("expected parallel aggregate");
    };
    assert_eq!(
        errors.as_slice(),
        &[
            EventListenerError::new("first"),
            EventListenerError::new("second"),
        ]
    );
    assert_eq!(
        errors.into_inner(),
        vec![
            EventListenerError::new("first"),
            EventListenerError::new("second"),
        ]
    );
}

#[test]
fn parallel_without_failures_has_no_result() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(
        event("test/parallel-ok"),
        DispatchMode::Parallel,
        |_: &()| (),
    );
    let effects = EffectStore::new();
    bus.declare(&spec).unwrap();
    bus.on_parallel(&spec, &effects, None, |_| async { Ok(()) })
        .unwrap();

    assert!(block_on(bus.parallel(&spec, (), None)).is_ok());
}

#[test]
fn serial_runs_in_registration_order_and_returns_the_last_result() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(event("test/serial"), DispatchMode::Serial, |_: &u32| {
        String::new()
    });
    let empty = EventSpec::new(
        event("test/serial-empty"),
        DispatchMode::Serial,
        |_: &u32| String::new(),
    );
    let effects = EffectStore::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    bus.declare(&spec).unwrap();
    bus.declare(&empty).unwrap();

    for label in ["first", "second"] {
        let seen = Arc::clone(&seen);
        bus.on_serial(&spec, &effects, None, move |payload| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock().push(label);
                format!("{label}-{payload}")
            }
        })
        .unwrap();
    }

    assert_eq!(
        block_on(bus.serial(&spec, 7, None)).unwrap(),
        Some("second-7".to_owned())
    );
    assert_eq!(&*seen.lock(), &["first", "second"]);
    assert_eq!(block_on(bus.serial(&empty, 7, None)).unwrap(), None);
}

#[test]
fn a_dispatch_uses_its_listener_snapshot_even_when_a_handle_is_disposed() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(
        event("test/snapshot"),
        DispatchMode::Serial,
        |_: &()| "terminal",
    );
    let effects = EffectStore::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = oneshot::channel();
    let release_receiver = Arc::new(Mutex::new(Some(release_receiver)));
    bus.declare(&spec).unwrap();

    bus.on_serial(&spec, &effects, None, {
        let seen = Arc::clone(&seen);
        let release_receiver = Arc::clone(&release_receiver);
        move |_| {
            let release_receiver = release_receiver.lock().take();
            let seen = Arc::clone(&seen);
            let started_sender = started_sender.clone();
            async move {
                seen.lock().push("a");
                if let Some(release_receiver) = release_receiver {
                    started_sender.send(()).unwrap();
                    release_receiver.await.unwrap();
                }
                "a"
            }
        }
    })
    .unwrap();
    let second = bus
        .on_serial(&spec, &effects, None, {
            let seen = Arc::clone(&seen);
            move |_| {
                let seen = Arc::clone(&seen);
                async move {
                    seen.lock().push("b");
                    "b"
                }
            }
        })
        .unwrap();

    let mut current = Box::pin(bus.serial(&spec, (), None));
    let mut context = Context::from_waker(noop_waker_ref());
    assert!(matches!(current.as_mut().poll(&mut context), Poll::Pending));
    started_receiver.recv().unwrap();
    block_on(second.dispose()).unwrap();
    release_sender.send(()).unwrap();

    assert_eq!(block_on(current).unwrap(), Some("b"));
    assert_eq!(&*seen.lock(), &["a", "b"]);
    assert_eq!(block_on(bus.serial(&spec, (), None)).unwrap(), Some("a"));
    assert_eq!(&*seen.lock(), &["a", "b", "a"]);
}

#[test]
fn waterfall_supports_replacement_delegation_and_both_result_transformations() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(
        event("test/waterfall"),
        DispatchMode::Waterfall,
        |payload: &String| format!("terminal<{payload}>"),
    );
    let effects = EffectStore::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    bus.declare(&spec).unwrap();

    bus.on_waterfall(&spec, &effects, None, {
        let seen = Arc::clone(&seen);
        move |payload, next| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock().push(format!("outer-before:{payload}"));
                let downstream = next.call(Some(format!("{payload}-outer"))).await?;
                seen.lock().push(format!("outer-after:{downstream}"));
                Ok(format!("outer<{downstream}>"))
            }
        }
    })
    .unwrap();
    bus.on_waterfall(&spec, &effects, None, {
        let seen = Arc::clone(&seen);
        move |payload, next| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock().push(format!("inner:{payload}"));
                let terminal = next.call(Some(format!("{payload}-inner"))).await?;
                Ok(format!("inner<{terminal}>"))
            }
        }
    })
    .unwrap();

    let result = block_on(bus.waterfall(&spec, "original".to_owned(), None)).unwrap();
    assert_eq!(result, "outer<inner<terminal<original-outer-inner>>>");
    assert_eq!(
        &*seen.lock(),
        &[
            "outer-before:original",
            "inner:original-outer",
            "outer-after:inner<terminal<original-outer-inner>>",
        ]
    );
}

#[test]
fn waterfall_short_circuits_without_running_downstream() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(
        event("test/waterfall-short"),
        DispatchMode::Waterfall,
        |payload: &&str| *payload,
    );
    let effects = EffectStore::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    bus.declare(&spec).unwrap();

    bus.on_waterfall(&spec, &effects, None, {
        let seen = Arc::clone(&seen);
        move |_, _| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock().push("decider");
                Ok("decided")
            }
        }
    })
    .unwrap();
    bus.on_waterfall(&spec, &effects, None, {
        let seen = Arc::clone(&seen);
        move |_, _| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock().push("never");
                Ok("never")
            }
        }
    })
    .unwrap();

    assert_eq!(
        block_on(bus.waterfall(&spec, "original", None)).unwrap(),
        "decided"
    );
    assert_eq!(&*seen.lock(), &["decider"]);
}

#[test]
fn empty_waterfall_invokes_the_required_terminal() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(
        event("test/waterfall-empty"),
        DispatchMode::Waterfall,
        |payload: &u32| payload + 1,
    );
    bus.declare(&spec).unwrap();

    assert_eq!(block_on(bus.waterfall(&spec, 41, None)).unwrap(), 42);
}

#[test]
fn next_rejects_a_second_call_while_the_first_call_is_pending() {
    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(
        event("test/waterfall-greedy"),
        DispatchMode::Waterfall,
        |payload: &u32| *payload,
    );
    let effects = EffectStore::new();
    let (_never_release, gate) = oneshot::channel::<()>();
    let gate = Arc::new(Mutex::new(Some(gate)));
    bus.declare(&spec).unwrap();

    bus.on_waterfall(&spec, &effects, None, move |_, next| async move {
        let mut first = next.call(None);
        poll_fn(|context| match first.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("the first next call must still be pending"),
        })
        .await;
        next.call(Some(2)).await
    })
    .unwrap();
    bus.on_waterfall(&spec, &effects, None, {
        let gate = Arc::clone(&gate);
        move |payload, _| {
            let gate = gate.lock().take().unwrap();
            async move {
                let _ = gate.await;
                Ok(payload)
            }
        }
    })
    .unwrap();

    let error = block_on(bus.waterfall(&spec, 1, None)).unwrap_err();
    assert!(matches!(
        error,
        EventError::Waterfall(WaterfallError::NextAlreadyCalled)
    ));
}

#[test]
fn waterfall_delegation_is_stack_safe_for_a_deep_listener_chain() {
    const LISTENERS: usize = 5_000;
    const MAX_ALLOWED_POLL_DEPTH: usize = 64;

    let tree = ScopeTree::new();
    let bus = EventBus::new(tree);
    let spec = EventSpec::new(
        event("test/waterfall-deep"),
        DispatchMode::Waterfall,
        |payload: &usize| *payload,
    );
    let effects = EffectStore::new();
    let max_depth = Arc::new(AtomicUsize::new(0));
    bus.declare(&spec).unwrap();

    for _ in 0..LISTENERS {
        let max_depth = Arc::clone(&max_depth);
        bus.on_waterfall(&spec, &effects, None, move |payload, next| {
            PollDepthGuard::new(
                async move { next.call(Some(payload + 1)).await },
                Arc::clone(&max_depth),
                MAX_ALLOWED_POLL_DEPTH,
            )
        })
        .unwrap();
    }

    assert_eq!(block_on(bus.waterfall(&spec, 0, None)).unwrap(), LISTENERS);
    assert!(max_depth.load(Ordering::SeqCst) <= MAX_ALLOWED_POLL_DEPTH);
}

struct RegistrationDrop(Arc<AtomicUsize>);

impl Drop for RegistrationDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct ReentrantRegistrationDrop {
    bus: EventBus,
    spec: EventSpec<(), ()>,
    completed: Option<mpsc::Sender<Result<(), String>>>,
}

impl Drop for ReentrantRegistrationDrop {
    fn drop(&mut self) {
        let result = self
            .bus
            .emit(&self.spec, &(), None)
            .map_err(|error| error.to_string());
        self.completed.take().unwrap().send(result).unwrap();
    }
}

thread_local! {
    static WATERFALL_POLL_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct PollDepthGuard<F> {
    inner: Pin<Box<F>>,
    max_depth: Arc<AtomicUsize>,
    allowed_depth: usize,
}

impl<F> PollDepthGuard<F> {
    fn new(inner: F, max_depth: Arc<AtomicUsize>, allowed_depth: usize) -> Self {
        Self {
            inner: Box::pin(inner),
            max_depth,
            allowed_depth,
        }
    }
}

impl<F> Future for PollDepthGuard<F>
where
    F: Future,
{
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let depth = WATERFALL_POLL_DEPTH.with(|current| {
            let depth = current.get() + 1;
            current.set(depth);
            depth
        });
        self.max_depth.fetch_max(depth, Ordering::SeqCst);
        assert!(
            depth <= self.allowed_depth,
            "waterfall recursively polled {depth} listener frames"
        );
        let outcome = self.inner.as_mut().poll(context);
        WATERFALL_POLL_DEPTH.with(|current| current.set(depth - 1));
        outcome
    }
}
