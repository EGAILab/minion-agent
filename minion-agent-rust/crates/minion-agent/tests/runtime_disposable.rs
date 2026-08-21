use std::{
    collections::{BTreeSet, HashSet},
    future::Future,
    sync::{Arc, mpsc},
    task::{Context, Poll},
    thread,
};

use futures::{channel::oneshot, executor::block_on, future::BoxFuture, task::noop_waker_ref};
use minion_agent::{DisposeError, EffectStore, EventName, RuntimeError, ServiceName};
use parking_lot::Mutex;

fn successful_disposer(
    label: &'static str,
    seen: Arc<Mutex<Vec<&'static str>>>,
) -> impl FnOnce() -> BoxFuture<'static, Result<(), DisposeError>> + Send {
    move || {
        Box::pin(async move {
            seen.lock().push(label);
            Ok(())
        })
    }
}

#[test]
fn normative_names_have_value_identity_and_accept_qualified_events() {
    let first = ServiceName::new(String::from("tools")).unwrap();
    let second = ServiceName::new(String::from("tools")).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.as_str(), "tools");
    assert_eq!(
        EventName::new("test/transform").unwrap().as_str(),
        "test/transform"
    );
    assert_eq!(
        EventName::new("test/transform-name").unwrap().as_str(),
        "test/transform-name"
    );

    let mut hashed = HashSet::new();
    hashed.insert(first.clone());
    hashed.insert(second.clone());
    assert_eq!(hashed.len(), 1);

    let mut ordered = BTreeSet::new();
    ordered.insert(first);
    ordered.insert(second);
    assert_eq!(ordered.len(), 1);
}

#[test]
fn normative_names_reject_empty_uppercase_and_malformed_values() {
    for invalid in ["", "Tools", "tool-name", "test/transform"] {
        assert!(matches!(
            ServiceName::new(invalid),
            Err(RuntimeError::InvalidName(value)) if value == invalid
        ));
    }

    for invalid in [
        "",
        "Test/transform",
        "/test",
        "test/",
        "test//transform",
        "test/-transform",
    ] {
        assert!(matches!(
            EventName::new(invalid),
            Err(RuntimeError::InvalidName(value)) if value == invalid
        ));
    }
}

#[test]
fn disposers_run_sequentially_in_reverse_order() {
    let effects = Arc::new(EffectStore::new());
    let (started_sender, started_receiver) = mpsc::channel();
    let mut releases = Vec::new();

    for label in ["first", "second", "third"] {
        let started_sender = started_sender.clone();
        let (release_sender, release_receiver) = oneshot::channel::<()>();
        releases.push(release_sender);
        effects
            .push(label, move || {
                Box::pin(async move {
                    started_sender.send(label).unwrap();
                    release_receiver.await.unwrap();
                    Ok(())
                })
            })
            .unwrap();
    }
    drop(started_sender);

    let close = thread::spawn({
        let effects = Arc::clone(&effects);
        move || block_on(effects.close_and_dispose())
    });

    assert_eq!(started_receiver.recv().unwrap(), "third");
    assert!(started_receiver.try_recv().is_err());
    releases.pop().unwrap().send(()).unwrap();

    assert_eq!(started_receiver.recv().unwrap(), "second");
    assert!(started_receiver.try_recv().is_err());
    releases.pop().unwrap().send(()).unwrap();

    assert_eq!(started_receiver.recv().unwrap(), "first");
    assert!(started_receiver.try_recv().is_err());
    releases.pop().unwrap().send(()).unwrap();

    close.join().unwrap().unwrap();
}

#[test]
fn a_second_bulk_disposal_is_a_no_op() {
    let effects = EffectStore::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    effects
        .push("only", successful_disposer("only", Arc::clone(&seen)))
        .unwrap();

    block_on(effects.close_and_dispose()).unwrap();
    block_on(effects.close_and_dispose()).unwrap();

    assert_eq!(&*seen.lock(), &["only"]);
}

#[test]
fn disposal_returns_failures_in_reverse_execution_order_after_running_every_entry() {
    let effects = EffectStore::new();
    let seen = Arc::new(Mutex::new(Vec::new()));

    for (label, message) in [
        ("first", None),
        ("second", Some("two")),
        ("third", Some("three")),
    ] {
        let seen = Arc::clone(&seen);
        effects
            .push(label, move || {
                Box::pin(async move {
                    seen.lock().push(label);
                    match message {
                        Some(message) => Err(DisposeError::new(label, message)),
                        None => Ok(()),
                    }
                })
            })
            .unwrap();
    }

    let errors = block_on(effects.close_and_dispose()).unwrap_err();

    assert_eq!(&*seen.lock(), &["third", "second", "first"]);
    assert_eq!(
        errors.as_slice(),
        &[
            DisposeError::new("third", "three"),
            DisposeError::new("second", "two"),
        ]
    );
}

#[test]
fn close_prevents_later_effect_registration() {
    let effects = EffectStore::new();
    effects.close();

    assert!(matches!(
        effects.push("late", || Box::pin(async { Ok(()) })),
        Err(RuntimeError::InactiveOwner { .. })
    ));
}

#[test]
fn a_handle_winning_the_slot_leaves_bulk_unwind_as_a_no_op() {
    let effects = Arc::new(EffectStore::new());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (started_sender, started_receiver) = oneshot::channel::<()>();
    let (release_sender, release_receiver) = oneshot::channel::<()>();
    let disposer_seen = Arc::clone(&seen);

    let handle = effects
        .push("shared", move || {
            Box::pin(async move {
                started_sender.send(()).unwrap();
                release_receiver.await.unwrap();
                disposer_seen.lock().push("shared");
                Ok(())
            })
        })
        .unwrap();

    let handle_disposal = thread::spawn(move || block_on(handle.dispose()));
    block_on(started_receiver).unwrap();

    let bulk_disposal = thread::spawn({
        let effects = Arc::clone(&effects);
        move || block_on(effects.close_and_dispose())
    });
    release_sender.send(()).unwrap();

    handle_disposal.join().unwrap().unwrap();
    bulk_disposal.join().unwrap().unwrap();
    assert_eq!(&*seen.lock(), &["shared"]);
}

#[test]
fn bulk_unwind_winning_the_slot_leaves_the_handle_as_a_no_op() {
    let effects = Arc::new(EffectStore::new());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (started_sender, started_receiver) = oneshot::channel::<()>();
    let (release_sender, release_receiver) = oneshot::channel::<()>();
    let disposer_seen = Arc::clone(&seen);

    let handle = effects
        .push("shared", move || {
            Box::pin(async move {
                started_sender.send(()).unwrap();
                release_receiver.await.unwrap();
                disposer_seen.lock().push("shared");
                Ok(())
            })
        })
        .unwrap();

    let bulk_disposal = thread::spawn({
        let effects = Arc::clone(&effects);
        move || block_on(effects.close_and_dispose())
    });
    block_on(started_receiver).unwrap();

    let handle_disposal = thread::spawn(move || block_on(handle.dispose()));
    release_sender.send(()).unwrap();

    bulk_disposal.join().unwrap().unwrap();
    handle_disposal.join().unwrap().unwrap();
    assert_eq!(&*seen.lock(), &["shared"]);
}

#[test]
fn concurrent_bulk_unwind_waits_for_the_owner_and_does_not_split_errors() {
    let effects = Arc::new(EffectStore::new());
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = oneshot::channel::<()>();
    let older_started_sender = started_sender.clone();

    effects
        .push("older", move || {
            Box::pin(async move {
                older_started_sender.send("older").unwrap();
                Err(DisposeError::new("older", "old failure"))
            })
        })
        .unwrap();
    effects
        .push("newest", move || {
            Box::pin(async move {
                started_sender.send("newest").unwrap();
                release_receiver.await.unwrap();
                Err(DisposeError::new("newest", "new failure"))
            })
        })
        .unwrap();

    let owner = thread::spawn({
        let effects = Arc::clone(&effects);
        move || block_on(effects.close_and_dispose())
    });
    assert_eq!(started_receiver.recv().unwrap(), "newest");

    let mut follower = Box::pin(effects.close_and_dispose());
    let mut context = Context::from_waker(noop_waker_ref());
    assert!(matches!(
        follower.as_mut().poll(&mut context),
        Poll::Pending
    ));
    assert!(started_receiver.try_recv().is_err());

    release_sender.send(()).unwrap();
    let owner_errors = owner.join().unwrap().unwrap_err();
    assert_eq!(
        owner_errors.as_slice(),
        &[
            DisposeError::new("newest", "new failure"),
            DisposeError::new("older", "old failure"),
        ]
    );
    assert_eq!(started_receiver.recv().unwrap(), "older");
    assert_eq!(
        block_on(follower).unwrap_err().as_slice(),
        &[
            DisposeError::new("newest", "new failure"),
            DisposeError::new("older", "old failure"),
        ]
    );
}

#[test]
fn dropped_bulk_owner_transfers_the_full_aggregate_to_a_survivor() {
    let effects = Arc::new(EffectStore::new());
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = oneshot::channel::<()>();
    let older_started_sender = started_sender.clone();

    effects
        .push("older", move || {
            Box::pin(async move {
                older_started_sender.send("older").unwrap();
                Err(DisposeError::new("older", "old failure"))
            })
        })
        .unwrap();
    effects
        .push("newest", move || {
            Box::pin(async move {
                started_sender.send("newest").unwrap();
                release_receiver.await.unwrap();
                Err(DisposeError::new("newest", "new failure"))
            })
        })
        .unwrap();

    let mut owner = Box::pin(effects.close_and_dispose());
    let mut context = Context::from_waker(noop_waker_ref());
    assert!(matches!(owner.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(started_receiver.recv().unwrap(), "newest");
    drop(owner);

    let mut survivor = Box::pin(effects.close_and_dispose());
    assert!(matches!(
        survivor.as_mut().poll(&mut context),
        Poll::Pending
    ));
    assert!(started_receiver.try_recv().is_err());

    release_sender.send(()).unwrap();
    let errors = block_on(survivor).unwrap_err();
    assert_eq!(
        errors.as_slice(),
        &[
            DisposeError::new("newest", "new failure"),
            DisposeError::new("older", "old failure"),
        ]
    );
    assert_eq!(started_receiver.recv().unwrap(), "older");
    assert!(started_receiver.try_recv().is_err());
    assert!(block_on(effects.close_and_dispose()).is_ok());
}

#[test]
fn pending_followers_keep_the_aggregate_after_the_owner_is_dropped() {
    let effects = Arc::new(EffectStore::new());
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = oneshot::channel::<()>();
    let older_started_sender = started_sender.clone();

    effects
        .push("older", move || {
            Box::pin(async move {
                older_started_sender.send("older").unwrap();
                Err(DisposeError::new("older", "old failure"))
            })
        })
        .unwrap();
    effects
        .push("newest", move || {
            Box::pin(async move {
                started_sender.send("newest").unwrap();
                release_receiver.await.unwrap();
                Err(DisposeError::new("newest", "new failure"))
            })
        })
        .unwrap();

    let mut owner = Box::pin(effects.close_and_dispose());
    let mut follower = Box::pin(effects.close_and_dispose());
    let mut context = Context::from_waker(noop_waker_ref());
    assert!(matches!(owner.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(started_receiver.recv().unwrap(), "newest");
    assert!(matches!(
        follower.as_mut().poll(&mut context),
        Poll::Pending
    ));

    release_sender.send(()).unwrap();
    let driver_errors = block_on(effects.close_and_dispose()).unwrap_err();
    assert_eq!(
        driver_errors.as_slice(),
        &[
            DisposeError::new("newest", "new failure"),
            DisposeError::new("older", "old failure"),
        ]
    );
    assert_eq!(started_receiver.recv().unwrap(), "older");
    drop(owner);

    assert_eq!(
        block_on(follower).unwrap_err().as_slice(),
        &[
            DisposeError::new("newest", "new failure"),
            DisposeError::new("older", "old failure"),
        ]
    );
    assert!(block_on(effects.close_and_dispose()).is_ok());
}
