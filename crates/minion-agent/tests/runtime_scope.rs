use std::{
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

use futures::{channel::oneshot, executor::block_on};
use minion_agent::{ScopeTree, ScopedRegistry};
use parking_lot::Mutex;

fn visible(
    registry: &ScopedRegistry<&'static str>,
    scope: minion_agent::ScopeId,
) -> Vec<&'static str> {
    registry
        .visible_from(scope)
        .into_iter()
        .map(|value| *value)
        .collect()
}

#[test]
fn scoped_entries_are_visible_nearest_first_without_sibling_or_descendant_leaks() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let definition = tree.create_child(&root).unwrap();
    let instance = tree.create_child(&definition).unwrap();
    let turn = tree.create_child(&instance).unwrap();
    let sibling = tree.create_child(&definition).unwrap();
    let registry = ScopedRegistry::new(tree.clone());

    registry.register(None, "global-first").unwrap();
    registry.register(None, "global-second").unwrap();
    registry.register(None, "global-first").unwrap();
    registry
        .register(Some(&definition), "definition-first")
        .unwrap();
    registry
        .register(Some(&definition), "definition-second")
        .unwrap();
    registry.register(Some(&instance), "instance").unwrap();
    registry.register(Some(&instance), "instance").unwrap();
    registry.register(Some(&turn), "turn").unwrap();
    registry.register(Some(&turn), "turn").unwrap();
    registry.register(Some(&sibling), "sibling").unwrap();

    assert!(tree.is_ancestor(definition.id(), turn.id()));
    assert!(!tree.is_ancestor(turn.id(), definition.id()));
    assert!(!tree.is_ancestor(sibling.id(), turn.id()));
    assert_eq!(
        visible(&registry, turn.id()),
        vec![
            "turn",
            "turn",
            "instance",
            "instance",
            "definition-first",
            "definition-second",
            "global-first",
            "global-second",
            "global-first",
        ]
    );
    assert_eq!(
        visible(&registry, definition.id()),
        vec![
            "definition-first",
            "definition-second",
            "global-first",
            "global-second",
            "global-first",
        ]
    );
    assert_eq!(
        visible(&registry, sibling.id()),
        vec![
            "sibling",
            "definition-first",
            "definition-second",
            "global-first",
            "global-second",
            "global-first",
        ]
    );
}

#[test]
fn disposing_a_scope_makes_its_subtree_ineligible_before_descendant_effects_settle() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let definition = tree.create_child(&root).unwrap();
    let instance = tree.create_child(&definition).unwrap();
    let turn = tree.create_child(&instance).unwrap();
    let sibling = tree.create_child(&definition).unwrap();
    let registry = ScopedRegistry::new(tree.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_turn, turn_gate) = oneshot::channel();
    let instance_started = started_sender.clone();

    registry.register(None, "global").unwrap();
    registry.register(Some(&definition), "definition").unwrap();
    registry.register(Some(&instance), "instance").unwrap();
    registry.register(Some(&turn), "turn").unwrap();
    registry.register(Some(&sibling), "sibling").unwrap();

    turn.effects()
        .push("turn", move || {
            Box::pin(async move {
                started_sender.send("turn").unwrap();
                turn_gate.await.unwrap();
                Ok(())
            })
        })
        .unwrap();
    let instance_seen = Arc::clone(&seen);
    instance
        .effects()
        .push("instance", move || {
            Box::pin(async move {
                instance_started.send("instance").unwrap();
                instance_seen.lock().push("instance");
                Ok(())
            })
        })
        .unwrap();
    let definition_seen = Arc::clone(&seen);
    definition
        .effects()
        .push("definition", move || {
            Box::pin(async move {
                definition_seen.lock().push("definition");
                Ok(())
            })
        })
        .unwrap();
    let sibling_seen = Arc::clone(&seen);
    sibling
        .effects()
        .push("sibling", move || {
            Box::pin(async move {
                sibling_seen.lock().push("sibling");
                Ok(())
            })
        })
        .unwrap();

    let dispose = thread::spawn({
        let instance = instance.clone();
        move || block_on(instance.dispose())
    });
    assert_eq!(started_receiver.recv().unwrap(), "turn");
    assert!(started_receiver.try_recv().is_err());
    assert!(visible(&registry, instance.id()).is_empty());
    assert!(visible(&registry, turn.id()).is_empty());
    assert_eq!(
        visible(&registry, definition.id()),
        vec!["definition", "global"]
    );
    assert_eq!(
        visible(&registry, sibling.id()),
        vec!["sibling", "definition", "global"]
    );

    release_turn.send(()).unwrap();
    dispose.join().unwrap().unwrap();

    assert_eq!(started_receiver.recv().unwrap(), "instance");
    assert_eq!(&*seen.lock(), &["instance"]);
}

#[test]
fn concurrent_scope_disposals_share_effect_unwind_without_double_running() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let scope = tree.create_child(&root).unwrap();
    let runs = Arc::new(AtomicUsize::new(0));
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = oneshot::channel();
    let effect_runs = Arc::clone(&runs);
    scope
        .effects()
        .push("blocked", move || {
            Box::pin(async move {
                effect_runs.fetch_add(1, Ordering::SeqCst);
                started_sender.send(()).unwrap();
                release_receiver.await.unwrap();
                Ok(())
            })
        })
        .unwrap();

    let scope = Arc::new(scope);
    let start = Arc::new(Barrier::new(3));
    let first = thread::spawn({
        let scope = Arc::clone(&scope);
        let start = Arc::clone(&start);
        move || {
            start.wait();
            block_on(scope.dispose())
        }
    });
    let second = thread::spawn({
        let scope = Arc::clone(&scope);
        let start = Arc::clone(&start);
        move || {
            start.wait();
            block_on(scope.dispose())
        }
    });
    start.wait();

    started_receiver.recv().unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    release_sender.send(()).unwrap();

    first.join().unwrap().unwrap();
    second.join().unwrap().unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert!(!scope.is_active());
}

#[test]
fn handle_removal_wins_while_scope_disposal_is_blocked_before_registration() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let scope = tree.create_child(&root).unwrap();
    let registry = ScopedRegistry::new(tree.clone());
    let drops = Arc::new(AtomicUsize::new(0));
    let registration = registry
        .register(Some(&scope), DropCounter(Arc::clone(&drops)))
        .unwrap();
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = oneshot::channel();

    scope
        .effects()
        .push("gate", move || {
            Box::pin(async move {
                started_sender.send(()).unwrap();
                release_receiver.await.unwrap();
                Ok(())
            })
        })
        .unwrap();

    let dispose = thread::spawn({
        let scope = scope.clone();
        move || block_on(scope.dispose())
    });
    started_receiver.recv().unwrap();
    block_on(registration.dispose()).unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(registry.visible_from(scope.id()).is_empty());

    release_sender.send(()).unwrap();
    dispose.join().unwrap().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn scope_removal_wins_before_handle_when_its_registration_is_newest() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let scope = tree.create_child(&root).unwrap();
    let registry = ScopedRegistry::new(tree.clone());
    let drops = Arc::new(AtomicUsize::new(0));
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = oneshot::channel();

    scope
        .effects()
        .push("gate", move || {
            Box::pin(async move {
                started_sender.send(()).unwrap();
                release_receiver.await.unwrap();
                Ok(())
            })
        })
        .unwrap();
    let registration = registry
        .register(Some(&scope), DropCounter(Arc::clone(&drops)))
        .unwrap();

    let dispose = thread::spawn({
        let scope = scope.clone();
        move || block_on(scope.dispose())
    });
    started_receiver.recv().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    block_on(registration.dispose()).unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(registry.visible_from(scope.id()).is_empty());

    release_sender.send(()).unwrap();
    dispose.join().unwrap().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
