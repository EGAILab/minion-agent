use std::{
    sync::{Arc, mpsc},
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

    registry.register(None, "global").unwrap();
    registry.register(Some(&definition), "definition").unwrap();
    registry.register(Some(&instance), "instance").unwrap();
    registry.register(Some(&turn), "turn").unwrap();
    registry.register(Some(&sibling), "sibling").unwrap();

    assert!(tree.is_ancestor(definition.id(), turn.id()));
    assert!(!tree.is_ancestor(turn.id(), definition.id()));
    assert!(!tree.is_ancestor(sibling.id(), turn.id()));
    assert_eq!(
        visible(&registry, turn.id()),
        vec!["turn", "instance", "definition", "global"]
    );
    assert_eq!(
        visible(&registry, definition.id()),
        vec!["definition", "global"]
    );
    assert_eq!(
        visible(&registry, sibling.id()),
        vec!["sibling", "definition", "global"]
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

    let dispose = thread::spawn(move || block_on(instance.dispose()));
    assert_eq!(started_receiver.recv().unwrap(), "turn");
    assert!(started_receiver.try_recv().is_err());
    assert_eq!(visible(&registry, definition.id()), vec!["definition"]);
    assert_eq!(
        visible(&registry, sibling.id()),
        vec!["sibling", "definition"]
    );

    release_turn.send(()).unwrap();
    dispose.join().unwrap().unwrap();

    assert_eq!(started_receiver.recv().unwrap(), "instance");
    assert_eq!(&*seen.lock(), &["instance"]);
}

#[test]
fn scoped_registration_is_removed_exactly_once_when_scope_and_handle_race() {
    let tree = ScopeTree::new();
    let root = tree.create_root();
    let scope = tree.create_child(&root).unwrap();
    let registry = ScopedRegistry::new(tree.clone());
    let registration = registry.register(Some(&scope), "owned").unwrap();

    block_on(scope.dispose()).unwrap();
    block_on(registration.dispose()).unwrap();

    assert!(visible(&registry, root.id()).is_empty());
}
