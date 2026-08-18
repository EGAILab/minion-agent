use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use futures::executor::block_on;
use minion_agent::{
    EffectStore, RuntimeError, ScopeTree, Service, ServiceCheck, ServiceOwner, ServiceRegistry,
};

#[derive(Debug, Eq, PartialEq)]
struct ToolsA(&'static str);

impl Service for ToolsA {
    const NAME: &'static str = "tools";
}

#[derive(Debug)]
struct ToolsB;

impl Service for ToolsB {
    const NAME: &'static str = "tools";
}

struct ReentrantLosingTools {
    effects: Arc<EffectStore>,
    dropped_sender: mpsc::Sender<()>,
}

impl Service for ReentrantLosingTools {
    const NAME: &'static str = "tools";
}

impl Drop for ReentrantLosingTools {
    fn drop(&mut self) {
        self.effects
            .push("registered while rejected value drops", || {
                Box::pin(async { Ok(()) })
            })
            .unwrap();
        self.dropped_sender.send(()).unwrap();
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Models;

impl Service for Models {
    const NAME: &'static str = "models";
}

struct FakeOwner {
    name: String,
    active: AtomicBool,
    effects: Arc<EffectStore>,
}

impl FakeOwner {
    fn new(name: impl Into<String>, active: bool) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            active: AtomicBool::new(active),
            effects: Arc::new(EffectStore::new()),
        })
    }

    fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::SeqCst);
    }
}

impl ServiceOwner for FakeOwner {
    fn service_owner_name(&self) -> String {
        self.name.clone()
    }

    fn service_owner_is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    fn service_effect_store(&self) -> Arc<EffectStore> {
        Arc::clone(&self.effects)
    }
}

#[test]
fn active_provider_resolves_through_the_typed_name_key() {
    let registry = ServiceRegistry::new();
    let owner = FakeOwner::new("provider", true);

    let _registration = registry
        .provide::<ToolsA>(owner, Arc::new(ToolsA("primary")), None)
        .unwrap();

    assert_eq!(*registry.require::<ToolsA>().unwrap(), ToolsA("primary"));
}

#[test]
fn loading_and_unloading_owners_are_invisible_but_still_occupy_the_slot() {
    let registry = ServiceRegistry::new();
    let owner = FakeOwner::new("loading-provider", false);
    let contender = FakeOwner::new("contender", true);
    let _registration = registry
        .provide::<ToolsA>(owner.clone(), Arc::new(ToolsA("value")), None)
        .unwrap();

    assert!(matches!(
        registry.require::<ToolsA>(),
        Err(RuntimeError::ServiceUnavailable { name }) if name.as_str() == "tools"
    ));
    assert!(matches!(
        registry.provide::<ToolsA>(contender, Arc::new(ToolsA("late")), None),
        Err(RuntimeError::ServiceConflict { name, holder })
            if name.as_str() == "tools" && holder == "loading-provider"
    ));

    owner.set_active(true);
    assert_eq!(*registry.require::<ToolsA>().unwrap(), ToolsA("value"));

    owner.set_active(false);
    assert!(matches!(
        registry.require::<ToolsA>(),
        Err(RuntimeError::ServiceUnavailable { .. })
    ));
}

#[test]
fn optional_check_predicate_narrows_visibility_including_scope_eligibility() {
    let registry = ServiceRegistry::new();
    let owner = FakeOwner::new("provider", true);
    let tree = ScopeTree::new();
    let scope = tree.create_root();
    let check: ServiceCheck = Arc::new({
        let scope = scope.clone();
        move || scope.is_active()
    });
    let _registration = registry
        .provide::<ToolsA>(owner, Arc::new(ToolsA("scoped")), Some(check))
        .unwrap();

    assert_eq!(*registry.require::<ToolsA>().unwrap(), ToolsA("scoped"));
    block_on(scope.dispose()).unwrap();
    assert!(matches!(
        registry.require::<ToolsA>(),
        Err(RuntimeError::ServiceUnavailable { .. })
    ));
}

#[test]
fn visibility_check_may_reenter_the_registry_without_a_global_lock() {
    let registry = ServiceRegistry::default();
    let _models = registry
        .provide::<Models>(FakeOwner::new("models", true), Arc::new(Models), None)
        .unwrap();
    let check: ServiceCheck = Arc::new({
        let registry = registry.clone();
        move || registry.require::<Models>().is_ok()
    });
    let _tools = registry
        .provide::<ToolsA>(
            FakeOwner::new("tools", true),
            Arc::new(ToolsA("checked")),
            Some(check),
        )
        .unwrap();

    assert_eq!(*registry.require::<ToolsA>().unwrap(), ToolsA("checked"));
}

#[test]
fn removing_the_current_provider_never_reveals_a_fallback() {
    let registry = ServiceRegistry::new();
    let first = registry
        .provide::<ToolsA>(
            FakeOwner::new("first", true),
            Arc::new(ToolsA("first")),
            None,
        )
        .unwrap();
    assert!(matches!(
        registry.provide::<ToolsA>(
            FakeOwner::new("rejected", true),
            Arc::new(ToolsA("rejected")),
            None,
        ),
        Err(RuntimeError::ServiceConflict { .. })
    ));
    block_on(first.dispose()).unwrap();

    assert!(matches!(
        registry.require::<ToolsA>(),
        Err(RuntimeError::ServiceUnavailable { .. })
    ));

    let second = registry
        .provide::<ToolsA>(
            FakeOwner::new("second", true),
            Arc::new(ToolsA("second")),
            None,
        )
        .unwrap();
    assert_eq!(*registry.require::<ToolsA>().unwrap(), ToolsA("second"));
    block_on(second.dispose()).unwrap();

    assert!(matches!(
        registry.require::<ToolsA>(),
        Err(RuntimeError::ServiceUnavailable { .. })
    ));
}

#[test]
fn service_name_type_contract_survives_temporary_unload() {
    let registry = ServiceRegistry::new();
    let registration = registry
        .provide::<ToolsA>(
            FakeOwner::new("first", true),
            Arc::new(ToolsA("first")),
            None,
        )
        .unwrap();
    block_on(registration.dispose()).unwrap();

    assert!(matches!(
        registry.provide::<ToolsB>(FakeOwner::new("other", true), Arc::new(ToolsB), None),
        Err(RuntimeError::ServiceTypeMismatch {
            name,
            expected,
            actual,
        }) if name.as_str() == "tools"
            && expected.contains("ToolsA")
            && actual.contains("ToolsB")
    ));
    assert!(matches!(
        registry.require::<ToolsB>(),
        Err(RuntimeError::ServiceTypeMismatch { .. })
    ));
}

#[test]
fn closing_the_owner_before_effect_acceptance_never_publishes_the_service() {
    let registry = ServiceRegistry::new();
    let (reached_sender, reached_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = mpsc::channel();
    let owner = Arc::new(BlockingOwner {
        effects: Arc::new(EffectStore::new()),
        reached_sender,
        release_receiver: std::sync::Mutex::new(release_receiver),
    });

    let provide = thread::spawn({
        let registry = registry.clone();
        let owner = owner.clone();
        move || registry.provide::<ToolsA>(owner, Arc::new(ToolsA("late")), None)
    });
    reached_receiver.recv().unwrap();
    block_on(owner.effects.close_and_dispose()).unwrap();
    release_sender.send(()).unwrap();

    assert!(matches!(
        provide.join().unwrap(),
        Err(RuntimeError::InactiveOwner { .. })
    ));
    assert!(matches!(
        registry.require::<ToolsA>(),
        Err(RuntimeError::ServiceUnavailable { .. })
    ));
}

#[test]
fn concurrent_providers_that_both_pass_preflight_still_commit_one_exclusive_slot() {
    let registry = ServiceRegistry::new();
    let (first_reached_sender, first_reached_receiver) = mpsc::channel();
    let (release_first, first_release_receiver) = mpsc::channel();
    let first_owner = Arc::new(BlockingOwner {
        effects: Arc::new(EffectStore::new()),
        reached_sender: first_reached_sender,
        release_receiver: std::sync::Mutex::new(first_release_receiver),
    });
    let (second_reached_sender, second_reached_receiver) = mpsc::channel();
    let (release_second, second_release_receiver) = mpsc::channel();
    let second_owner = Arc::new(BlockingOwner {
        effects: Arc::new(EffectStore::new()),
        reached_sender: second_reached_sender,
        release_receiver: std::sync::Mutex::new(second_release_receiver),
    });

    let first = thread::spawn({
        let registry = registry.clone();
        let owner = first_owner.clone();
        move || registry.provide::<ToolsA>(owner, Arc::new(ToolsA("first")), None)
    });
    let second = thread::spawn({
        let registry = registry.clone();
        let owner = second_owner.clone();
        move || registry.provide::<ToolsA>(owner, Arc::new(ToolsA("second")), None)
    });
    first_reached_receiver.recv().unwrap();
    second_reached_receiver.recv().unwrap();

    release_first.send(()).unwrap();
    let first_registration = first.join().unwrap().unwrap();
    release_second.send(()).unwrap();
    assert!(matches!(
        second.join().unwrap(),
        Err(RuntimeError::ServiceConflict { .. })
    ));
    assert_eq!(*registry.require::<ToolsA>().unwrap(), ToolsA("first"));

    block_on(second_owner.effects.close_and_dispose()).unwrap();
    assert_eq!(*registry.require::<ToolsA>().unwrap(), ToolsA("first"));
    block_on(first_registration.dispose()).unwrap();
}

#[test]
fn rejected_concurrent_provider_drops_user_value_after_releasing_its_effect_lock() {
    let registry = ServiceRegistry::new();
    let (winner_reached_sender, winner_reached_receiver) = mpsc::channel();
    let (release_winner, winner_release_receiver) = mpsc::channel();
    let winner_owner = Arc::new(BlockingOwner {
        effects: Arc::new(EffectStore::new()),
        reached_sender: winner_reached_sender,
        release_receiver: std::sync::Mutex::new(winner_release_receiver),
    });
    let (loser_reached_sender, loser_reached_receiver) = mpsc::channel();
    let (release_loser, loser_release_receiver) = mpsc::channel();
    let loser_owner = Arc::new(BlockingOwner {
        effects: Arc::new(EffectStore::new()),
        reached_sender: loser_reached_sender,
        release_receiver: std::sync::Mutex::new(loser_release_receiver),
    });
    let (dropped_sender, dropped_receiver) = mpsc::channel();

    let winner = thread::spawn({
        let registry = registry.clone();
        let owner = winner_owner.clone();
        move || registry.provide::<ToolsA>(owner, Arc::new(ToolsA("winner")), None)
    });
    let loser = thread::spawn({
        let registry = registry.clone();
        let owner = loser_owner.clone();
        let effects = Arc::clone(&loser_owner.effects);
        move || {
            registry.provide::<ReentrantLosingTools>(
                owner,
                Arc::new(ReentrantLosingTools {
                    effects,
                    dropped_sender,
                }),
                None,
            )
        }
    });
    winner_reached_receiver.recv().unwrap();
    loser_reached_receiver.recv().unwrap();

    release_winner.send(()).unwrap();
    let winner_registration = winner.join().unwrap().unwrap();
    release_loser.send(()).unwrap();
    dropped_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("rejected service Drop must not run under its owner's effect lock");
    assert!(matches!(
        loser.join().unwrap(),
        Err(RuntimeError::ServiceTypeMismatch { .. })
    ));

    block_on(winner_registration.dispose()).unwrap();
    block_on(loser_owner.effects.close_and_dispose()).unwrap();
}

struct BlockingOwner {
    effects: Arc<EffectStore>,
    reached_sender: mpsc::Sender<()>,
    release_receiver: std::sync::Mutex<mpsc::Receiver<()>>,
}

impl ServiceOwner for BlockingOwner {
    fn service_owner_name(&self) -> String {
        "blocked".to_owned()
    }

    fn service_owner_is_active(&self) -> bool {
        true
    }

    fn service_effect_store(&self) -> Arc<EffectStore> {
        self.reached_sender.send(()).unwrap();
        self.release_receiver.lock().unwrap().recv().unwrap();
        Arc::clone(&self.effects)
    }
}

#[test]
fn owner_effect_unwind_removes_the_registration_exactly_once() {
    let registry = ServiceRegistry::new();
    let owner = FakeOwner::new("provider", true);
    let registration = registry
        .provide::<ToolsA>(owner.clone(), Arc::new(ToolsA("owned")), None)
        .unwrap();

    block_on(owner.effects.close_and_dispose()).unwrap();
    assert!(matches!(
        registry.require::<ToolsA>(),
        Err(RuntimeError::ServiceUnavailable { .. })
    ));
    block_on(registration.dispose()).unwrap();
}

#[test]
fn registration_disposal_is_safe_after_the_registry_is_dropped() {
    let registration = {
        let registry = ServiceRegistry::new();
        registry
            .provide::<ToolsA>(
                FakeOwner::new("provider", true),
                Arc::new(ToolsA("owned")),
                None,
            )
            .unwrap()
    };

    block_on(registration.dispose()).unwrap();
}

#[test]
fn service_value_destructor_may_reenter_registry_after_removal() {
    let registry = ServiceRegistry::new();
    let (dropped_sender, dropped_receiver) = mpsc::channel();
    let value = Arc::new(ReentrantService {
        registry: registry.clone(),
        dropped_sender,
    });
    let registration = registry
        .provide::<ReentrantService>(FakeOwner::new("provider", true), value, None)
        .unwrap();

    let dispose = thread::spawn(move || block_on(registration.dispose()));
    dropped_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("service destructor should not deadlock while reentering the registry");
    dispose.join().unwrap().unwrap();
}

struct ReentrantService {
    registry: ServiceRegistry,
    dropped_sender: mpsc::Sender<()>,
}

impl Service for ReentrantService {
    const NAME: &'static str = "reentrant";
}

impl Drop for ReentrantService {
    fn drop(&mut self) {
        assert!(matches!(
            self.registry.require::<Self>(),
            Err(RuntimeError::ServiceUnavailable { .. })
        ));
        self.dropped_sender.send(()).unwrap();
    }
}
