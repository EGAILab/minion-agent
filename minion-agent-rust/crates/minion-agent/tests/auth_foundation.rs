use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use minion_agent::auth::*;
use parking_lot::{Mutex, RwLock};
use serde_json::json;
use tokio::sync::Notify;

fn oauth(access: &str, expires: f64) -> OAuthCredential {
    OAuthCredential::new(
        access.into(),
        "refresh".into(),
        expires,
        Arc::new(RwLock::new(BTreeMap::new())),
    )
}

async fn seed(store: &InMemoryCredentialStore, provider: &str, credential: Credential) {
    store
        .modify(
            provider.to_owned(),
            move |_| async move { Ok::<_, ()>(Some(credential)) },
            AuthOperationOptions::default(),
        )
        .await
        .unwrap();
}

#[test]
fn credential_handles_preserve_live_scalar_and_nested_aliasing() {
    let env = Arc::new(RwLock::new(BTreeMap::from([("A".into(), "1".into())])));
    let api = ApiKeyCredential::new(Some("old".into()), Some(env.clone()));
    api.set_key(Some("new".into()));
    env.write().insert("B".into(), "2".into());
    assert_eq!(api.key().as_deref(), Some("new"));
    assert_eq!(
        api.env().unwrap().read().get("B").map(String::as_str),
        Some("2")
    );

    let extra = Arc::new(RwLock::new(BTreeMap::from([(
        "nested".into(),
        json!({"a": [1]}),
    )])));
    let oauth = OAuthCredential::new("a".into(), "r".into(), 1.0, extra.clone());
    oauth.set_access("b");
    extra.write().insert("new".into(), json!([1, {"x": true}]));
    assert_eq!(oauth.access(), "b");
    assert_eq!(
        oauth.extra().read().get("new"),
        Some(&json!([1, {"x": true}]))
    );
}

#[tokio::test]
async fn store_preserves_insertion_order_and_stale_absence() {
    let store = InMemoryCredentialStore::new();
    seed(&store, "b", Credential::OAuth(oauth("b", 1.0))).await;
    seed(&store, "a", Credential::OAuth(oauth("a", 1.0))).await;
    assert_eq!(
        store
            .list(Default::default())
            .await
            .unwrap()
            .into_iter()
            .map(|i| i.provider_id)
            .collect::<Vec<_>>(),
        ["b", "a"]
    );
    store.delete("b", Default::default()).await.unwrap();
    seed(&store, "b", Credential::OAuth(oauth("b2", 1.0))).await;
    assert_eq!(
        store
            .list(Default::default())
            .await
            .unwrap()
            .into_iter()
            .map(|i| i.provider_id)
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert!(
        store
            .read("missing", Default::default())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn store_reads_expose_the_same_live_credential_and_none_does_not_delete() {
    let store = InMemoryCredentialStore::new();
    let value = oauth("before", 1.0);
    seed(&store, "p", Credential::OAuth(value.clone())).await;
    let read = store
        .read("p", Default::default())
        .await
        .unwrap()
        .unwrap()
        .as_oauth()
        .unwrap();
    read.set_access("mutated");
    store
        .modify(
            "p",
            |_| async { Ok::<Option<Credential>, ()>(None) },
            Default::default(),
        )
        .await
        .unwrap();
    assert_eq!(value.access(), "mutated");
    assert_eq!(
        store
            .read("p", Default::default())
            .await
            .unwrap()
            .unwrap()
            .as_oauth()
            .unwrap()
            .access(),
        "mutated"
    );
}

#[tokio::test]
async fn store_serializes_same_provider_and_failure_does_not_sour_chain() {
    let store = Arc::new(InMemoryCredentialStore::new());
    seed(&store, "p", Credential::OAuth(oauth("initial", 1.0))).await;
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let first = {
        let store = store.clone();
        let entered = entered.clone();
        let release = release.clone();
        tokio::spawn(async move {
            store
                .modify(
                    "p",
                    move |_| async move {
                        entered.notify_one();
                        release.notified().await;
                        Ok::<_, &'static str>(Some(Credential::OAuth(oauth("one", 1.0))))
                    },
                    Default::default(),
                )
                .await
        })
    };
    entered.notified().await;
    let second = {
        let store = store.clone();
        tokio::spawn(async move {
            store
                .modify(
                    "p",
                    move |current| async move {
                        assert_eq!(current.unwrap().as_oauth().unwrap().access(), "one");
                        Err::<Option<Credential>, _>("boom")
                    },
                    Default::default(),
                )
                .await
        })
    };
    release.notify_one();
    first.await.unwrap().unwrap();
    assert!(matches!(
        second.await.unwrap(),
        Err(ModifyError::Callback("boom"))
    ));
    assert_eq!(
        store
            .read("p", Default::default())
            .await
            .unwrap()
            .unwrap()
            .as_oauth()
            .unwrap()
            .access(),
        "one"
    );
}

#[tokio::test]
async fn cancellation_returns_promptly_and_discards_late_modify_result() {
    let store = Arc::new(InMemoryCredentialStore::new());
    seed(&store, "p", Credential::OAuth(oauth("old", 1.0))).await;
    let controller = AuthAbortController::default();
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let operation = {
        let store = store.clone();
        let entered = entered.clone();
        let release = release.clone();
        let options = AuthOperationOptions {
            signal: Some(controller.signal()),
        };
        tokio::spawn(async move {
            store
                .modify(
                    "p",
                    move |_| async move {
                        entered.notify_one();
                        release.notified().await;
                        Ok::<_, ()>(Some(Credential::OAuth(oauth("late", 1.0))))
                    },
                    options,
                )
                .await
        })
    };
    entered.notified().await;
    controller.abort();
    assert!(matches!(
        operation.await.unwrap(),
        Err(ModifyError::Store(CredentialStoreError::Cancelled))
    ));
    release.notify_one();
    tokio::task::yield_now().await;
    assert_eq!(
        store
            .read("p", Default::default())
            .await
            .unwrap()
            .unwrap()
            .as_oauth()
            .unwrap()
            .access(),
        "old"
    );
}

#[tokio::test]
async fn queued_abort_prevents_the_queued_callback_from_running() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let first = {
        let store = store.clone();
        let entered = entered.clone();
        let release = release.clone();
        tokio::spawn(async move {
            store
                .modify(
                    "p",
                    move |_| async move {
                        entered.notify_one();
                        release.notified().await;
                        Ok::<_, ()>(Some(Credential::OAuth(oauth("first", 1.0))))
                    },
                    Default::default(),
                )
                .await
        })
    };
    entered.notified().await;
    let controller = AuthAbortController::default();
    let ran = Arc::new(AtomicUsize::new(0));
    let second = {
        let store = store.clone();
        let ran = ran.clone();
        let options = AuthOperationOptions {
            signal: Some(controller.signal()),
        };
        tokio::spawn(async move {
            store
                .modify(
                    "p",
                    move |_| async move {
                        ran.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, ()>(None)
                    },
                    options,
                )
                .await
        })
    };
    tokio::task::yield_now().await;
    controller.abort();
    assert!(matches!(
        second.await.unwrap(),
        Err(ModifyError::Store(CredentialStoreError::Cancelled))
    ));
    release.notify_one();
    first.await.unwrap().unwrap();
    tokio::task::yield_now().await;
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn refresh_double_checks_and_refreshes_once() {
    let store = Arc::new(InMemoryCredentialStore::new());
    seed(&store, "p", Credential::OAuth(oauth("old", 100.0))).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let refresh: RefreshOperation = {
        let calls = calls.clone();
        Arc::new(move |_credential, signal| {
            let calls = calls.clone();
            Box::pin(async move {
                assert!(!signal.aborted());
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(oauth("new", 1_000_000.0))
            })
        })
    };
    let left = refresh_if_expiring_at(
        store.as_ref(),
        "p",
        refresh.clone(),
        None,
        Default::default(),
        || 0.0,
    );
    let right = refresh_if_expiring_at(
        store.as_ref(),
        "p",
        refresh,
        None,
        Default::default(),
        || 0.0,
    );
    let (left, right) = tokio::join!(left, right);
    assert_eq!(left.unwrap().unwrap().access(), "new");
    assert_eq!(right.unwrap().unwrap().access(), "new");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn refresh_nan_suppresses_work_and_combined_signal_honors_deadline() {
    let store = InMemoryCredentialStore::new();
    seed(&store, "p", Credential::OAuth(oauth("old", 0.0))).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let refresh: RefreshOperation = {
        let calls = calls.clone();
        Arc::new(move |_, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(oauth("new", 1.0)) })
        })
    };
    assert_eq!(
        refresh_if_expiring_at(
            &store,
            "p",
            refresh,
            Some(f64::NAN),
            Default::default(),
            || 0.0
        )
        .await
        .unwrap()
        .unwrap()
        .access(),
        "old"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        CombinedSignal::with_deadline(None, Instant::now() - Duration::from_millis(1)).aborted()
    );
}

#[tokio::test]
async fn refresh_post_validation_uses_the_effective_default_threshold() {
    let store = InMemoryCredentialStore::new();
    seed(&store, "p", Credential::OAuth(oauth("old", 0.0))).await;
    let refresh: RefreshOperation =
        Arc::new(|_, _| Box::pin(async { Ok(oauth("still-soon", 100_000.0)) }));
    assert!(matches!(
        refresh_if_expiring_at(
            &store,
            "p",
            refresh,
            Some(60_000.0),
            Default::default(),
            || 0.0
        )
        .await,
        Err(AuthorityError::OAuthRefresh { .. })
    ));
}

#[test]
fn pkce_matches_rfc_vector_and_generation_shape() {
    let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
    assert_eq!(
        derive_pkce_challenge(verifier),
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    );
    let pair = generate_pkce().unwrap();
    assert_eq!(pair.verifier.len(), 43);
    assert_eq!(pair.challenge, derive_pkce_challenge(&pair.verifier));
}

#[derive(Default)]
struct FakeClock {
    nanos: Mutex<u64>,
    sleeps: Mutex<Vec<f64>>,
}

#[async_trait::async_trait]
impl DeviceClock for FakeClock {
    fn elapsed_seconds(&self) -> f64 {
        *self.nanos.lock() as f64 / 1_000_000_000.0
    }
    async fn sleep_seconds(&self, seconds: f64) {
        self.sleeps.lock().push(seconds);
        *self.nanos.lock() += (seconds * 1_000_000_000.0).round() as u64;
    }
}

#[tokio::test]
async fn device_polling_uses_server_interval_and_terminal_outcomes() {
    let clock = FakeClock::default();
    let mut outcomes = std::collections::VecDeque::from([
        DevicePollResult::SlowDown {
            interval_seconds: Some(1.2349),
        },
        DevicePollResult::Pending,
        DevicePollResult::Complete("token"),
    ]);
    let result = poll_device_code_flow(
        || std::future::ready(outcomes.pop_front().unwrap()),
        DeviceFlowOptions {
            interval_seconds: Some(5.0),
            ..Default::default()
        },
        &clock,
    )
    .await
    .unwrap();
    assert_eq!(result, "token");
    assert!((clock.elapsed_seconds() - 2.468).abs() < 1e-9);
}

#[tokio::test]
async fn abortable_sleep_models_settimeout_boundaries() {
    for (input, expected) in [
        (f64::NAN, 0.001),
        (f64::INFINITY, 0.001),
        (f64::NEG_INFINITY, 0.001),
        (0.0, 0.001),
        (-1.0, 0.001),
        (0.0019, 0.001),
    ] {
        let clock = FakeClock::default();
        abortable_sleep(input, None, &clock).await.unwrap();
        assert!((clock.elapsed_seconds() - expected).abs() < 1e-9, "{input}");
    }
}

#[tokio::test]
async fn poll_loop_normalizes_negative_infinity_before_the_timer_boundary() {
    let clock = FakeClock::default();
    let mut outcomes = std::collections::VecDeque::from([
        DevicePollResult::Pending,
        DevicePollResult::Complete("done"),
    ]);
    let value = poll_device_code_flow(
        || std::future::ready(outcomes.pop_front().unwrap()),
        DeviceFlowOptions {
            interval_seconds: Some(f64::NEG_INFINITY),
            ..Default::default()
        },
        &clock,
    )
    .await
    .unwrap();
    assert_eq!(value, "done");
    assert!((clock.elapsed_seconds() - 1.0).abs() < 1e-9);
}
