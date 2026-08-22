use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use minion_agent::{
    DynPluginSpec, FiberError, FiberHandle, FiberInitContext, FiberState, PluginConfigError,
    PluginInitError, PluginSpec, RuntimeError, ServiceName, ServiceOwner,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::oneshot;

macro_rules! async_test {
    ($name:ident, $body:block) => {
        #[test]
        fn $name() {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .unwrap()
                .block_on(async move $body);
        }
    };
}

#[derive(Debug, Deserialize)]
struct TestConfig {
    value: usize,
}

fn spec<F, Fut>(name: &str, initializer: F) -> DynPluginSpec
where
    F: Fn(FiberInitContext, Arc<TestConfig>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), PluginInitError>> + Send + 'static,
{
    PluginSpec::new(
        name,
        vec![ServiceName::new("tools").unwrap()],
        || json!({ "type": "object", "required": ["value"] }),
        initializer,
    )
    .erase()
}

#[test]
fn config_is_deserialized_and_validated_before_a_fiber_is_created() {
    let initializations = Arc::new(AtomicUsize::new(0));
    let plugin = spec("configured", {
        let initializations = Arc::clone(&initializations);
        move |_ctx, config| {
            initializations.fetch_add(config.value, Ordering::SeqCst);
            async { Ok(()) }
        }
    });

    let error = plugin
        .mount(json!({ "value": "not a number" }), || true)
        .unwrap_err();

    assert!(
        matches!(error, PluginConfigError::Deserialize { plugin, .. } if plugin == "configured")
    );
    assert_eq!(initializations.load(Ordering::SeqCst), 0);
    assert!(plugin.schema().is_object());
}

async_test!(pending_disposes_directly_and_disposed_dispose_is_a_no_op, {
    let fiber = spec("pending", |_ctx, _config| async { Ok(()) })
        .mount(json!({ "value": 1 }), || false)
        .unwrap();

    fiber.reconcile().await.unwrap();
    assert_eq!(fiber.trace(), vec![FiberState::Pending]);

    fiber.dispose().await.unwrap();
    fiber.dispose().await.unwrap();
    assert_eq!(
        fiber.trace(),
        vec![FiberState::Pending, FiberState::Disposed]
    );
});

async_test!(a_pending_disposal_request_prevents_a_racing_load, {
    let initializations = Arc::new(AtomicUsize::new(0));
    let fiber = spec("dispose-before-load", {
        let initializations = Arc::clone(&initializations);
        move |_ctx, _config| {
            initializations.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }
        }
    })
    .mount(json!({ "value": 1 }), || true)
    .unwrap();

    let disposal = fiber.dispose();
    fiber.reconcile().await.unwrap();
    disposal.await.unwrap();

    assert_eq!(initializations.load(Ordering::SeqCst), 0);
    assert_eq!(
        fiber.trace(),
        vec![FiberState::Pending, FiberState::Disposed]
    );
});

async_test!(
    requesting_active_disposal_immediately_closes_service_visibility,
    {
        let fiber = spec("visible", |_ctx, _config| async { Ok(()) })
            .mount(json!({ "value": 1 }), || true)
            .unwrap();
        fiber.reconcile().await.unwrap();
        assert!(fiber.service_owner_is_active());

        let disposal = fiber.dispose();

        assert!(!fiber.service_owner_is_active());
        assert!(matches!(
            fiber
                .service_effect_store()
                .push("too-late", || Box::pin(async { Ok(()) })),
            Err(RuntimeError::InactiveOwner { .. })
        ));
        disposal.await.unwrap();
    }
);

async_test!(
    active_dependency_invalidation_is_sticky_and_immediately_invisible,
    {
        let dependencies = Arc::new(AtomicBool::new(true));
        let fiber = spec("sticky-loss", |_ctx, _config| async { Ok(()) })
            .mount(json!({ "value": 1 }), {
                let dependencies = Arc::clone(&dependencies);
                move || dependencies.load(Ordering::SeqCst)
            })
            .unwrap();
        fiber.reconcile().await.unwrap();

        dependencies.store(false, Ordering::SeqCst);
        fiber.dependencies_changed();
        assert!(!fiber.service_owner_is_active());
        assert!(matches!(
            fiber
                .service_effect_store()
                .push("too-late", || Box::pin(async { Ok(()) })),
            Err(RuntimeError::InactiveOwner { .. })
        ));
        dependencies.store(true, Ordering::SeqCst);
        fiber.dependencies_changed();

        fiber.reconcile().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Pending);
        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Active,
                FiberState::Unloading,
                FiberState::Pending,
            ]
        );
        fiber.reconcile().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Active);
    }
);

async_test!(
    active_dependency_loss_unloads_in_reverse_and_can_load_again,
    {
        let dependencies = Arc::new(AtomicBool::new(true));
        let disposals = Arc::new(Mutex::new(Vec::new()));
        let initializations = Arc::new(AtomicUsize::new(0));
        let fiber = spec("reloadable", {
            let disposals = Arc::clone(&disposals);
            let initializations = Arc::clone(&initializations);
            move |ctx, config| {
                initializations.fetch_add(config.value, Ordering::SeqCst);
                let disposals_for_first = Arc::clone(&disposals);
                ctx.effect("first", move || {
                    Box::pin(async move {
                        disposals_for_first.lock().unwrap().push("first");
                        Ok(())
                    })
                })
                .unwrap();
                let disposals_for_second = Arc::clone(&disposals);
                ctx.effect("second", move || {
                    Box::pin(async move {
                        disposals_for_second.lock().unwrap().push("second");
                        Ok(())
                    })
                })
                .unwrap();
                async { Ok(()) }
            }
        })
        .mount(json!({ "value": 1 }), {
            let dependencies = Arc::clone(&dependencies);
            move || dependencies.load(Ordering::SeqCst)
        })
        .unwrap();

        fiber.reconcile().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Active);
        assert_eq!(
            fiber.trace(),
            vec![FiberState::Pending, FiberState::Loading, FiberState::Active]
        );

        dependencies.store(false, Ordering::SeqCst);
        fiber.dependencies_changed();
        fiber.reconcile().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Pending);
        assert_eq!(*disposals.lock().unwrap(), vec!["second", "first"]);

        dependencies.store(true, Ordering::SeqCst);
        fiber.dependencies_changed();
        fiber.reconcile().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Active);
        assert_eq!(initializations.load(Ordering::SeqCst), 2);

        fiber.dispose().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Disposed);
        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Active,
                FiberState::Unloading,
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Active,
                FiberState::Unloading,
                FiberState::Disposed,
            ]
        );
        assert_eq!(
            *disposals.lock().unwrap(),
            vec!["second", "first", "second", "first"]
        );
    }
);

async_test!(
    initialization_failure_unwinds_then_remains_failed_until_disposal,
    {
        let dependencies = Arc::new(AtomicBool::new(true));
        let initializations = Arc::new(AtomicUsize::new(0));
        let disposals = Arc::new(AtomicUsize::new(0));
        let fiber = spec("failing", {
            let initializations = Arc::clone(&initializations);
            let disposals = Arc::clone(&disposals);
            move |ctx, _config| {
                initializations.fetch_add(1, Ordering::SeqCst);
                let disposals = Arc::clone(&disposals);
                ctx.effect("only_once", move || {
                    Box::pin(async move {
                        disposals.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                })
                .unwrap();
                async { Err(PluginInitError::new("expected failure")) }
            }
        })
        .mount(json!({ "value": 1 }), {
            let dependencies = Arc::clone(&dependencies);
            move || dependencies.load(Ordering::SeqCst)
        })
        .unwrap();

        assert!(matches!(
            fiber.reconcile().await,
            Err(FiberError::Initialization(error)) if error.message() == "expected failure"
        ));
        assert_eq!(
            fiber.trace(),
            vec![FiberState::Pending, FiberState::Loading, FiberState::Failed]
        );
        assert_eq!(disposals.load(Ordering::SeqCst), 1);

        dependencies.store(false, Ordering::SeqCst);
        fiber.dependencies_changed();
        fiber.reconcile().await.unwrap();
        dependencies.store(true, Ordering::SeqCst);
        fiber.dependencies_changed();
        fiber.reconcile().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Failed);
        assert_eq!(initializations.load(Ordering::SeqCst), 1);

        fiber.dispose().await.unwrap();
        assert_eq!(fiber.state(), FiberState::Disposed);
        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Failed,
                FiberState::Disposed,
            ]
        );
        assert_eq!(disposals.load(Ordering::SeqCst), 1);
    }
);

async_test!(
    initialization_and_cleanup_failures_still_settle_failed_and_both_surface,
    {
        let fiber = spec("double-failure", |ctx, _config| {
            ctx.effect("failing-cleanup", || {
                Box::pin(async {
                    Err(minion_agent::DisposeError::new("ignored", "cleanup failed"))
                })
            })
            .unwrap();
            async { Err(PluginInitError::new("initialization failed")) }
        })
        .mount(json!({ "value": 1 }), || true)
        .unwrap();

        let error = fiber.reconcile().await.unwrap_err();

        let FiberError::InitializationAndCleanup {
            initialization,
            cleanup,
        } = error
        else {
            panic!("expected initialization and cleanup aggregate");
        };
        assert_eq!(initialization.message(), "initialization failed");
        assert_eq!(cleanup.as_slice().len(), 1);
        assert_eq!(cleanup.as_slice()[0].label, "failing-cleanup");
        assert_eq!(cleanup.as_slice()[0].message, "cleanup failed");
        assert_eq!(fiber.state(), FiberState::Failed);
        assert_eq!(
            fiber.trace(),
            vec![FiberState::Pending, FiberState::Loading, FiberState::Failed]
        );
    }
);

async_test!(
    dependency_loss_cleanup_failure_still_settles_pending_and_surfaces,
    {
        let dependencies = Arc::new(AtomicBool::new(true));
        let fiber = spec("failed-unload", |ctx, _config| {
            ctx.effect("failing-cleanup", || {
                Box::pin(async {
                    Err(minion_agent::DisposeError::new("ignored", "cleanup failed"))
                })
            })
            .unwrap();
            async { Ok(()) }
        })
        .mount(json!({ "value": 1 }), {
            let dependencies = Arc::clone(&dependencies);
            move || dependencies.load(Ordering::SeqCst)
        })
        .unwrap();
        fiber.reconcile().await.unwrap();

        dependencies.store(false, Ordering::SeqCst);
        let error = fiber.reconcile().await.unwrap_err();

        let FiberError::Cleanup(cleanup) = error else {
            panic!("expected cleanup aggregate");
        };
        assert_eq!(cleanup.as_slice().len(), 1);
        assert_eq!(cleanup.as_slice()[0].label, "failing-cleanup");
        assert_eq!(cleanup.as_slice()[0].message, "cleanup failed");
        assert_eq!(fiber.state(), FiberState::Pending);
        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Active,
                FiberState::Unloading,
                FiberState::Pending,
            ]
        );
    }
);

async_test!(
    dependency_invalidation_wins_over_a_simultaneous_init_error,
    {
        let dependencies = Arc::new(AtomicBool::new(true));
        let mounted = Arc::new(Mutex::new(None::<FiberHandle>));
        let fiber = spec("invalidated-error", {
            let dependencies = Arc::clone(&dependencies);
            let mounted = Arc::clone(&mounted);
            move |_ctx, _config| {
                let dependencies = Arc::clone(&dependencies);
                let mounted = Arc::clone(&mounted);
                async move {
                    dependencies.store(false, Ordering::SeqCst);
                    mounted
                        .lock()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .dependencies_changed();
                    Err(PluginInitError::new("stale error"))
                }
            }
        })
        .mount(json!({ "value": 1 }), {
            let dependencies = Arc::clone(&dependencies);
            move || dependencies.load(Ordering::SeqCst)
        })
        .unwrap();
        *mounted.lock().unwrap() = Some(fiber.clone());

        fiber.reconcile().await.unwrap();

        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Pending
            ]
        );
        assert!(fiber.failure().is_none());
    }
);

async_test!(
    dependency_loss_during_loading_never_transiently_activates,
    {
        let dependencies = Arc::new(AtomicBool::new(true));
        let (entered_tx, entered_rx) = oneshot::channel();
        let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(Mutex::new(Some(release_rx)));
        let fiber = spec("stale", {
            let entered_tx = Arc::clone(&entered_tx);
            let release_rx = Arc::clone(&release_rx);
            move |_ctx, _config| {
                entered_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                let release_rx = release_rx.lock().unwrap().take().unwrap();
                async move {
                    let _ = release_rx.await;
                    Ok(())
                }
            }
        })
        .mount(json!({ "value": 1 }), {
            let dependencies = Arc::clone(&dependencies);
            move || dependencies.load(Ordering::SeqCst)
        })
        .unwrap();

        let loading = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.reconcile().await }
        });
        entered_rx.await.unwrap();
        dependencies.store(false, Ordering::SeqCst);
        fiber.dependencies_changed();
        let _ = release_tx.send(());

        loading.await.unwrap().unwrap();
        assert_eq!(fiber.state(), FiberState::Pending);
        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Pending
            ]
        );
    }
);

async_test!(
    commit_rechecks_dependency_visibility_without_a_notification,
    {
        let dependencies = Arc::new(AtomicBool::new(true));
        let (entered_tx, entered_rx) = oneshot::channel();
        let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(Mutex::new(Some(release_rx)));
        let fiber = spec("commit-check", {
            let entered_tx = Arc::clone(&entered_tx);
            let release_rx = Arc::clone(&release_rx);
            move |_ctx, _config| {
                entered_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                let release_rx = release_rx.lock().unwrap().take().unwrap();
                async move {
                    release_rx.await.unwrap();
                    Ok(())
                }
            }
        })
        .mount(json!({ "value": 1 }), {
            let dependencies = Arc::clone(&dependencies);
            move || dependencies.load(Ordering::SeqCst)
        })
        .unwrap();

        let loading = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.reconcile().await }
        });
        entered_rx.await.unwrap();
        dependencies.store(false, Ordering::SeqCst);
        release_tx.send(()).unwrap();
        loading.await.unwrap().unwrap();

        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Pending
            ]
        );
    }
);

async_test!(
    disposal_during_loading_never_transiently_activates_or_unloads,
    {
        let (entered_tx, entered_rx) = oneshot::channel();
        let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(Mutex::new(Some(release_rx)));
        let fiber = spec("disposed-loading", {
            let entered_tx = Arc::clone(&entered_tx);
            let release_rx = Arc::clone(&release_rx);
            move |_ctx, _config| {
                entered_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                let release_rx = release_rx.lock().unwrap().take().unwrap();
                async move {
                    let _ = release_rx.await;
                    Ok(())
                }
            }
        })
        .mount(json!({ "value": 1 }), || true)
        .unwrap();

        let loading = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.reconcile().await }
        });
        entered_rx.await.unwrap();
        let disposal = fiber.dispose();
        let _ = release_tx.send(());

        disposal.await.unwrap();
        loading.await.unwrap().unwrap();
        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Disposed
            ]
        );
    }
);

async_test!(invalidation_closes_effect_creation_before_unwind_begins, {
    let dependencies = Arc::new(AtomicBool::new(true));
    let first_disposals = Arc::new(AtomicUsize::new(0));
    let (context_tx, context_rx) = oneshot::channel();
    let context_tx = Arc::new(Mutex::new(Some(context_tx)));
    let (unwind_started_tx, unwind_started_rx) = oneshot::channel();
    let unwind_started_tx = Arc::new(Mutex::new(Some(unwind_started_tx)));
    let (unwind_release_tx, unwind_release_rx) = oneshot::channel();
    let unwind_release_rx = Arc::new(Mutex::new(Some(unwind_release_rx)));
    let fiber = spec("effect-race", {
        let first_disposals = Arc::clone(&first_disposals);
        let context_tx = Arc::clone(&context_tx);
        let unwind_started_tx = Arc::clone(&unwind_started_tx);
        let unwind_release_rx = Arc::clone(&unwind_release_rx);
        move |ctx, _config| {
            let first_disposals = Arc::clone(&first_disposals);
            let unwind_started_tx = Arc::clone(&unwind_started_tx);
            let unwind_release_rx = unwind_release_rx.lock().unwrap().take().unwrap();
            ctx.effect("first", move || {
                Box::pin(async move {
                    first_disposals.fetch_add(1, Ordering::SeqCst);
                    unwind_started_tx
                        .lock()
                        .unwrap()
                        .take()
                        .unwrap()
                        .send(())
                        .unwrap();
                    unwind_release_rx.await.unwrap();
                    Ok(())
                })
            })
            .unwrap();
            context_tx
                .lock()
                .unwrap()
                .take()
                .unwrap()
                .send(ctx)
                .unwrap();
            async { std::future::pending::<Result<(), PluginInitError>>().await }
        }
    })
    .mount(json!({ "value": 1 }), {
        let dependencies = Arc::clone(&dependencies);
        move || dependencies.load(Ordering::SeqCst)
    })
    .unwrap();

    let loading = tokio::spawn({
        let fiber = fiber.clone();
        async move { fiber.reconcile().await }
    });
    let stale_context = context_rx.await.unwrap();
    dependencies.store(false, Ordering::SeqCst);
    fiber.dependencies_changed();
    unwind_started_rx.await.unwrap();

    assert!(matches!(
        stale_context.effect("too_late", || Box::pin(async { Ok(()) })),
        Err(RuntimeError::InactiveOwner { .. })
    ));
    unwind_release_tx.send(()).unwrap();
    loading.await.unwrap().unwrap();
    assert_eq!(first_disposals.load(Ordering::SeqCst), 1);
    assert_eq!(fiber.state(), FiberState::Pending);
});

async_test!(transitions_for_one_fiber_are_serialized, {
    let initializations = Arc::new(AtomicUsize::new(0));
    let (entered_tx, entered_rx) = oneshot::channel();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    let (release_tx, release_rx) = oneshot::channel();
    let release_rx = Arc::new(Mutex::new(Some(release_rx)));
    let fiber = spec("serialized", {
        let initializations = Arc::clone(&initializations);
        let entered_tx = Arc::clone(&entered_tx);
        let release_rx = Arc::clone(&release_rx);
        move |_ctx, _config| {
            initializations.fetch_add(1, Ordering::SeqCst);
            entered_tx.lock().unwrap().take().unwrap().send(()).unwrap();
            let release_rx = release_rx.lock().unwrap().take().unwrap();
            async move {
                release_rx.await.unwrap();
                Ok(())
            }
        }
    })
    .mount(json!({ "value": 1 }), || true)
    .unwrap();

    let first = tokio::spawn({
        let fiber = fiber.clone();
        async move { fiber.reconcile().await }
    });
    entered_rx.await.unwrap();
    let second = fiber.reconcile();
    release_tx.send(()).unwrap();
    first.await.unwrap().unwrap();
    second.await.unwrap();

    assert_eq!(initializations.load(Ordering::SeqCst), 1);
    assert_eq!(
        fiber.trace(),
        vec![FiberState::Pending, FiberState::Loading, FiberState::Active]
    );
});

async_test!(
    dispose_joins_an_in_flight_unload_and_observes_its_cleanup_error,
    {
        let dependencies = Arc::new(AtomicBool::new(true));
        let (unwind_started_tx, unwind_started_rx) = oneshot::channel();
        let unwind_started_tx = Arc::new(Mutex::new(Some(unwind_started_tx)));
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(Mutex::new(Some(release_rx)));
        let fiber = spec("joined-unload", {
            let unwind_started_tx = Arc::clone(&unwind_started_tx);
            let release_rx = Arc::clone(&release_rx);
            move |ctx, _config| {
                let unwind_started_tx = Arc::clone(&unwind_started_tx);
                let release_rx = release_rx.lock().unwrap().take().unwrap();
                ctx.effect("fails", move || {
                    Box::pin(async move {
                        unwind_started_tx
                            .lock()
                            .unwrap()
                            .take()
                            .unwrap()
                            .send(())
                            .unwrap();
                        release_rx.await.unwrap();
                        Err(minion_agent::DisposeError::new("ignored", "cleanup failed"))
                    })
                })
                .unwrap();
                async { Ok(()) }
            }
        })
        .mount(json!({ "value": 1 }), {
            let dependencies = Arc::clone(&dependencies);
            move || dependencies.load(Ordering::SeqCst)
        })
        .unwrap();
        fiber.reconcile().await.unwrap();

        dependencies.store(false, Ordering::SeqCst);
        let unloading = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.reconcile().await }
        });
        unwind_started_rx.await.unwrap();
        let disposal = fiber.dispose();
        release_tx.send(()).unwrap();

        let unload_error = unloading.await.unwrap().unwrap_err();
        let dispose_error = disposal.await.unwrap_err();
        for error in [unload_error, dispose_error] {
            let FiberError::Cleanup(errors) = error else {
                panic!("expected cleanup aggregate");
            };
            assert_eq!(errors.as_slice()[0].label, "fails");
        }
        assert_eq!(fiber.state(), FiberState::Disposed);
    }
);

async_test!(
    panics_are_not_converted_into_plugin_initialization_failures,
    {
        let fiber = spec("panicking", |_ctx, _config| async {
            panic!("broken plugin invariant")
        })
        .mount(json!({ "value": 1 }), || true)
        .unwrap();

        let task = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.reconcile().await }
        });
        assert!(task.await.unwrap_err().is_panic());
        assert_ne!(fiber.state(), FiberState::Failed);
    }
);

async_test!(
    a_synchronous_initializer_panic_unwinds_owned_effects_before_propagating,
    {
        let disposals = Arc::new(AtomicUsize::new(0));
        let fiber = spec("sync-panicking", {
            let disposals = Arc::clone(&disposals);
            move |ctx, _config| -> std::future::Ready<Result<(), PluginInitError>> {
                let disposals = Arc::clone(&disposals);
                ctx.effect("before-panic", move || {
                    Box::pin(async move {
                        disposals.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                })
                .unwrap();
                panic!("synchronous plugin invariant")
            }
        })
        .mount(json!({ "value": 1 }), || true)
        .unwrap();

        let task = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.reconcile().await }
        });
        assert!(task.await.unwrap_err().is_panic());
        assert_eq!(disposals.load(Ordering::SeqCst), 1);
        assert_ne!(fiber.state(), FiberState::Failed);
    }
);

async_test!(initializer_panic_does_not_swallow_a_cleanup_failure, {
    let fiber = spec("panic-and-cleanup-failure", |ctx, _config| {
        ctx.effect("failing-cleanup", || {
            Box::pin(async {
                Err(minion_agent::DisposeError::new(
                    "ignored",
                    "cleanup also failed",
                ))
            })
        })
        .unwrap();
        async { panic!("initializer invariant failed") }
    })
    .mount(json!({ "value": 1 }), || true)
    .unwrap();

    let join_error = tokio::spawn({
        let fiber = fiber.clone();
        async move { fiber.reconcile().await }
    })
    .await
    .unwrap_err();
    let payload = join_error.into_panic();
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("fiber panic should retain a diagnostic string");

    assert!(message.contains("initializer invariant failed"));
    assert!(message.contains("cleanup also failed"));
    assert!(message.contains("failing-cleanup"));
    assert_eq!(fiber.state(), FiberState::Disposed);
});

async_test!(
    trace_subscription_carries_the_complete_transition_history,
    {
        let fiber = spec("observed", |_ctx, _config| async { Ok(()) })
            .mount(json!({ "value": 1 }), || true)
            .unwrap();
        let mut trace = fiber.subscribe();

        fiber.reconcile().await.unwrap();
        trace.changed().await.unwrap();
        assert_eq!(
            *trace.borrow_and_update(),
            vec![FiberState::Pending, FiberState::Loading, FiberState::Active]
        );
    }
);

async_test!(
    aborting_reconciliation_during_loading_does_not_abandon_the_transition,
    {
        let disposals = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = oneshot::channel();
        let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
        let fiber = spec("cancelled-loader", {
            let disposals = Arc::clone(&disposals);
            let entered_tx = Arc::clone(&entered_tx);
            move |ctx, _config| {
                let disposals = Arc::clone(&disposals);
                ctx.effect("owned", move || {
                    Box::pin(async move {
                        disposals.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                })
                .unwrap();
                entered_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                async { std::future::pending::<Result<(), PluginInitError>>().await }
            }
        })
        .mount(json!({ "value": 1 }), || true)
        .unwrap();

        let reconcile = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.reconcile().await }
        });
        entered_rx.await.unwrap();
        reconcile.abort();
        assert!(reconcile.await.unwrap_err().is_cancelled());

        fiber.dispose().await.unwrap();

        assert_eq!(disposals.load(Ordering::SeqCst), 1);
        assert_eq!(
            fiber.trace(),
            vec![
                FiberState::Pending,
                FiberState::Loading,
                FiberState::Disposed
            ]
        );
    }
);

async_test!(
    aborting_disposal_during_unwind_leaves_a_joinable_shared_transition,
    {
        let calls = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = oneshot::channel();
        let started_tx = Arc::new(Mutex::new(Some(started_tx)));
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(Mutex::new(Some(release_rx)));
        let fiber = spec("cancelled-disposer", {
            let calls = Arc::clone(&calls);
            let started_tx = Arc::clone(&started_tx);
            let release_rx = Arc::clone(&release_rx);
            move |ctx, _config| {
                let calls = Arc::clone(&calls);
                let started_tx = Arc::clone(&started_tx);
                let release_rx = release_rx.lock().unwrap().take().unwrap();
                ctx.effect("blocked", move || {
                    Box::pin(async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        started_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                        release_rx.await.unwrap();
                        Err(minion_agent::DisposeError::new("ignored", "failed"))
                    })
                })
                .unwrap();
                async { Ok(()) }
            }
        })
        .mount(json!({ "value": 1 }), || true)
        .unwrap();
        fiber.reconcile().await.unwrap();

        let first = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.dispose().await }
        });
        started_rx.await.unwrap();
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        let joined = fiber.dispose();
        release_tx.send(()).unwrap();

        let FiberError::Cleanup(errors) = joined.await.unwrap_err() else {
            panic!("expected cleanup aggregate");
        };
        assert_eq!(errors.as_slice()[0].label, "blocked");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(fiber.state(), FiberState::Disposed);
    }
);

async_test!(
    dropping_an_unpolled_disposal_future_leaves_a_sticky_drivable_request,
    {
        let disposals = Arc::new(AtomicUsize::new(0));
        let fiber = spec("dropped-dispose", {
            let disposals = Arc::clone(&disposals);
            move |ctx, _config| {
                let disposals = Arc::clone(&disposals);
                ctx.effect("owned", move || {
                    Box::pin(async move {
                        disposals.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                })
                .unwrap();
                async { Ok(()) }
            }
        })
        .mount(json!({ "value": 1 }), || true)
        .unwrap();
        fiber.reconcile().await.unwrap();

        drop(fiber.dispose());
        assert!(!fiber.service_owner_is_active());
        fiber.reconcile().await.unwrap();

        assert_eq!(fiber.state(), FiberState::Disposed);
        assert_eq!(disposals.load(Ordering::SeqCst), 1);
    }
);

async_test!(
    a_late_dispose_joiner_gets_failures_from_already_completed_reverse_cleanup,
    {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (second_done_tx, second_done_rx) = oneshot::channel();
        let second_done_tx = Arc::new(Mutex::new(Some(second_done_tx)));
        let (first_started_tx, first_started_rx) = oneshot::channel();
        let first_started_tx = Arc::new(Mutex::new(Some(first_started_tx)));
        let (release_first_tx, release_first_rx) = oneshot::channel();
        let release_first_rx = Arc::new(Mutex::new(Some(release_first_rx)));
        let fiber = spec("late-join", {
            let calls = Arc::clone(&calls);
            let second_done_tx = Arc::clone(&second_done_tx);
            let first_started_tx = Arc::clone(&first_started_tx);
            let release_first_rx = Arc::clone(&release_first_rx);
            move |ctx, _config| {
                let first_calls = Arc::clone(&calls);
                let first_started_tx = Arc::clone(&first_started_tx);
                let release_first_rx = release_first_rx.lock().unwrap().take().unwrap();
                ctx.effect("first", move || {
                    Box::pin(async move {
                        first_calls.lock().unwrap().push("first");
                        first_started_tx
                            .lock()
                            .unwrap()
                            .take()
                            .unwrap()
                            .send(())
                            .unwrap();
                        release_first_rx.await.unwrap();
                        Err(minion_agent::DisposeError::new("ignored", "first failed"))
                    })
                })
                .unwrap();
                let second_calls = Arc::clone(&calls);
                let second_done_tx = Arc::clone(&second_done_tx);
                ctx.effect("second", move || {
                    Box::pin(async move {
                        second_calls.lock().unwrap().push("second");
                        second_done_tx
                            .lock()
                            .unwrap()
                            .take()
                            .unwrap()
                            .send(())
                            .unwrap();
                        Err(minion_agent::DisposeError::new("ignored", "second failed"))
                    })
                })
                .unwrap();
                async { Ok(()) }
            }
        })
        .mount(json!({ "value": 1 }), || true)
        .unwrap();
        fiber.reconcile().await.unwrap();

        let first_waiter = tokio::spawn({
            let fiber = fiber.clone();
            async move { fiber.dispose().await }
        });
        second_done_rx.await.unwrap();
        first_started_rx.await.unwrap();
        let mut late_waiter = Box::pin(fiber.dispose());
        assert!(matches!(
            futures::poll!(late_waiter.as_mut()),
            std::task::Poll::Pending
        ));
        release_first_tx.send(()).unwrap();

        for result in [first_waiter.await.unwrap(), late_waiter.await] {
            let FiberError::Cleanup(errors) = result.unwrap_err() else {
                panic!("expected cleanup aggregate");
            };
            assert_eq!(
                errors
                    .as_slice()
                    .iter()
                    .map(|error| error.label.as_str())
                    .collect::<Vec<_>>(),
                vec!["second", "first"]
            );
        }
        assert_eq!(*calls.lock().unwrap(), vec!["second", "first"]);
    }
);
