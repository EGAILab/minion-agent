use minion_agent::Service;
use minion_agent::execution::{
    ExecutionWorldIdentity, FileSystem, FileSystemService, IncompatiblePair,
    LocalExecutionProviders, Shell, ShellService, Subprocess, SubprocessService, compatible,
    validate_execution_worlds,
};
use uuid::Uuid;

#[test]
fn compatibility_is_equality_only_and_symmetric() {
    let a = ExecutionWorldIdentity::fresh();
    let b = ExecutionWorldIdentity::fresh();
    assert!(compatible(&a, &a));
    assert_eq!(compatible(&a, &b), compatible(&b, &a));
    assert!(!compatible(&a, &b));
}

#[test]
fn validation_reports_every_incompatible_pair_in_input_order() {
    let a = ExecutionWorldIdentity::fresh();
    let b = ExecutionWorldIdentity::fresh();
    let error = validate_execution_worlds(&[("fs", &a), ("shell", &b), ("subprocess", &b)])
        .expect_err("two pairs must be incompatible");
    assert_eq!(
        error.incompatible_pairs,
        vec![
            IncompatiblePair {
                left: "fs".into(),
                right: "shell".into(),
            },
            IncompatiblePair {
                left: "fs".into(),
                right: "subprocess".into(),
            },
        ]
    );
}

#[test]
fn validation_accepts_equal_worlds_and_ignores_unpassed_providers() {
    let local = ExecutionWorldIdentity::local();
    let unrelated = ExecutionWorldIdentity::fresh();
    validate_execution_worlds(&[("fs", &local), ("shell", &local)]).unwrap();
    validate_execution_worlds(&[("unrelated", &unrelated)]).unwrap();
}

#[test]
fn local_provider_bundle_shares_one_world_and_uses_ctx_service_names() {
    let root = std::env::temp_dir().join(format!("minion-world-test-{}", Uuid::new_v4()));
    let providers = LocalExecutionProviders::new(root);
    assert_eq!(
        providers.fs.execution_world(),
        providers.shell.execution_world()
    );
    assert_eq!(
        providers.shell.execution_world(),
        providers.subprocess.execution_world()
    );
    assert_eq!(FileSystemService::NAME, "fs");
    assert_eq!(ShellService::NAME, "shell");
    assert_eq!(SubprocessService::NAME, "subprocess");
}
