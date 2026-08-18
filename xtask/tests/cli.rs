use std::path::Path;
use std::process::Command;
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .output()
        .expect("xtask binary should run")
}

#[test]
fn help_prints_usage() {
    let output = run(&["help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage: cargo run -p xtask -- <command>"));
}

#[test]
fn accepts_non_conformance_top_level_commands() {
    for command in ["help", "coverage", "layering"] {
        assert!(
            run(&[command]).status.success(),
            "command failed: {command}"
        );
    }
}

#[test]
fn conformance_requires_a_supported_subcommand() {
    assert!(!run(&["conformance"]).status.success());
    assert!(!run(&["conformance", "sync"]).status.success());
    assert!(!run(&["conformance", "unexpected"]).status.success());
}

#[test]
fn missing_command_fails() {
    let output = run(&[]);
    assert!(!output.status.success());
}

#[test]
fn unknown_command_fails() {
    let output = run(&["unknown"]);
    assert!(!output.status.success());
}

#[test]
fn conformance_verify_is_dispatched_instead_of_rejected_as_unsupported() {
    let output = run(&["conformance", "verify"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("read snapshot manifest"));
    assert!(!stderr.contains("usage: xtask conformance"));
}

#[test]
fn conformance_sync_is_dispatched_instead_of_rejected_as_unsupported() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after Unix epoch")
        .as_nanos();
    let source = std::env::temp_dir().join(format!("xtask-cli-source-{nonce}"));
    fs::create_dir_all(source.join("conformance")).expect("source fixture");
    fs::write(source.join("conformance/example.yaml"), b"name: example\n")
        .expect("source fixture file");
    let destination = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("conformance");
    assert!(!destination.exists(), "test must not use a real snapshot");

    let output = run(&["conformance", "sync", "--source", &source.to_string_lossy()]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("git command failed"));
    assert!(!stderr.contains("usage: xtask conformance"));
    assert!(!destination.exists());
    fs::remove_dir_all(source).expect("source cleanup");
}
