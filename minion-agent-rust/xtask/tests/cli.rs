use std::process::Command;

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
    assert!(
        !run(&[
            "conformance",
            "sync",
            "--source",
            "retired-sibling-checkout",
        ])
        .status
        .success()
    );
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
    assert!(
        output.status.success(),
        "root canonical contract should verify: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
