use minion_agent::VERSION;

#[test]
fn package_exposes_a_non_empty_version() {
    assert!(!VERSION.is_empty());
}
