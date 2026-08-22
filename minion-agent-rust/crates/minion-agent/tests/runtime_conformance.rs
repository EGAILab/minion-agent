#[path = "support/runtime_scenario.rs"]
mod runtime_scenario;

#[test]
fn canonical_runtime_scenarios_use_the_real_typed_runtime() {
    runtime_scenario::run_all().unwrap();
}
