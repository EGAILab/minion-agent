use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use minion_agent::auth::{
    DeviceClock, DeviceFlowError, DeviceFlowOptions, DevicePollResult, poll_device_code_flow,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Scenario {
    auth_device_code: Input,
    expect: Expected,
}

#[derive(Deserialize)]
struct Input {
    poll_sequence: Vec<PollOutcome>,
    interval_seconds: Option<f64>,
    expires_in_seconds: Option<f64>,
}

#[derive(Deserialize)]
struct PollOutcome {
    pending: Option<bool>,
    slow_down: Option<SlowDown>,
    failed: Option<Failed>,
    complete: Option<Complete>,
}

#[derive(Deserialize)]
struct SlowDown {
    interval_seconds: Option<f64>,
}
#[derive(Deserialize)]
struct Failed {
    message: String,
}
#[derive(Deserialize)]
struct Complete {
    value: String,
}

#[derive(Deserialize)]
struct Expected {
    poll_count: usize,
    complete: Option<Complete>,
    error: Option<ExpectedError>,
}
#[derive(Deserialize)]
struct ExpectedError {
    #[serde(rename = "type")]
    kind: String,
    message_contains: Option<String>,
}

#[derive(Default)]
struct CanonicalClock {
    nanos: Mutex<u64>,
}

#[async_trait::async_trait]
impl DeviceClock for CanonicalClock {
    fn elapsed_seconds(&self) -> f64 {
        *self.nanos.lock().unwrap() as f64 / 1_000_000_000.0
    }
    async fn sleep_seconds(&self, seconds: f64) {
        *self.nanos.lock().unwrap() += (seconds * 1_000_000_000.0).round() as u64;
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[tokio::test]
async fn all_layer_11_device_code_scenarios_drive_the_real_rust_state_machine() {
    let mut paths = fs::read_dir(root().join("conformance/agent"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("auth-device-code-")
                && path.extension().is_some_and(|value| value == "yaml")
        })
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(paths.len(), 6);

    for path in paths {
        let scenario: Scenario = serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let mut sequence = VecDeque::from(scenario.auth_device_code.poll_sequence);
        let polls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = polls.clone();
        let clock = CanonicalClock::default();
        let result = poll_device_code_flow(
            move || {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let outcome = sequence
                    .pop_front()
                    .expect("canonical poll sequence exhausted");
                std::future::ready(if outcome.pending.is_some() {
                    DevicePollResult::Pending
                } else if let Some(value) = outcome.slow_down {
                    DevicePollResult::SlowDown {
                        interval_seconds: value.interval_seconds,
                    }
                } else if let Some(value) = outcome.failed {
                    DevicePollResult::Failed {
                        message: value.message,
                    }
                } else {
                    DevicePollResult::Complete(
                        outcome.complete.expect("one canonical outcome").value,
                    )
                })
            },
            DeviceFlowOptions {
                interval_seconds: scenario.auth_device_code.interval_seconds,
                expires_in_seconds: scenario.auth_device_code.expires_in_seconds,
                ..Default::default()
            },
            &clock,
        )
        .await;

        assert_eq!(
            polls.load(std::sync::atomic::Ordering::SeqCst),
            scenario.expect.poll_count,
            "{}",
            path.display()
        );
        match (result, scenario.expect.complete, scenario.expect.error) {
            (Ok(actual), Some(expected), None) => {
                assert_eq!(actual, expected.value, "{}", path.display())
            }
            (Err(actual), None, Some(expected)) => {
                let kind = match actual {
                    DeviceFlowError::Failed(_) => "failed",
                    DeviceFlowError::TimedOut(_) => "timed_out",
                    DeviceFlowError::Cancelled => "cancelled",
                };
                assert_eq!(kind, expected.kind, "{}", path.display());
                if let Some(fragment) = expected.message_contains {
                    assert!(
                        actual.to_string().contains(&fragment),
                        "{}: {actual}",
                        path.display()
                    );
                }
            }
            _ => panic!("unexpected canonical result shape for {}", path.display()),
        }
    }
}
