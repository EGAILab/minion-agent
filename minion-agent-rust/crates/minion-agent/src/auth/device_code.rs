use std::future::Future;

use async_trait::async_trait;
use thiserror::Error;

use super::Abortable;

pub const DEFAULT_POLL_INTERVAL_SECONDS: f64 = 5.0;
pub const MINIMUM_INTERVAL_SECONDS: f64 = 1.0;
pub const SLOW_DOWN_INCREMENT_SECONDS: f64 = 5.0;
pub const SET_TIMEOUT_FALLBACK_SECONDS: f64 = 0.001;
pub const CANCEL_MESSAGE: &str = "Login cancelled";
pub const TIMEOUT_MESSAGE: &str = "Device flow timed out";
pub const SLOW_DOWN_TIMEOUT_MESSAGE: &str = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.";

#[derive(Clone, Debug, PartialEq)]
pub enum DevicePollResult<T> {
    Pending,
    SlowDown { interval_seconds: Option<f64> },
    Failed { message: String },
    Complete(T),
}

#[derive(Clone, Default)]
pub struct DeviceFlowOptions {
    pub interval_seconds: Option<f64>,
    pub expires_in_seconds: Option<f64>,
    pub wait_before_first_poll: bool,
    pub signal: Option<std::sync::Arc<dyn Abortable>>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum DeviceFlowError {
    #[error("{CANCEL_MESSAGE}")]
    Cancelled,
    #[error("{0}")]
    TimedOut(&'static str),
    #[error("{0}")]
    Failed(String),
}

#[async_trait]
pub trait DeviceClock: Send + Sync {
    fn elapsed_seconds(&self) -> f64;
    async fn sleep_seconds(&self, seconds: f64);
}

pub struct SystemDeviceClock {
    started: std::time::Instant,
}
impl Default for SystemDeviceClock {
    fn default() -> Self {
        Self {
            started: std::time::Instant::now(),
        }
    }
}

#[async_trait]
impl DeviceClock for SystemDeviceClock {
    fn elapsed_seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }
    async fn sleep_seconds(&self, seconds: f64) {
        tokio::time::sleep(std::time::Duration::from_secs_f64(seconds)).await;
    }
}

pub async fn abortable_sleep(
    seconds: f64,
    signal: Option<&dyn Abortable>,
    clock: &dyn DeviceClock,
) -> Result<(), DeviceFlowError> {
    let seconds = normalize_timer_delay(seconds);
    check_signal(signal)?;
    let mut remaining = seconds;
    while remaining > 0.0 {
        let step = remaining.min(0.05);
        clock.sleep_seconds(step).await;
        remaining -= step;
        check_signal(signal)?;
    }
    Ok(())
}

pub async fn poll_device_code_flow<T, P, F>(
    mut poll: P,
    options: DeviceFlowOptions,
    clock: &dyn DeviceClock,
) -> Result<T, DeviceFlowError>
where
    P: FnMut() -> F,
    F: Future<Output = DevicePollResult<T>>,
{
    let deadline = options
        .expires_in_seconds
        .map_or(f64::INFINITY, |expires| clock.elapsed_seconds() + expires);
    let floored = floor_milliseconds(
        options
            .interval_seconds
            .unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS),
    );
    let mut interval = if floored.is_nan() {
        floored
    } else {
        floored.max(MINIMUM_INTERVAL_SECONDS)
    };
    let mut slow_downs = 0_u64;

    if options.wait_before_first_poll {
        let remaining = deadline - clock.elapsed_seconds();
        if remaining > 0.0 {
            abortable_sleep(
                js_min(interval, remaining),
                options.signal.as_deref(),
                clock,
            )
            .await?;
        }
    }
    while clock.elapsed_seconds() < deadline {
        check_signal(options.signal.as_deref())?;
        match poll().await {
            DevicePollResult::Complete(value) => return Ok(value),
            DevicePollResult::Failed { message } => return Err(DeviceFlowError::Failed(message)),
            DevicePollResult::Pending => {}
            DevicePollResult::SlowDown { interval_seconds } => {
                slow_downs += 1;
                interval = match interval_seconds {
                    Some(value) if value.is_finite() && value > 0.0 => {
                        floor_milliseconds(value).max(MINIMUM_INTERVAL_SECONDS)
                    }
                    _ => {
                        let increased = interval + SLOW_DOWN_INCREMENT_SECONDS;
                        if increased.is_nan() {
                            increased
                        } else {
                            increased.max(MINIMUM_INTERVAL_SECONDS)
                        }
                    }
                };
            }
        }
        let remaining = deadline - clock.elapsed_seconds();
        if remaining <= 0.0 {
            break;
        }
        abortable_sleep(
            js_min(interval, remaining),
            options.signal.as_deref(),
            clock,
        )
        .await?;
    }
    Err(DeviceFlowError::TimedOut(if slow_downs > 0 {
        SLOW_DOWN_TIMEOUT_MESSAGE
    } else {
        TIMEOUT_MESSAGE
    }))
}

fn check_signal(signal: Option<&dyn Abortable>) -> Result<(), DeviceFlowError> {
    if signal.is_some_and(Abortable::aborted) {
        Err(DeviceFlowError::Cancelled)
    } else {
        Ok(())
    }
}

fn floor_milliseconds(seconds: f64) -> f64 {
    if seconds.is_finite() {
        (seconds * 1000.0).floor() / 1000.0
    } else {
        seconds
    }
}

fn normalize_timer_delay(seconds: f64) -> f64 {
    let milliseconds = seconds * 1000.0;
    if !milliseconds.is_finite() || !(1.0..=2_147_483_647.0).contains(&milliseconds) {
        SET_TIMEOUT_FALLBACK_SECONDS
    } else {
        milliseconds.floor() / 1000.0
    }
}

fn js_min(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left.min(right)
    }
}
