use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::RunSignal;

pub trait Abortable: Send + Sync {
    fn aborted(&self) -> bool;
}

/// Explicit cancellation authority for standalone auth operations and tests.
#[derive(Clone, Default)]
pub struct AuthAbortController {
    state: Arc<AtomicBool>,
}

impl AuthAbortController {
    pub fn signal(&self) -> Arc<dyn Abortable> {
        Arc::new(AuthSignal {
            state: self.state.clone(),
        })
    }
    pub fn abort(&self) {
        self.state.store(true, Ordering::Release);
    }
}

#[derive(Clone)]
struct AuthSignal {
    state: Arc<AtomicBool>,
}

impl Abortable for AuthSignal {
    fn aborted(&self) -> bool {
        self.state.load(Ordering::Acquire)
    }
}

impl Abortable for RunSignal {
    fn aborted(&self) -> bool {
        RunSignal::aborted(self)
    }
}

#[derive(Clone)]
pub struct CombinedSignal {
    caller: Option<Arc<dyn Abortable>>,
    deadline: Instant,
}

impl CombinedSignal {
    pub fn new(caller: Option<Arc<dyn Abortable>>, timeout: std::time::Duration) -> Self {
        Self {
            caller,
            deadline: Instant::now() + timeout,
        }
    }

    pub fn with_deadline(caller: Option<Arc<dyn Abortable>>, deadline: Instant) -> Self {
        Self { caller, deadline }
    }
}

impl Abortable for CombinedSignal {
    fn aborted(&self) -> bool {
        self.caller.as_ref().is_some_and(|signal| signal.aborted())
            || Instant::now() >= self.deadline
    }
}

#[derive(Clone, Default)]
pub struct AuthOperationOptions {
    pub signal: Option<Arc<dyn Abortable>>,
}
