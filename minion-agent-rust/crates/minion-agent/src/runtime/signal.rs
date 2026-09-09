use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::tools::ToolExecutionSignal;

/// Read-only cooperative cancellation state shared by one Agent run.
#[derive(Clone, Debug, Default)]
pub struct RunSignal {
    aborted: Arc<AtomicBool>,
}

impl PartialEq for RunSignal {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.aborted, &other.aborted)
    }
}

impl Eq for RunSignal {}

impl RunSignal {
    pub fn aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }
}

impl ToolExecutionSignal for RunSignal {
    fn is_cancelled(&self) -> bool {
        self.aborted()
    }
}

/// Private mutation authority paired with a public [`RunSignal`] view.
#[derive(Clone, Debug, Default)]
pub(crate) struct RunAbortController {
    signal: RunSignal,
}

impl RunAbortController {
    pub(crate) fn signal(&self) -> RunSignal {
        self.signal.clone()
    }

    pub(crate) fn abort(&self) {
        self.signal.aborted.store(true, Ordering::Release);
    }
}
