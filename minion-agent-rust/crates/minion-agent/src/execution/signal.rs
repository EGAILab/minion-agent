use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::runtime::RunSignal;

pub trait AbortSignal: Send + Sync {
    fn aborted(&self) -> bool;
}

impl AbortSignal for RunSignal {
    fn aborted(&self) -> bool {
        RunSignal::aborted(self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CancellationSignal {
    aborted: Arc<AtomicBool>,
}

impl AbortSignal for CancellationSignal {
    fn aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CancellationController {
    signal: CancellationSignal,
}

impl CancellationController {
    pub fn signal(&self) -> CancellationSignal {
        self.signal.clone()
    }

    pub fn abort(&self) {
        self.signal.aborted.store(true, Ordering::Release);
    }
}
