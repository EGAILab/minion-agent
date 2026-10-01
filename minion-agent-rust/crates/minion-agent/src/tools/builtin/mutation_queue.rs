//! Process-wide registration FIFO, followed by independently running per-provider/target FIFOs.
//!
//! Only registration holds the global completion-chain gate across provider key lookup:
//! that serialization is required by TOOL-032. No map lock spans any external call.

use crate::execution::{FileSystem, FsError, FsErrorCode};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::Notify;

type Key = (usize, String);

#[derive(Default)]
struct Completion {
    done: AtomicBool,
    changed: Notify,
}

impl Completion {
    async fn wait(&self) {
        loop {
            let notified = self.changed.notified();
            // Register before testing the latch, including on a multithreaded executor.
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.done.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
    fn release(&self) {
        self.done.store(true, Ordering::Release);
        self.changed.notify_waiters();
    }
}

#[derive(Default)]
struct Queues {
    registration: Mutex<Option<Arc<Completion>>>,
    tails: Mutex<HashMap<Key, Arc<Completion>>>,
}

fn queues() -> &'static Queues {
    static QUEUES: OnceLock<Queues> = OnceLock::new();
    QUEUES.get_or_init(Queues::default)
}

pub(super) struct Entry {
    key: Key,
    current: Arc<Completion>,
    previous: Option<Arc<Completion>>,
    // Retaining the provider prevents identity/address reuse until its entry has released.
    _provider: Arc<dyn FileSystem>,
}

/// Registration is linked synchronously before spawning the filesystem worker. Task
/// scheduling/key-resolution order therefore cannot change registration call order.
pub(super) struct Registration {
    fs: Arc<dyn FileSystem>,
    path: String,
    current: Arc<Completion>,
    previous: Option<Arc<Completion>>,
}

impl Registration {
    pub(super) fn new(fs: Arc<dyn FileSystem>, path: String) -> Self {
        let current = Arc::new(Completion::default());
        let previous = queues().registration.lock().replace(current.clone());
        Self {
            fs,
            path,
            current,
            previous,
        }
    }

    pub(super) async fn acquire(self) -> Result<Entry, FsError> {
        if let Some(previous) = &self.previous {
            previous.wait().await;
        }
        let target = match self.fs.canonical_path(&self.path, None).await {
            Ok(target) => target,
            Err(e)
                if matches!(
                    e.code,
                    FsErrorCode::NotFound | FsErrorCode::NotDirectory | FsErrorCode::NotSupported
                ) =>
            {
                self.fs.absolute_path(&self.path, None).await?
            }
            Err(e) => return Err(e),
        };
        let key = (Arc::as_ptr(&self.fs) as *const () as usize, target);
        let current = Arc::new(Completion::default());
        let previous = queues().tails.lock().insert(key.clone(), current.clone());
        let entry = Entry {
            key,
            current,
            previous,
            _provider: self.fs.clone(),
        };
        drop(self); // registration releases after linkage, not after the operation.
        if let Some(previous) = &entry.previous {
            previous.wait().await;
        }
        Ok(entry)
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.current.release();
        let mut tail = queues().registration.lock();
        if tail
            .as_ref()
            .is_some_and(|tail| Arc::ptr_eq(tail, &self.current))
        {
            *tail = None;
        }
    }
}

impl Drop for Entry {
    fn drop(&mut self) {
        // A detached worker owns its entry until the operation actually settles.
        self.current.release();
        let mut tails = queues().tails.lock();
        if tails
            .get(&self.key)
            .is_some_and(|tail| Arc::ptr_eq(tail, &self.current))
        {
            tails.remove(&self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::LocalFileSystem;

    #[tokio::test]
    async fn registration_order_survives_reverse_worker_poll_order() {
        let root = tempfile::tempdir().unwrap();
        let fs: Arc<dyn FileSystem> = Arc::new(LocalFileSystem::new(root.path()));
        let first = Registration::new(fs.clone(), "a".into());
        let second = Registration::new(fs, "b".into());
        let second_done = second.current.clone();
        let pending = tokio::spawn(second.acquire());
        tokio::task::yield_now().await;
        assert!(!second_done.done.load(Ordering::Acquire));
        let _first = first.acquire().await.unwrap();
        // Once registration settles, different keys may acquire while A is still held.
        let _second = tokio::time::timeout(std::time::Duration::from_secs(2), pending)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn same_key_waits_until_drop_then_drops_its_tail() {
        let root = tempfile::tempdir().unwrap();
        let fs: Arc<dyn FileSystem> = Arc::new(LocalFileSystem::new(root.path()));
        let first = Registration::new(fs.clone(), "a".into())
            .acquire()
            .await
            .unwrap();
        let key = first.key.clone();
        let pending = tokio::spawn(Registration::new(fs, "a".into()).acquire());
        tokio::task::yield_now().await;
        assert!(!pending.is_finished());
        drop(first);
        let second = tokio::time::timeout(std::time::Duration::from_secs(2), pending)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        drop(second);
        assert!(!queues().tails.lock().contains_key(&key));
    }

    #[tokio::test]
    async fn equal_paths_on_different_provider_instances_do_not_share_a_lock() {
        let root = tempfile::tempdir().unwrap();
        let first_fs: Arc<dyn FileSystem> = Arc::new(LocalFileSystem::new(root.path()));
        let second_fs: Arc<dyn FileSystem> = Arc::new(LocalFileSystem::new(root.path()));
        let _first = Registration::new(first_fs, "a".into())
            .acquire()
            .await
            .unwrap();
        let _second = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            Registration::new(second_fs, "a".into()).acquire(),
        )
        .await
        .unwrap()
        .unwrap();
    }

    #[tokio::test]
    async fn completion_latch_remembers_a_release_before_the_waiter_is_polled() {
        let completion = Completion::default();
        completion.release();
        tokio::time::timeout(std::time::Duration::from_secs(2), completion.wait())
            .await
            .unwrap();
    }
}
