//! DIV-009 native-boundary witnesses. No real cwd or environment is mutated.
use super::*;
use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Debug, Default)]
struct FailedCwd {
    reads: AtomicUsize,
}

impl path::NativeInputs for FailedCwd {
    fn cwd_view(&self) -> io::Result<FsPath> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Err(io::Error::from_raw_os_error(2))
    }

    fn drive_view(&self, _device: &[u16]) -> Option<FsPath> {
        None
    }
}

#[cfg(windows)]
#[tokio::test]
#[ignore = "CONTRACT_ASSURANCE_DEFECT: explicit diagnostic, not a certification test"]
async fn extended_provider_read_domain_matches_the_checkpoint() {
    // CE01 rd-1: provider \\?\C:\base, input c:leaf, entry absent.
    // Node reads once and its injected native failure must propagate.
    let inputs = Arc::new(FailedCwd::default());
    let logical = path::resolve_logical_with_inputs(
        &r"\\?\C:\base".into(),
        &"c:leaf".into(),
        inputs.as_ref(),
    );
    assert!(
        logical.is_err(),
        "paired logical resolver observes the native failure"
    );
    assert_eq!(inputs.reads.load(Ordering::SeqCst), 1);
    inputs.reads.store(0, Ordering::SeqCst);
    let mut fs = LocalFileSystem::new(r"\\?\C:\base");
    fs.native_inputs = inputs.clone();
    let result = FileSystem::absolute_path(&fs, &"c:leaf".into(), None).await;
    assert!(
        result.is_err(),
        "real provider must reach the checkpoint's N cell: {result:?}"
    );
    assert_eq!(inputs.reads.load(Ordering::SeqCst), 1);
}

#[cfg(windows)]
#[tokio::test]
async fn ordinary_provider_is_the_no_read_positive() {
    let inputs = Arc::new(FailedCwd::default());
    let mut fs = LocalFileSystem::new(r"C:\base");
    fs.native_inputs = inputs.clone();
    let value = FileSystem::absolute_path(&fs, &"c:leaf".into(), None)
        .await
        .unwrap();
    assert_eq!(value, r"c:\base\leaf");
    assert_eq!(inputs.reads.load(Ordering::SeqCst), 0);
}

#[cfg(windows)]
#[tokio::test]
async fn unc_provider_is_the_read_positive() {
    let inputs = Arc::new(FailedCwd::default());
    let mut fs = LocalFileSystem::new(r"\\srv\share\base");
    fs.native_inputs = inputs.clone();
    let error = FileSystem::absolute_path(&fs, &"c:leaf".into(), None)
        .await
        .unwrap_err();
    assert_eq!(error.code, FsErrorCode::NotFound);
    assert_eq!(error.message, io::Error::from_raw_os_error(2).to_string());
    assert_eq!(inputs.reads.load(Ordering::SeqCst), 1);
}
