use std::{path::PathBuf, sync::Arc};

use crate::runtime::Service;

use super::{FileSystem, LocalFileSystem, LocalShell, LocalSubprocess, Shell, Subprocess};

pub struct LocalExecutionProviders {
    pub fs: Arc<LocalFileSystem>,
    pub shell: Arc<LocalShell>,
    pub subprocess: Arc<LocalSubprocess>,
}

impl LocalExecutionProviders {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        let cwd = cwd.into();
        let fs = Arc::new(LocalFileSystem::new(&cwd));
        let subprocess = Arc::new(LocalSubprocess::new(&cwd));
        let shell = Arc::new(LocalShell::new(subprocess.clone()));
        Self {
            fs,
            shell,
            subprocess,
        }
    }
}

#[derive(Clone)]
pub struct FileSystemService(Arc<dyn FileSystem>);

impl FileSystemService {
    pub fn new(provider: Arc<dyn FileSystem>) -> Self {
        Self(provider)
    }

    pub fn provider(&self) -> Arc<dyn FileSystem> {
        Arc::clone(&self.0)
    }
}

impl Service for FileSystemService {
    const NAME: &'static str = "fs";
}

#[derive(Clone)]
pub struct ShellService(Arc<dyn Shell>);

impl ShellService {
    pub fn new(provider: Arc<dyn Shell>) -> Self {
        Self(provider)
    }

    pub fn provider(&self) -> Arc<dyn Shell> {
        Arc::clone(&self.0)
    }
}

impl Service for ShellService {
    const NAME: &'static str = "shell";
}

#[derive(Clone)]
pub struct SubprocessService(Arc<dyn Subprocess>);

impl SubprocessService {
    pub fn new(provider: Arc<dyn Subprocess>) -> Self {
        Self(provider)
    }

    pub fn provider(&self) -> Arc<dyn Subprocess> {
        Arc::clone(&self.0)
    }
}

impl Service for SubprocessService {
    const NAME: &'static str = "subprocess";
}
