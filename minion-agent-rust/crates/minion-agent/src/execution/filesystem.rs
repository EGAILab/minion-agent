use std::{
    collections::VecDeque,
    env,
    future::Future,
    io,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use url::Url;
use uuid::Uuid;

use super::{AbortSignal, ExecutionWorldIdentity, FsError, FsErrorCode};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileKind {
    File,
    Directory,
    Symlink,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub kind: FileKind,
    pub size: u64,
    pub mtime_ms: u64,
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct TargetKey(Arc<str>);

impl std::fmt::Debug for TargetKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TargetKey(<opaque>)")
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct FsTarget {
    provider_id: Uuid,
    target_key: TargetKey,
    process_path: Arc<str>,
}

impl FsTarget {
    pub fn target_key(&self) -> &TargetKey {
        &self.target_key
    }
}

impl std::fmt::Debug for FsTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FsTarget")
            .field("target_key", &self.target_key)
            .finish_non_exhaustive()
    }
}

#[async_trait]
pub trait FileSystem: Send + Sync {
    fn cwd(&self) -> &Path;
    fn execution_world(&self) -> &ExecutionWorldIdentity;

    async fn absolute_path(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError>;
    async fn join_path(
        &self,
        parts: &[&str],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError>;
    async fn read_text_file(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError>;
    async fn read_text_lines(
        &self,
        path: &str,
        max_lines: Option<isize>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError>;
    async fn read_binary_file(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError>;
    async fn write_file(
        &self,
        path: &str,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn append_file(
        &self,
        path: &str,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn rename_file(
        &self,
        source: &str,
        destination: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn file_info(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError>;
    async fn list_dir(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError>;
    async fn canonical_path(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError>;
    async fn exists(&self, path: &str, signal: Option<&dyn AbortSignal>) -> Result<bool, FsError>;
    async fn create_dir(
        &self,
        path: &str,
        recursive: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn remove(
        &self,
        path: &str,
        recursive: bool,
        force: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn create_temp_dir(
        &self,
        prefix: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError>;
    async fn create_temp_file(
        &self,
        prefix: &str,
        suffix: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError>;
    async fn resolve(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsTarget, FsError>;
    async fn process_path(&self, target: &FsTarget) -> Result<String, FsError>;
    async fn cleanup(&self);
}

#[derive(Clone, Debug)]
pub struct LocalFileSystem {
    cwd: PathBuf,
    provider_id: Uuid,
    world: ExecutionWorldIdentity,
}

impl LocalFileSystem {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self::with_world(cwd, ExecutionWorldIdentity::local())
    }

    pub fn with_world(cwd: impl Into<PathBuf>, world: ExecutionWorldIdentity) -> Self {
        Self {
            cwd: cwd.into(),
            provider_id: Uuid::new_v4(),
            world,
        }
    }

    fn resolved(&self, raw: &str) -> PathBuf {
        let expanded = expand_path(raw);
        let path = if expanded.is_absolute() {
            expanded
        } else {
            self.cwd.join(expanded)
        };
        lexical_normalize(&path)
    }

    fn aborted(signal: Option<&dyn AbortSignal>) -> Result<(), FsError> {
        if signal.is_some_and(AbortSignal::aborted) {
            Err(FsError::new(FsErrorCode::Aborted, "operation aborted"))
        } else {
            Ok(())
        }
    }

    async fn info_for(&self, path: PathBuf) -> Result<FileInfo, FsError> {
        let metadata = tokio::fs::symlink_metadata(&path)
            .await
            .map_err(map_fs_error)?;
        let file_type = metadata.file_type();
        let kind = if file_type.is_symlink() {
            FileKind::Symlink
        } else if file_type.is_dir() {
            FileKind::Directory
        } else if file_type.is_file() {
            FileKind::File
        } else {
            return Err(FsError::new(
                FsErrorCode::Invalid,
                format!("unsupported file type: {}", path.display()),
            ));
        };
        let mtime_ms = metadata
            .modified()
            .ok()
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_millis() as u64);
        Ok(FileInfo {
            name: path
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
            path: path.to_string_lossy().into_owned(),
            kind,
            size: metadata.len(),
            mtime_ms,
        })
    }
}

#[async_trait]
impl FileSystem for LocalFileSystem {
    fn cwd(&self) -> &Path {
        &self.cwd
    }

    fn execution_world(&self) -> &ExecutionWorldIdentity {
        &self.world
    }

    async fn absolute_path(
        &self,
        path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        Ok(self.resolved(path).to_string_lossy().into_owned())
    }

    async fn join_path(
        &self,
        parts: &[&str],
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        let Some((first, rest)) = parts.split_first() else {
            return Ok(".".to_owned());
        };
        let mut path = PathBuf::from(first);
        for part in rest {
            path.push(part.trim_start_matches(['/', '\\']));
        }
        Ok(lexical_normalize(&path).to_string_lossy().into_owned())
    }

    async fn read_text_file(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        Self::aborted(signal)?;
        let bytes = abortable_io(signal, tokio::fs::read(self.resolved(path))).await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn read_text_lines(
        &self,
        path: &str,
        max_lines: Option<isize>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        Self::aborted(signal)?;
        if max_lines.is_some_and(|limit| limit <= 0) {
            return Ok(Vec::new());
        }
        let file = tokio::fs::File::open(self.resolved(path))
            .await
            .map_err(map_fs_error)?;
        let mut reader = BufReader::new(file);
        let mut result = Vec::new();
        loop {
            let mut line = Vec::new();
            let count = abortable_io(signal, reader.read_until(b'\n', &mut line)).await?;
            if count == 0 {
                break;
            }
            Self::aborted(signal)?;
            if line.last() == Some(&b'\n') {
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
            }
            result.push(String::from_utf8_lossy(&line).into_owned());
            if max_lines.is_some_and(|limit| result.len() >= limit as usize) {
                break;
            }
        }
        Self::aborted(signal)?;
        Ok(result)
    }

    async fn read_binary_file(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError> {
        Self::aborted(signal)?;
        abortable_io(signal, tokio::fs::read(self.resolved(path))).await
    }

    async fn write_file(
        &self,
        path: &str,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Self::aborted(signal)?;
        let path = self.resolved(path);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(map_fs_error)?;
        }
        Self::aborted(signal)?;
        abortable_io(signal, tokio::fs::write(path, content)).await
    }

    async fn append_file(
        &self,
        path: &str,
        content: &[u8],
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let path = self.resolved(path);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(map_fs_error)?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .map_err(map_fs_error)?;
        file.write_all(content).await.map_err(map_fs_error)
    }

    async fn rename_file(
        &self,
        source: &str,
        destination: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Self::aborted(signal)?;
        let source = self.resolved(source);
        let destination = self.resolved(destination);
        match tokio::fs::rename(&source, &destination).await {
            Ok(()) => Ok(()),
            #[cfg(windows)]
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                remove_addressed(&destination, true, true).await?;
                tokio::fs::rename(source, destination)
                    .await
                    .map_err(map_fs_error)
            }
            Err(error) => Err(map_fs_error(error)),
        }
    }

    async fn file_info(
        &self,
        path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError> {
        self.info_for(self.resolved(path)).await
    }

    async fn list_dir(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError> {
        Self::aborted(signal)?;
        let mut directory = tokio::fs::read_dir(self.resolved(path))
            .await
            .map_err(map_fs_error)?;
        let mut entries = Vec::new();
        loop {
            let Some(entry) = directory.next_entry().await.map_err(map_fs_error)? else {
                break;
            };
            Self::aborted(signal)?;
            entries.push(self.info_for(entry.path()).await?);
        }
        Ok(entries)
    }

    async fn canonical_path(
        &self,
        path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        tokio::fs::canonicalize(self.resolved(path))
            .await
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(map_fs_error)
    }

    async fn exists(&self, path: &str, _signal: Option<&dyn AbortSignal>) -> Result<bool, FsError> {
        match self.info_for(self.resolved(path)).await {
            Ok(_) => Ok(true),
            Err(error) if error.code == FsErrorCode::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    async fn create_dir(
        &self,
        path: &str,
        recursive: bool,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let path = self.resolved(path);
        if recursive {
            tokio::fs::create_dir_all(path).await.map_err(map_fs_error)
        } else {
            tokio::fs::create_dir(path).await.map_err(map_fs_error)
        }
    }

    async fn remove(
        &self,
        path: &str,
        recursive: bool,
        force: bool,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        remove_addressed(&self.resolved(path), recursive, force).await
    }

    async fn create_temp_dir(
        &self,
        prefix: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        let path = env::temp_dir().join(format!("{prefix}{}", Uuid::new_v4()));
        tokio::fs::create_dir(&path).await.map_err(map_fs_error)?;
        Ok(path.to_string_lossy().into_owned())
    }

    async fn create_temp_file(
        &self,
        prefix: &str,
        suffix: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        let directory = env::temp_dir().join(format!("tmp-{}", Uuid::new_v4()));
        tokio::fs::create_dir(&directory)
            .await
            .map_err(map_fs_error)?;
        let path = directory.join(format!("{prefix}{}{suffix}", Uuid::new_v4()));
        tokio::fs::File::create(&path).await.map_err(map_fs_error)?;
        Ok(path.to_string_lossy().into_owned())
    }

    async fn resolve(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsTarget, FsError> {
        let process_path = self.absolute_path(path, signal).await?;
        let target_key = match self.canonical_path(path, signal).await {
            Ok(canonical) => canonical,
            Err(error)
                if matches!(
                    error.code,
                    FsErrorCode::NotFound | FsErrorCode::NotSupported
                ) =>
            {
                process_path.clone()
            }
            Err(error) => return Err(error),
        };
        Ok(FsTarget {
            provider_id: self.provider_id,
            target_key: TargetKey(Arc::from(target_key)),
            process_path: Arc::from(process_path),
        })
    }

    async fn process_path(&self, target: &FsTarget) -> Result<String, FsError> {
        if target.provider_id != self.provider_id {
            return Err(FsError::new(
                FsErrorCode::Invalid,
                "filesystem target belongs to a different provider",
            ));
        }
        Ok(target.process_path.to_string())
    }

    async fn cleanup(&self) {}
}

async fn remove_addressed(path: &Path, recursive: bool, force: bool) -> Result<(), FsError> {
    let metadata = match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if force && error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(map_fs_error(error)),
    };
    let result = if metadata.is_dir() && !metadata.file_type().is_symlink() {
        if recursive {
            tokio::fs::remove_dir_all(path).await
        } else {
            tokio::fs::remove_dir(path).await
        }
    } else {
        tokio::fs::remove_file(path).await
    };
    match result {
        Ok(()) => Ok(()),
        Err(error) if force && error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(map_fs_error(error)),
    }
}

fn expand_path(raw: &str) -> PathBuf {
    if let Ok(url) = Url::parse(raw)
        && url.scheme() == "file"
        && let Ok(path) = url.to_file_path()
    {
        return path;
    }
    if raw == "~" || raw.starts_with("~/") || raw.starts_with("~\\") {
        let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"));
        if let Some(home) = home {
            let rest = raw.strip_prefix('~').unwrap_or(raw);
            return PathBuf::from(home).join(rest.trim_start_matches(['/', '\\']));
        }
    }
    PathBuf::from(raw)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut prefix = None;
    let mut root = false;
    let mut parts = VecDeque::new();
    for component in path.components() {
        match component {
            Component::Prefix(value) => prefix = Some(value.as_os_str().to_owned()),
            Component::RootDir => root = true,
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.back().is_some_and(|part| part != "..") {
                    parts.pop_back();
                } else if !root {
                    parts.push_back("..".into());
                }
            }
            Component::Normal(value) => parts.push_back(value.to_owned()),
        }
    }
    let mut result = PathBuf::new();
    if let Some(prefix) = prefix {
        result.push(prefix);
    }
    if root {
        result.push(Path::new(std::path::MAIN_SEPARATOR_STR));
    }
    result.extend(parts);
    result
}

fn map_fs_error(error: io::Error) -> FsError {
    let code = match error.kind() {
        io::ErrorKind::NotFound => FsErrorCode::NotFound,
        io::ErrorKind::PermissionDenied => FsErrorCode::PermissionDenied,
        io::ErrorKind::NotADirectory => FsErrorCode::NotDirectory,
        io::ErrorKind::IsADirectory => FsErrorCode::IsDirectory,
        io::ErrorKind::InvalidInput | io::ErrorKind::InvalidData => FsErrorCode::Invalid,
        io::ErrorKind::Unsupported => FsErrorCode::NotSupported,
        _ => FsErrorCode::Unknown,
    };
    FsError::new(code, error.to_string())
}

async fn abortable_io<T>(
    signal: Option<&dyn AbortSignal>,
    future: impl Future<Output = io::Result<T>>,
) -> Result<T, FsError> {
    let Some(signal) = signal else {
        return future.await.map_err(map_fs_error);
    };
    tokio::pin!(future);
    loop {
        if signal.aborted() {
            return Err(FsError::new(FsErrorCode::Aborted, "operation aborted"));
        }
        tokio::select! {
            biased;
            result = &mut future => return result.map_err(map_fs_error),
            () = tokio::time::sleep(std::time::Duration::from_millis(1)) => {}
        }
    }
}
