use std::{
    collections::VecDeque,
    env,
    fmt::Debug,
    future::Future,
    io,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

use ada_url::{HostType, Idna, Url as AdaUrl};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use uuid::Uuid;

use super::path::{basename, from_native, join, native, resolve};
use super::{AbortSignal, ExecutionWorldIdentity, FsError, FsErrorCode, FsPath};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileKind {
    File,
    Directory,
    Symlink,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileInfo {
    pub name: FsPath,
    pub path: FsPath,
    pub kind: FileKind,
    pub size: u64,
    pub mtime_ms: u64,
}

/// The resolved kind of one addressed directory entry (`EXEC-007`).
///
/// Unlike [`FileKind`], this preserves whether a file or directory was reached through a
/// symlink. Symlinks to every other kind deliberately collapse to [`Self::Other`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DirEntryProbeKind {
    File,
    Directory,
    SymlinkToFile,
    SymlinkToDirectory,
    Other,
}

/// The classification of one resolved, addressed directory entry (`EXEC-007`).
///
/// `name` and `path` identify the addressed entry itself, including when it is a symlink; they
/// never substitute the followed target's identity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DirEntryProbe {
    pub name: FsPath,
    pub path: FsPath,
    pub kind: DirEntryProbeKind,
}

#[async_trait]
trait DirectoryProbeOperations: Debug + Send + Sync {
    async fn read_dir_names(&self, path: &Path) -> io::Result<Vec<String>>;
    async fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata>;
    async fn metadata(&self, path: &Path) -> io::Result<std::fs::Metadata>;
}

#[derive(Debug, Default)]
struct TokioDirectoryProbeOperations;

#[async_trait]
impl DirectoryProbeOperations for TokioDirectoryProbeOperations {
    async fn read_dir_names(&self, path: &Path) -> io::Result<Vec<String>> {
        let mut directory = tokio::fs::read_dir(path).await?;
        let mut names = Vec::new();
        while let Some(entry) = directory.next_entry().await? {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        Ok(names)
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        tokio::fs::symlink_metadata(path).await
    }

    async fn metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        tokio::fs::metadata(path).await
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct TargetKey(Arc<FsPath>);

impl std::fmt::Debug for TargetKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TargetKey(<opaque>)")
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct FsTarget {
    provider_id: Uuid,
    target_key: TargetKey,
    process_path: Arc<FsPath>,
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
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError>;
    async fn join_path(
        &self,
        parts: &[&FsPath],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError>;
    async fn read_text_file(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError>;
    async fn read_text_lines(
        &self,
        path: &FsPath,
        max_lines: Option<isize>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError>;
    async fn read_binary_file(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError>;
    async fn write_file(
        &self,
        path: &FsPath,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn append_file(
        &self,
        path: &FsPath,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn rename_file(
        &self,
        source: &FsPath,
        destination: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn file_info(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError>;
    async fn list_dir(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError>;
    async fn list_dir_raw(
        &self,
        _path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        Err(FsError::new(
            FsErrorCode::NotSupported,
            "list_dir_raw is not supported by this filesystem provider",
        ))
    }
    async fn probe_dir_entry(
        &self,
        _path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<DirEntryProbe, FsError> {
        Err(FsError::new(
            FsErrorCode::NotSupported,
            "probe_dir_entry is not supported by this filesystem provider",
        ))
    }
    /// Readability capability used by the Layer-13 read tool (`EXEC-008`).
    /// Providers without it answer `not_supported`, never a fabricated success.
    async fn check_readable(
        &self,
        _path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Err(FsError::new(
            FsErrorCode::NotSupported,
            "check_readable is not supported by this filesystem provider",
        ))
    }
    /// One combined read-and-write access decision (`EXEC-009`). The signal is accepted but
    /// is not inspected by this metadata-class query.
    async fn check_read_write(
        &self,
        _path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Err(FsError::new(
            FsErrorCode::NotSupported,
            "check_read_write is not supported by this filesystem provider",
        ))
    }
    async fn canonical_path(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError>;
    async fn exists(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<bool, FsError>;
    async fn create_dir(
        &self,
        path: &FsPath,
        recursive: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError>;
    async fn remove(
        &self,
        path: &FsPath,
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
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsTarget, FsError>;
    async fn process_path(&self, target: &FsTarget) -> Result<FsPath, FsError>;
    async fn cleanup(&self);
}

#[derive(Clone, Debug)]
pub struct LocalFileSystem {
    cwd: PathBuf,
    provider_id: Uuid,
    world: ExecutionWorldIdentity,
    directory_probe_operations: Arc<dyn DirectoryProbeOperations>,
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
            directory_probe_operations: Arc::new(TokioDirectoryProbeOperations),
        }
    }

    fn resolved(&self, raw: &FsPath) -> FsPath {
        resolve(&self.cwd, raw)
    }

    fn aborted(signal: Option<&dyn AbortSignal>, logical: &FsPath) -> Result<(), FsError> {
        if signal.is_some_and(AbortSignal::aborted) {
            Err(FsError::new(FsErrorCode::Aborted, "operation aborted").with_path(logical))
        } else {
            Ok(())
        }
    }

    async fn info_for(&self, logical: &FsPath) -> Result<FileInfo, FsError> {
        let path = native(logical);
        let metadata = self
            .directory_probe_operations
            .symlink_metadata(&path)
            .await
            .map_err(|e| call_error(e, &path, logical))?;
        let file_type = metadata.file_type();
        let kind = if file_type.is_symlink() {
            FileKind::Symlink
        } else if file_type.is_dir() {
            FileKind::Directory
        } else if file_type.is_file() {
            FileKind::File
        } else {
            return Err(
                FsError::new(FsErrorCode::Invalid, "unsupported file type").with_path(logical)
            );
        };
        let mtime_ms = metadata
            .modified()
            .ok()
            .and_then(|m| m.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_millis() as u64);
        Ok(FileInfo {
            name: basename(logical),
            path: logical.clone(),
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
        path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        Ok(self.resolved(path))
    }
    async fn join_path(
        &self,
        parts: &[&FsPath],
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        Ok(join(parts))
    }
    async fn read_text_file(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        let bytes = <Self as FileSystem>::read_binary_file(self, path, signal).await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    async fn read_text_lines(
        &self,
        path: &FsPath,
        max_lines: Option<isize>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        let logical = self.resolved(path);
        Self::aborted(signal, &logical)?;
        if max_lines.is_some_and(|limit| limit <= 0) {
            return Ok(Vec::new());
        }
        let os = native(&logical);
        let file = tokio::fs::File::open(&os)
            .await
            .map_err(|e| call_error(e, &os, &logical))?;
        let mut reader = BufReader::new(file);
        let mut result = Vec::new();
        loop {
            let mut line = Vec::new();
            let count = abortable_io(signal, reader.read_until(b'\n', &mut line))
                .await
                .map_err(|e| io_origin(e, &logical, &os, true))?;
            if count == 0 {
                break;
            }
            Self::aborted(signal, &logical)?;
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
        Self::aborted(signal, &logical)?;
        Ok(result)
    }
    async fn read_binary_file(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError> {
        let logical = self.resolved(path);
        Self::aborted(signal, &logical)?;
        let os = native(&logical);
        abortable_io(signal, tokio::fs::read(&os))
            .await
            .map_err(|e| io_origin(e, &logical, &os, true))
    }
    async fn write_file(
        &self,
        path: &FsPath,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let logical = self.resolved(path);
        Self::aborted(signal, &logical)?;
        let os = native(&logical);
        if let Some(parent) = os.parent() {
            node_mkdirp(parent)
                .await
                .map_err(|e| logical_fallback(e, &logical))?;
        }
        Self::aborted(signal, &logical)?;
        abortable_io(signal, tokio::fs::write(&os, content))
            .await
            .map_err(|e| io_origin(e, &logical, &os, false))
    }
    async fn append_file(
        &self,
        path: &FsPath,
        content: &[u8],
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let logical = self.resolved(path);
        let os = native(&logical);
        if let Some(parent) = os.parent() {
            node_mkdirp(parent)
                .await
                .map_err(|e| logical_fallback(e, &logical))?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&os)
            .await
            .map_err(|e| append_origin(e, &logical, &os))?;
        file.write_all(content)
            .await
            .map_err(|e| append_origin(e, &logical, &os))?;
        // Join Tokio's pending blocking write before exposing append completion.
        file.flush()
            .await
            .map_err(|e| append_origin(e, &logical, &os))
    }
    async fn rename_file(
        &self,
        source: &FsPath,
        destination: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let logical = self.resolved(source);
        // Pi's pre-call abort uses the destination; caught OS errors use source.
        let logical_destination = self.resolved(destination);
        Self::aborted(signal, &logical_destination)?;
        let source = native(&logical);
        let destination = native(&logical_destination);
        match tokio::fs::rename(&source, &destination).await {
            Ok(()) => Ok(()),
            #[cfg(windows)]
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                remove_addressed(&destination, true, true)
                    .await
                    .map_err(|e| e.with_path(from_native(&source)))?;
                tokio::fs::rename(&source, &destination)
                    .await
                    .map_err(|e| native_error(e, &source))
            }
            Err(error) => {
                if nul_binding_error(&error, &source) || nul_binding_error(&error, &destination) {
                    Err(map_fs_error(error).with_path(&logical))
                } else {
                    Err(native_error(error, &source))
                }
            }
        }
    }
    async fn file_info(
        &self,
        path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError> {
        self.info_for(&self.resolved(path)).await
    }
    async fn list_dir(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError> {
        let logical = self.resolved(path);
        Self::aborted(signal, &logical)?;
        let os = native(&logical);
        let names = self
            .directory_probe_operations
            .read_dir_names(&os)
            .await
            .map_err(|e| call_error(e, &os, &logical))?;
        let mut entries = Vec::new();
        for name in names {
            Self::aborted(signal, &logical)?;
            let entry_path = join(&[&logical, &name.into()]);
            entries.push(self.info_for(&entry_path).await?);
        }
        Ok(entries)
    }
    async fn list_dir_raw(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        let logical = self.resolved(path);
        Self::aborted(signal, &logical)?;
        let os = native(&logical);
        self.directory_probe_operations
            .read_dir_names(&os)
            .await
            .map_err(|e| call_error(e, &os, &logical))
    }
    async fn probe_dir_entry(
        &self,
        path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<DirEntryProbe, FsError> {
        let logical = self.resolved(path);
        let os = native(&logical);
        let addressed = self
            .directory_probe_operations
            .symlink_metadata(&os)
            .await
            .map_err(|e| call_error(e, &os, &logical))?;
        let file_type = addressed.file_type();
        let kind = if file_type.is_symlink() {
            let target = self
                .directory_probe_operations
                .metadata(&os)
                .await
                .map_err(|e| call_error(e, &os, &logical))?;
            if target.is_file() {
                DirEntryProbeKind::SymlinkToFile
            } else if target.is_dir() {
                DirEntryProbeKind::SymlinkToDirectory
            } else {
                DirEntryProbeKind::Other
            }
        } else if file_type.is_file() {
            DirEntryProbeKind::File
        } else if file_type.is_dir() {
            DirEntryProbeKind::Directory
        } else {
            DirEntryProbeKind::Other
        };
        Ok(DirEntryProbe {
            name: basename(&logical),
            path: logical,
            kind,
        })
    }
    async fn check_readable(
        &self,
        path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let logical = self.resolved(path);
        let os = native(&logical);
        let worker_path = os.clone();
        tokio::task::spawn_blocking(move || check_local_readable(&worker_path))
            .await
            .map_err(|e| FsError::new(FsErrorCode::Unknown, e.to_string()).with_path(&logical))?
            .map_err(|e| access_origin(e, &logical, &os))
    }
    async fn check_read_write(
        &self,
        path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let logical = self.resolved(path);
        let os = native(&logical);
        let worker_path = os.clone();
        tokio::task::spawn_blocking(move || check_local_read_write(&worker_path))
            .await
            .map_err(|e| FsError::new(FsErrorCode::Unknown, e.to_string()).with_path(&logical))?
            .map_err(|e| access_origin(e, &logical, &os))
    }
    async fn canonical_path(
        &self,
        path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        let logical = self.resolved(path);
        let os = native(&logical);
        tokio::fs::canonicalize(&os)
            .await
            .map(|p| from_native(&p))
            .map_err(|e| call_error(e, &os, &logical))
    }
    async fn exists(
        &self,
        path: &FsPath,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<bool, FsError> {
        match self.info_for(&self.resolved(path)).await {
            Ok(_) => Ok(true),
            Err(e) if e.code == FsErrorCode::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }
    async fn create_dir(
        &self,
        path: &FsPath,
        recursive: bool,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let logical = self.resolved(path);
        let os = native(&logical);
        if recursive {
            node_mkdirp(&os)
                .await
                .map_err(|e| logical_fallback(e, &logical))
        } else {
            tokio::fs::create_dir(&os)
                .await
                .map_err(|e| call_error(e, &os, &logical))
        }
    }
    async fn remove(
        &self,
        path: &FsPath,
        recursive: bool,
        force: bool,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let logical = self.resolved(path);
        remove_addressed(&native(&logical), recursive, force)
            .await
            .map_err(|e| {
                if e.path.is_none() {
                    e.with_path(logical)
                } else {
                    e
                }
            })
    }
    async fn create_temp_dir(
        &self,
        prefix: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        let path = env::temp_dir().join(format!("{prefix}{}", Uuid::new_v4()));
        tokio::fs::create_dir(&path)
            .await
            .map_err(|e| native_error(e, &path))?;
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
            .map_err(|e| native_error(e, &directory))?;
        let path = directory.join(format!("{prefix}{}{suffix}", Uuid::new_v4()));
        tokio::fs::File::create(&path)
            .await
            .map_err(|e| native_error(e, &path))?;
        Ok(path.to_string_lossy().into_owned())
    }
    async fn resolve(
        &self,
        path: &FsPath,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsTarget, FsError> {
        let absolute = self.resolved(path);
        let key = match <Self as FileSystem>::canonical_path(self, path, signal).await {
            Ok(canonical) => canonical,
            Err(e) if matches!(e.code, FsErrorCode::NotFound | FsErrorCode::NotSupported) => {
                absolute
            }
            Err(e) => return Err(e),
        };
        Ok(FsTarget {
            provider_id: self.provider_id,
            target_key: TargetKey(Arc::new(key.clone())),
            process_path: Arc::new(key),
        })
    }
    async fn process_path(&self, target: &FsTarget) -> Result<FsPath, FsError> {
        if target.provider_id != self.provider_id {
            return Err(FsError::new(
                FsErrorCode::Invalid,
                "filesystem target belongs to a different provider",
            ));
        }
        Ok((*target.process_path).clone())
    }
    async fn cleanup(&self) {}
}

// Scalar callers are ergonomic conversions into the same lossless trait seam.
impl LocalFileSystem {
    pub async fn absolute_path(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        <Self as FileSystem>::absolute_path(self, &path.into(), signal).await
    }
    pub async fn read_text_file(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<String, FsError> {
        <Self as FileSystem>::read_text_file(self, &path.into(), signal).await
    }
    pub async fn read_binary_file(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<u8>, FsError> {
        <Self as FileSystem>::read_binary_file(self, &path.into(), signal).await
    }
    pub async fn file_info(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FileInfo, FsError> {
        <Self as FileSystem>::file_info(self, &path.into(), signal).await
    }
    pub async fn list_dir(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<FileInfo>, FsError> {
        <Self as FileSystem>::list_dir(self, &path.into(), signal).await
    }
    pub async fn list_dir_raw(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        <Self as FileSystem>::list_dir_raw(self, &path.into(), signal).await
    }
    pub async fn probe_dir_entry(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<DirEntryProbe, FsError> {
        <Self as FileSystem>::probe_dir_entry(self, &path.into(), signal).await
    }
    pub async fn check_readable(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        <Self as FileSystem>::check_readable(self, &path.into(), signal).await
    }
    pub async fn check_read_write(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        <Self as FileSystem>::check_read_write(self, &path.into(), signal).await
    }
    pub async fn canonical_path(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        <Self as FileSystem>::canonical_path(self, &path.into(), signal).await
    }
    pub async fn exists(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<bool, FsError> {
        <Self as FileSystem>::exists(self, &path.into(), signal).await
    }
    pub async fn resolve(
        &self,
        path: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsTarget, FsError> {
        <Self as FileSystem>::resolve(self, &path.into(), signal).await
    }
    pub async fn read_text_lines(
        &self,
        path: impl Into<FsPath>,
        max_lines: Option<isize>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        <Self as FileSystem>::read_text_lines(self, &path.into(), max_lines, signal).await
    }
    pub async fn write_file(
        &self,
        path: impl Into<FsPath>,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        <Self as FileSystem>::write_file(self, &path.into(), content, signal).await
    }
    pub async fn append_file(
        &self,
        path: impl Into<FsPath>,
        content: &[u8],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        <Self as FileSystem>::append_file(self, &path.into(), content, signal).await
    }
    pub async fn create_dir(
        &self,
        path: impl Into<FsPath>,
        recursive: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        <Self as FileSystem>::create_dir(self, &path.into(), recursive, signal).await
    }
    pub async fn remove(
        &self,
        path: impl Into<FsPath>,
        recursive: bool,
        force: bool,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        <Self as FileSystem>::remove(self, &path.into(), recursive, force, signal).await
    }
    pub async fn rename_file(
        &self,
        source: impl Into<FsPath>,
        destination: impl Into<FsPath>,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        <Self as FileSystem>::rename_file(self, &source.into(), &destination.into(), signal).await
    }
    pub async fn join_path<P: Clone + Into<FsPath>>(
        &self,
        parts: &[P],
        signal: Option<&dyn AbortSignal>,
    ) -> Result<FsPath, FsError> {
        let parts: Vec<FsPath> = parts.iter().cloned().map(Into::into).collect();
        let refs: Vec<&FsPath> = parts.iter().collect();
        <Self as FileSystem>::join_path(self, &refs, signal).await
    }
}

fn native_error(error: io::Error, path: &Path) -> FsError {
    if nul_binding_error(&error, path) {
        // Node's argument-validation failure has no err.path. Keep the existing
        // Rust code/message mapper, leaving the enclosing logical fallback intact.
        map_fs_error(error)
    } else {
        map_fs_error(error).with_path(from_native(path))
    }
}

fn nul_binding_error(error: &io::Error, path: &Path) -> bool {
    error.raw_os_error().is_none()
        && error.kind() == io::ErrorKind::InvalidInput
        && path.to_string_lossy().contains('\0')
}

fn logical_fallback(error: FsError, logical: &FsPath) -> FsError {
    if error.path.is_none() {
        error.with_path(logical)
    } else {
        error
    }
}

fn call_error(error: io::Error, os: &Path, logical: &FsPath) -> FsError {
    logical_fallback(native_error(error, os), logical)
}

fn access_origin(error: FsError, logical: &FsPath, os: &Path) -> FsError {
    if os.to_string_lossy().contains('\0') {
        error.with_path(logical)
    } else {
        error.with_path(from_native(os))
    }
}

fn io_origin(error: FsError, logical: &FsPath, os: &Path, directory_read: bool) -> FsError {
    if error.code == FsErrorCode::Aborted
        || (directory_read && error.code == FsErrorCode::IsDirectory)
        || (error.code == FsErrorCode::Invalid && os.to_string_lossy().contains('\0'))
    {
        error.with_path(logical)
    } else {
        error.with_path(from_native(os))
    }
}

fn append_origin(error: io::Error, logical: &FsPath, os: &Path) -> FsError {
    let no_path = nul_binding_error(&error, os);
    let error = map_fs_error(error);
    if no_path || (cfg!(windows) && error.code == FsErrorCode::IsDirectory) {
        error.with_path(logical)
    } else {
        error.with_path(from_native(os))
    }
}

/// Node v22.15.1 MKDirpAsync's explicit walk, including its failed-stat ENOTDIR branch.
#[async_trait]
trait MkdirOperations: Send + Sync {
    async fn mkdir(&self, path: &Path) -> io::Result<()>;
    async fn stat(&self, path: &Path) -> io::Result<std::fs::Metadata>;
}
struct TokioMkdirOperations;
#[async_trait]
impl MkdirOperations for TokioMkdirOperations {
    async fn mkdir(&self, path: &Path) -> io::Result<()> {
        tokio::fs::create_dir(path).await
    }
    async fn stat(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        tokio::fs::metadata(path).await
    }
}
async fn node_mkdirp(path: &Path) -> Result<(), FsError> {
    node_mkdirp_with(path, &TokioMkdirOperations).await
}
async fn node_mkdirp_with(path: &Path, operations: &dyn MkdirOperations) -> Result<(), FsError> {
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        match operations.mkdir(&current).await {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let parent = current.parent().filter(|p| *p != current);
                let Some(parent) = parent else {
                    return Err(native_error(e, &current));
                };
                stack.push(current.clone());
                stack.push(parent.to_path_buf());
            }
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::PermissionDenied | io::ErrorKind::NotADirectory
                ) =>
            {
                return Err(native_error(e, &current));
            }
            Err(e) => {
                let original_exists = e.kind() == io::ErrorKind::AlreadyExists;
                match operations.stat(&current).await {
                    Ok(meta) if meta.is_dir() => {
                        if !original_exists || stack.is_empty() {
                            return Ok(());
                        }
                    }
                    _ if original_exists && !stack.is_empty() => {
                        return Err(FsError::new(FsErrorCode::NotDirectory, "not a directory")
                            .with_path(from_native(&current)));
                    }
                    Err(stat_error) => return Err(native_error(stat_error, &current)),
                    Ok(_) => {
                        return Err(native_error(
                            io::Error::from(io::ErrorKind::AlreadyExists),
                            &current,
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Carry each actual failing call's origin. Multiple-failure settlement selection (#127)
/// and Windows EPERM retry (#126) remain explicitly outside this delta.
fn remove_addressed(
    path: &Path,
    recursive: bool,
    force: bool,
) -> std::pin::Pin<Box<dyn Future<Output = Result<(), FsError>> + Send + '_>> {
    Box::pin(async move {
        let metadata = match tokio::fs::symlink_metadata(path).await {
            Ok(metadata) => metadata,
            Err(e) if force && e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(native_error(e, path)),
        };
        let result = if metadata.is_dir() && !metadata.file_type().is_symlink() {
            if recursive {
                // Node rimraf attempts rmdir first; it enumerates only a nonempty directory.
                match tokio::fs::remove_dir(path).await {
                    Ok(()) => return Ok(()),
                    Err(e) if e.kind() == io::ErrorKind::DirectoryNotEmpty => {}
                    Err(e) if force && e.kind() == io::ErrorKind::NotFound => return Ok(()),
                    Err(e) => return Err(native_error(e, path)),
                }
                let mut directory = tokio::fs::read_dir(path)
                    .await
                    .map_err(|e| native_error(e, path))?;
                while let Some(entry) = directory
                    .next_entry()
                    .await
                    .map_err(|e| native_error(e, path))?
                {
                    remove_addressed(&entry.path(), true, force).await?;
                }
            }
            tokio::fs::remove_dir(path).await
        } else {
            tokio::fs::remove_file(path).await
        };
        match result {
            Ok(()) => Ok(()),
            Err(e) if force && e.kind() == io::ErrorKind::NotFound => Ok(()),
            // Preserve the excluded #125 code/outcome behavior, but use Pi's
            // logical no-path carrier for a non-recursive directory refusal.
            Err(e) if metadata.is_dir() && !recursive => Err(map_fs_error(e)),
            Err(e) => Err(native_error(e, path)),
        }
    })
}

pub(crate) fn resolve_local_path(cwd: &Path, raw: &str) -> PathBuf {
    let expanded = expand_path(raw);
    let path = if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    };
    lexical_normalize(&path)
}

fn expand_path(raw: &str) -> PathBuf {
    if raw.starts_with("file://")
        && let Some(path) = file_url_to_path(raw, cfg!(windows))
    {
        return PathBuf::from(path);
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

/// Strict form of the already-certified file URL conversion, exposed for callers that must
/// reject an invalid `file://` input before any filesystem access (`TOOL-026`, R002-A).
/// The ordinary `resolve_local_path` fallback remains unchanged.
pub fn file_url_to_path(raw: &str, windows: bool) -> Option<String> {
    let normalized = raw.replace('\\', "/");
    let url = AdaUrl::parse(normalized, None).ok()?;
    if url.protocol() != "file:" {
        return None;
    }
    let host = unicode_host(&url)?;

    let raw_pathname = url.pathname();
    let lowered_pathname = raw_pathname.to_ascii_lowercase();
    if lowered_pathname.contains("%2f") || (windows && lowered_pathname.contains("%5c")) {
        return None;
    }
    let pathname = strict_percent_decode(raw_pathname)?;

    if windows {
        if !host.is_empty() && host != "localhost" {
            return Some(format!("\\\\{host}{pathname}").replace('/', "\\"));
        }
        let bytes = pathname.as_bytes();
        if bytes.len() < 3
            || bytes[0] != b'/'
            || !bytes[1].is_ascii_alphabetic()
            || bytes[2] != b':'
        {
            return None;
        }
        return Some(pathname[1..].replace('/', "\\"));
    }

    if !host.is_empty() && host != "localhost" {
        return None;
    }
    Some(if pathname.is_empty() {
        "/".to_owned()
    } else {
        pathname
    })
}

fn unicode_host(url: &AdaUrl) -> Option<String> {
    let ascii_host = url.host();
    if url.host_type() == HostType::Domain
        && ascii_host.split('.').any(|label| label.starts_with("xn--"))
    {
        let decoded = Idna::unicode(ascii_host);
        (decoded != ascii_host).then_some(decoded)
    } else {
        Some(ascii_host.to_owned())
    }
}

fn strict_percent_decode(value: &str) -> Option<String> {
    let input = value.as_bytes();
    let mut output = Vec::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() {
        if input[index] == b'%' {
            let high = *input.get(index + 1)?;
            let low = *input.get(index + 2)?;
            output.push((hex_value(high)? << 4) | hex_value(low)?);
            index += 3;
        } else {
            output.push(input[index]);
            index += 1;
        }
    }
    String::from_utf8(output).ok()
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
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

/// `NotSupported` is reserved for a provider missing EXEC-008, not a host syscall failure.
/// Keep the certified general filesystem error mapping unchanged for every other operation.
fn map_readability_error(error: io::Error) -> FsError {
    if error.kind() == io::ErrorKind::Unsupported {
        FsError::new(FsErrorCode::Unknown, error.to_string())
    } else {
        map_fs_error(error)
    }
}

/// A host `Unsupported` error is not a provider-capability answer. The latter is reserved for
/// the trait default when EXEC-009 is absent.
fn map_read_write_error(error: io::Error) -> FsError {
    if error.kind() == io::ErrorKind::Unsupported {
        FsError::new(FsErrorCode::Unknown, error.to_string())
    } else {
        map_fs_error(error)
    }
}

#[cfg(unix)]
fn check_local_read_write(path: &Path) -> Result<(), FsError> {
    use nix::unistd::{AccessFlags, access};

    check_posix_read_write_with(path, |path| {
        access(path, AccessFlags::R_OK | AccessFlags::W_OK)
            .map_err(|errno| io::Error::from_raw_os_error(errno as i32))
    })
}

#[cfg(unix)]
fn check_posix_read_write_with(
    path: &Path,
    query: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<(), FsError> {
    if path.as_os_str().as_encoded_bytes().contains(&0) {
        return Err(FsError::new(
            FsErrorCode::Unknown,
            "path contains an embedded NUL byte",
        ));
    }
    // One host access(R_OK | W_OK), retaining its own errno; never a stat or content open.
    query(path).map_err(map_read_write_error)
}

#[cfg(windows)]
fn check_local_read_write(path: &Path) -> Result<(), FsError> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_READ_DATA, FILE_SHARE_DELETE, FILE_SHARE_READ,
        FILE_SHARE_WRITE, FILE_WRITE_DATA,
    };

    check_windows_read_write_with(path, |path| {
        // FILE_READ_DATA | FILE_WRITE_DATA is also LIST_DIRECTORY | ADD_FILE for a directory.
        // OPEN_EXISTING is OpenOptions' default disposition; no create/truncate flags are set.
        // Do not request FILE_DELETE_CHILD (0x40) or OPEN_REPARSE_POINT.
        std::fs::OpenOptions::new()
            .access_mode(FILE_READ_DATA | FILE_WRITE_DATA)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)
            .map(|_| ())
    })
}

#[cfg(windows)]
fn check_windows_read_write_with(
    path: &Path,
    query: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<(), FsError> {
    use std::os::windows::ffi::OsStrExt;

    if path.as_os_str().encode_wide().any(|unit| unit == 0) {
        return Err(FsError::new(
            FsErrorCode::Unknown,
            "path contains an embedded NUL byte",
        ));
    }
    query(path).map_err(map_read_write_error)
}

#[cfg(unix)]
fn check_local_readable(path: &Path) -> Result<(), FsError> {
    use nix::unistd::{AccessFlags, access};

    check_posix_readable_with(path, |path| {
        access(path, AccessFlags::R_OK).map_err(|errno| io::Error::from_raw_os_error(errno as i32))
    })
}

/// The injected syscall boundary lets a race witness remove the target immediately before the
/// one authoritative access query, without changing production behavior.
#[cfg(unix)]
fn check_posix_readable_with(
    path: &Path,
    query: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<(), FsError> {
    if path.as_os_str().as_encoded_bytes().contains(&0) {
        return Err(FsError::new(
            FsErrorCode::Unknown,
            "path contains an embedded NUL byte",
        ));
    }
    // Exactly one access(R_OK): no preliminary stat or content open may change the error site.
    query(path).map_err(map_readability_error)
}

#[cfg(windows)]
fn check_local_readable(path: &Path) -> Result<(), FsError> {
    use std::os::windows::{ffi::OsStrExt, fs::OpenOptionsExt};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_READ_DATA, FILE_SHARE_DELETE, FILE_SHARE_READ,
        FILE_SHARE_WRITE,
    };

    if path.as_os_str().encode_wide().any(|unit| unit == 0) {
        return Err(FsError::new(
            FsErrorCode::Unknown,
            "path contains an embedded NUL byte",
        ));
    }
    // FILE_READ_DATA and FILE_LIST_DIRECTORY share the same access bit. Backup semantics
    // permits directory handles; neither content nor directory entries are consumed.
    std::fs::OpenOptions::new()
        .access_mode(FILE_READ_DATA)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
        .map(|_| ())
        .map_err(map_readability_error)
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Debug)]
    struct VanishingEntry;
    #[async_trait]
    impl DirectoryProbeOperations for VanishingEntry {
        async fn read_dir_names(&self, path: &Path) -> io::Result<Vec<String>> {
            let names = TokioDirectoryProbeOperations.read_dir_names(path).await?;
            tokio::fs::remove_file(path.join("entry")).await?;
            Ok(names)
        }
        async fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
            TokioDirectoryProbeOperations.symlink_metadata(path).await
        }
        async fn metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
            TokioDirectoryProbeOperations.metadata(path).await
        }
    }
    #[tokio::test]
    async fn list_dir_failure_names_the_vanished_native_entry_not_the_directory() {
        let root = env::temp_dir().join(format!("minion-list-origin-{}", Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let mut fs = LocalFileSystem::new(&root);
        let logical = FsPath::from_code_units(vec![100, 0xd800]);
        fs.write_file(
            FsPath::from_code_units(vec![100, 0xd800, 47, 101, 110, 116, 114, 121]),
            b"x",
            None,
        )
        .await
        .unwrap();
        fs.directory_probe_operations = Arc::new(VanishingEntry);
        let error = fs.list_dir(&logical, None).await.unwrap_err();
        assert_eq!(error.code, FsErrorCode::NotFound);
        assert_eq!(
            error.path,
            Some(from_native(&root.join("d\u{fffd}").join("entry")))
        );
        assert!(!error.path.unwrap().code_units().contains(&0xd800));
        std::fs::remove_dir_all(root).unwrap();
    }

    struct FailedStatWalk;
    #[async_trait]
    impl MkdirOperations for FailedStatWalk {
        async fn mkdir(&self, path: &Path) -> io::Result<()> {
            Err(io::Error::from(if path.file_name().unwrap() == "leaf" {
                io::ErrorKind::NotFound
            } else {
                io::ErrorKind::AlreadyExists
            }))
        }
        async fn stat(&self, _path: &Path) -> io::Result<std::fs::Metadata> {
            Err(io::Error::from(io::ErrorKind::NotFound))
        }
    }
    #[tokio::test]
    async fn mkdir_walk_failed_stat_is_enotdir_only_with_pending_children() {
        let root = env::temp_dir().join("minion-mkdir-origin");
        let error = node_mkdirp_with(&root.join("parent").join("leaf"), &FailedStatWalk)
            .await
            .unwrap_err();
        assert_eq!(error.code, FsErrorCode::NotDirectory);
        assert_eq!(error.path, Some(from_native(&root.join("parent"))));
        let direct = node_mkdirp_with(&root.join("parent"), &FailedStatWalk)
            .await
            .unwrap_err();
        assert_eq!(direct.code, FsErrorCode::NotFound);
        assert_eq!(direct.path, Some(from_native(&root.join("parent"))));
    }

    #[test]
    fn host_readability_errors_never_report_missing_provider_capability() {
        let cases = [
            (io::ErrorKind::NotFound, FsErrorCode::NotFound),
            (io::ErrorKind::NotADirectory, FsErrorCode::NotDirectory),
            (
                io::ErrorKind::PermissionDenied,
                FsErrorCode::PermissionDenied,
            ),
            (io::ErrorKind::InvalidInput, FsErrorCode::Invalid),
            (io::ErrorKind::Other, FsErrorCode::Unknown),
            (io::ErrorKind::Unsupported, FsErrorCode::Unknown),
        ];
        for (kind, expected) in cases {
            assert_eq!(map_readability_error(io::Error::from(kind)).code, expected);
        }
    }

    #[test]
    fn host_read_write_errors_preserve_the_combined_querys_error_class() {
        let cases = [
            (io::ErrorKind::NotFound, FsErrorCode::NotFound),
            (io::ErrorKind::NotADirectory, FsErrorCode::NotDirectory),
            (
                io::ErrorKind::PermissionDenied,
                FsErrorCode::PermissionDenied,
            ),
            (io::ErrorKind::InvalidInput, FsErrorCode::Invalid),
            (io::ErrorKind::Other, FsErrorCode::Unknown),
            (io::ErrorKind::Unsupported, FsErrorCode::Unknown),
        ];
        for (kind, expected) in cases {
            assert_eq!(map_read_write_error(io::Error::from(kind)).code, expected);
        }
    }

    #[cfg(unix)]
    #[test]
    fn combined_access_calls_the_host_once_and_keeps_its_errno() {
        let mut calls = 0;
        let result = check_posix_read_write_with(Path::new("/witness"), |path| {
            calls += 1;
            assert_eq!(path, Path::new("/witness"));
            Err(io::Error::from_raw_os_error(
                nix::errno::Errno::EROFS as i32,
            ))
        });
        assert_eq!(calls, 1);
        assert_eq!(result.unwrap_err().code, FsErrorCode::Unknown);

        let mut nul_calls = 0;
        let result = check_posix_read_write_with(Path::new("bad\0path"), |_| {
            nul_calls += 1;
            Ok(())
        });
        assert_eq!(nul_calls, 0);
        assert_eq!(result.unwrap_err().code, FsErrorCode::Unknown);
    }

    #[cfg(windows)]
    #[test]
    fn windows_combined_access_calls_one_probe_and_rejects_nul_first() {
        let mut calls = 0;
        let result = check_windows_read_write_with(Path::new("C:\\witness"), |path| {
            calls += 1;
            assert_eq!(path, Path::new("C:\\witness"));
            Err(io::Error::from(io::ErrorKind::PermissionDenied))
        });
        assert_eq!(calls, 1);
        assert_eq!(result.unwrap_err().code, FsErrorCode::PermissionDenied);

        let mut nul_calls = 0;
        let result = check_windows_read_write_with(Path::new("bad\0path"), |_| {
            nul_calls += 1;
            Ok(())
        });
        assert_eq!(nul_calls, 0);
        assert_eq!(result.unwrap_err().code, FsErrorCode::Unknown);
    }

    #[cfg(unix)]
    #[test]
    fn posix_native_errno_classification_keeps_enosys_out_of_not_supported() {
        use nix::errno::Errno;
        let cases = [
            (Errno::ENOENT, FsErrorCode::NotFound),
            (Errno::ENOTDIR, FsErrorCode::NotDirectory),
            (Errno::EACCES, FsErrorCode::PermissionDenied),
            (Errno::ELOOP, FsErrorCode::Unknown),
            (Errno::EIO, FsErrorCode::Unknown),
            (Errno::EINVAL, FsErrorCode::Invalid),
            (Errno::ENOSYS, FsErrorCode::Unknown),
        ];
        for (errno, expected) in cases {
            assert_eq!(
                map_readability_error(io::Error::from_raw_os_error(errno as i32)).code,
                expected,
                "{errno}"
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_call_not_implemented_is_unknown_not_missing_capability() {
        assert_eq!(
            map_readability_error(io::Error::from_raw_os_error(120)).code,
            FsErrorCode::Unknown
        );
    }

    #[cfg(unix)]
    #[test]
    fn target_removed_at_access_boundary_preserves_not_found_and_one_query() {
        use std::cell::Cell;

        let root = env::temp_dir().join(format!("minion-access-race-{}", Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let target = root.join("target");
        std::fs::write(&target, b"content").unwrap();
        let calls = Cell::new(0);
        let result = check_posix_readable_with(&target, |path| {
            calls.set(calls.get() + 1);
            std::fs::remove_file(path).unwrap();
            nix::unistd::access(path, nix::unistd::AccessFlags::R_OK)
                .map_err(|errno| io::Error::from_raw_os_error(errno as i32))
        });
        assert_eq!(calls.get(), 1);
        assert_eq!(result.unwrap_err().code, FsErrorCode::NotFound);
        std::fs::remove_dir(root).unwrap();
    }

    #[derive(Debug, Default)]
    struct CountingDirectoryProbeOperations {
        read_dir_calls: AtomicUsize,
        symlink_metadata_calls: AtomicUsize,
        metadata_calls: AtomicUsize,
    }

    #[derive(Debug, Default)]
    struct CountingTokioDirectoryProbeOperations {
        inner: TokioDirectoryProbeOperations,
        read_dir_calls: AtomicUsize,
        symlink_metadata_calls: AtomicUsize,
        metadata_calls: AtomicUsize,
    }

    #[async_trait]
    impl DirectoryProbeOperations for CountingTokioDirectoryProbeOperations {
        async fn read_dir_names(&self, path: &Path) -> io::Result<Vec<String>> {
            self.read_dir_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.read_dir_names(path).await
        }

        async fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
            self.symlink_metadata_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.symlink_metadata(path).await
        }

        async fn metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
            self.metadata_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.metadata(path).await
        }
    }

    #[cfg(unix)]
    fn broken_symlink(target: &str, link: &Path) {
        std::os::unix::fs::symlink(target, link).unwrap();
    }

    #[cfg(windows)]
    fn broken_symlink(target: &str, link: &Path) {
        std::os::windows::fs::symlink_file(target, link).unwrap();
    }

    #[async_trait]
    impl DirectoryProbeOperations for CountingDirectoryProbeOperations {
        async fn read_dir_names(&self, _path: &Path) -> io::Result<Vec<String>> {
            self.read_dir_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec!["z_raw".to_owned(), "a_raw".to_owned()])
        }

        async fn symlink_metadata(&self, _path: &Path) -> io::Result<std::fs::Metadata> {
            self.symlink_metadata_calls.fetch_add(1, Ordering::SeqCst);
            Err(io::Error::other(
                "list_dir_raw must not inspect entry metadata",
            ))
        }

        async fn metadata(&self, _path: &Path) -> io::Result<std::fs::Metadata> {
            self.metadata_calls.fetch_add(1, Ordering::SeqCst);
            Err(io::Error::other(
                "list_dir_raw must not follow entry metadata",
            ))
        }
    }

    const ADA_292_ORACLE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt"
    ));

    #[test]
    fn exact_ada_292_host_oracle_matches_all_8246_cases() {
        let mut count = 0;
        for row in ADA_292_ORACLE.lines() {
            let (raw, expected) = row.split_once('\t').expect("oracle row has two fields");
            let actual = AdaUrl::parse(raw, None)
                .ok()
                .and_then(|url| unicode_host(&url));
            if expected == "PARSE_ERROR" {
                assert_eq!(actual, None, "{raw}");
            } else {
                assert_eq!(actual.as_deref(), Some(expected), "{raw}");
            }
            count += 1;
        }
        assert_eq!(count, 8_246);
    }

    #[test]
    fn file_url_conversion_applies_node_platform_rules_after_ada_parsing() {
        assert_eq!(
            file_url_to_path("file://xn--bcher-kva/share", true).as_deref(),
            Some("\\\\bücher\\share")
        );
        assert_eq!(
            file_url_to_path("file://localhost/C:/a%20b", true).as_deref(),
            Some("C:\\a b")
        );
        assert_eq!(
            file_url_to_path("file:///C%3A/foo", true).as_deref(),
            Some("C:\\foo")
        );
        assert_eq!(file_url_to_path("file:///C:/a%2Fb", true), None);
        assert_eq!(file_url_to_path("file:///C:/a%5Cb", true), None);
        assert_eq!(
            file_url_to_path("file:///home/user/a%20b", false).as_deref(),
            Some("/home/user/a b")
        );
        assert_eq!(file_url_to_path("file://remote/share", false), None);
        assert_eq!(file_url_to_path("file:///%ZZ", true), None);
    }

    #[tokio::test]
    async fn raw_listing_invokes_only_enumeration_and_zero_per_entry_probes() {
        let operations = Arc::new(CountingDirectoryProbeOperations::default());
        let filesystem = LocalFileSystem {
            cwd: PathBuf::from("root"),
            provider_id: Uuid::new_v4(),
            world: ExecutionWorldIdentity::local(),
            directory_probe_operations: operations.clone(),
        };

        assert_eq!(
            filesystem.list_dir_raw(".", None).await.unwrap(),
            ["z_raw", "a_raw"]
        );
        assert_eq!(operations.read_dir_calls.load(Ordering::SeqCst), 1);
        assert_eq!(operations.symlink_metadata_calls.load(Ordering::SeqCst), 0);
        assert_eq!(operations.metadata_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn real_tokio_raw_listing_invokes_zero_probe_operations() {
        let root = env::temp_dir().join(format!("minion-raw-list-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("ordinary"), "content").unwrap();
        broken_symlink("missing-target", &root.join("broken-link"));

        let operations = Arc::new(CountingTokioDirectoryProbeOperations::default());
        let filesystem = LocalFileSystem {
            cwd: root.clone(),
            provider_id: Uuid::new_v4(),
            world: ExecutionWorldIdentity::local(),
            directory_probe_operations: operations.clone(),
        };

        let names = filesystem.list_dir_raw(".", None).await.unwrap();
        assert!(names.iter().any(|name| name == "ordinary"));
        assert!(names.iter().any(|name| name == "broken-link"));
        assert_eq!(operations.read_dir_calls.load(Ordering::SeqCst), 1);
        assert_eq!(operations.symlink_metadata_calls.load(Ordering::SeqCst), 0);
        assert_eq!(operations.metadata_calls.load(Ordering::SeqCst), 0);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concrete_tokio_raw_enumerator_has_no_direct_metadata_probe() {
        let source = include_str!("filesystem.rs");
        let implementation = source
            .split_once("impl DirectoryProbeOperations for TokioDirectoryProbeOperations {")
            .unwrap()
            .1;
        let raw_enumerator = implementation
            .split_once("async fn read_dir_names")
            .unwrap()
            .1
            .split_once("async fn symlink_metadata")
            .unwrap()
            .0;

        assert!(!raw_enumerator.contains("symlink_metadata("));
        assert!(!raw_enumerator.contains("metadata("));
    }
}
