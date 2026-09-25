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
    pub name: String,
    pub path: String,
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
    async fn list_dir_raw(
        &self,
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        Err(FsError::new(
            FsErrorCode::NotSupported,
            "list_dir_raw is not supported by this filesystem provider",
        ))
    }
    async fn probe_dir_entry(
        &self,
        _path: &str,
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
        _path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        Err(FsError::new(
            FsErrorCode::NotSupported,
            "check_readable is not supported by this filesystem provider",
        ))
    }
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

    fn resolved(&self, raw: &str) -> PathBuf {
        resolve_local_path(&self.cwd, raw)
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

    async fn list_dir_raw(
        &self,
        path: &str,
        signal: Option<&dyn AbortSignal>,
    ) -> Result<Vec<String>, FsError> {
        Self::aborted(signal)?;
        self.directory_probe_operations
            .read_dir_names(&self.resolved(path))
            .await
            .map_err(map_fs_error)
    }

    async fn probe_dir_entry(
        &self,
        path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<DirEntryProbe, FsError> {
        let path = self.resolved(path);
        let addressed = self
            .directory_probe_operations
            .symlink_metadata(&path)
            .await
            .map_err(map_fs_error)?;
        let file_type = addressed.file_type();
        let kind = if file_type.is_symlink() {
            let target = self
                .directory_probe_operations
                .metadata(&path)
                .await
                .map_err(map_fs_error)?;
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
            name: path
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
            path: path.to_string_lossy().into_owned(),
            kind,
        })
    }

    async fn check_readable(
        &self,
        path: &str,
        _signal: Option<&dyn AbortSignal>,
    ) -> Result<(), FsError> {
        let path = self.resolved(path);
        tokio::task::spawn_blocking(move || check_local_readable(&path))
            .await
            .map_err(|error| FsError::new(FsErrorCode::Unknown, error.to_string()))?
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
        let absolute_path = self.absolute_path(path, signal).await?;
        let target_key = match self.canonical_path(path, signal).await {
            Ok(canonical) => canonical,
            Err(error)
                if matches!(
                    error.code,
                    FsErrorCode::NotFound | FsErrorCode::NotSupported
                ) =>
            {
                absolute_path
            }
            Err(error) => return Err(error),
        };
        Ok(FsTarget {
            provider_id: self.provider_id,
            target_key: TargetKey(Arc::from(target_key.clone())),
            process_path: Arc::from(target_key),
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

fn file_url_to_path(raw: &str, windows: bool) -> Option<String> {
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

#[cfg(unix)]
fn check_local_readable(path: &Path) -> Result<(), FsError> {
    use nix::unistd::{AccessFlags, access};

    if path.as_os_str().as_encoded_bytes().contains(&0) {
        return Err(FsError::new(
            FsErrorCode::Unknown,
            "path contains an embedded NUL byte",
        ));
    }
    // Exactly one access(R_OK): no preliminary stat or content open may change the error site.
    access(path, AccessFlags::R_OK)
        .map_err(|errno| map_fs_error(io::Error::from_raw_os_error(errno as i32)))
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
        .map_err(map_fs_error)
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
