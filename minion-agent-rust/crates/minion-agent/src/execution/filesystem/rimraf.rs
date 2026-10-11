//! Node v22.15.1 validateRmOptions + rimraf, expressed as an explicit work stack.
//! Errors retain the actual failing native call; the original error is carried
//! only where Node's _rmdir / fixWinEPERM explicitly requires it.
use super::*;

#[async_trait]
pub(super) trait Operations: Debug + Send + Sync {
    async fn lstat(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        tokio::fs::symlink_metadata(path).await
    }
    async fn stat(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        tokio::fs::metadata(path).await
    }
    async fn unlink(&self, path: &Path) -> io::Result<()> {
        #[cfg(windows)]
        {
            let path = path.to_owned();
            tokio::task::spawn_blocking(move || minion_agent_native_fs::delete_entry(&path, false))
                .await
                .map_err(io::Error::other)?
        }
        #[cfg(not(windows))]
        {
            tokio::fs::remove_file(path).await
        }
    }
    async fn rmdir(&self, path: &Path) -> io::Result<()> {
        #[cfg(windows)]
        {
            let path = path.to_owned();
            tokio::task::spawn_blocking(move || minion_agent_native_fs::delete_entry(&path, true))
                .await
                .map_err(io::Error::other)?
        }
        #[cfg(not(windows))]
        {
            tokio::fs::remove_dir(path).await
        }
    }
    async fn readdir(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        #[cfg(windows)]
        {
            let addressed = path.to_owned();
            let names =
                tokio::task::spawn_blocking(move || minion_agent_native_fs::scandir(&addressed))
                    .await
                    .map_err(io::Error::other)??;
            Ok(names.into_iter().map(|name| path.join(name)).collect())
        }
        #[cfg(not(windows))]
        {
            let mut dir = tokio::fs::read_dir(path).await?;
            let mut children = Vec::new();
            while let Some(entry) = dir.next_entry().await? {
                children.push(entry.path());
            }
            Ok(children)
        }
    }
    async fn chmod(&self, path: &Path) -> io::Result<()> {
        #[cfg(windows)]
        {
            let path = path.to_owned();
            tokio::task::spawn_blocking(move || {
                minion_agent_native_fs::clear_readonly_entry(&path).map(|_| ())
            })
            .await
            .map_err(io::Error::other)?
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            Ok(())
        }
    }
}

#[derive(Debug)]
pub(super) struct NativeOperations;
impl Operations for NativeOperations {}

fn missing(e: &io::Error) -> bool {
    map_fs_error(io::Error::from_raw_os_error(e.raw_os_error().unwrap_or(-1))).code
        == FsErrorCode::NotFound
        || (e.raw_os_error().is_none() && e.kind() == io::ErrorKind::NotFound)
}
fn eperm(e: &io::Error) -> bool {
    #[cfg(windows)]
    {
        matches!(e.raw_os_error(), Some(5 | 1314))
    }
    #[cfg(not(windows))]
    {
        e.raw_os_error() == Some(1)
    }
}
fn enotdir(e: &io::Error) -> bool {
    // libuv has no generic Windows ENOTDIR. Only explicit binding errors have it.
    #[cfg(windows)]
    {
        e.raw_os_error().is_none() && e.kind() == io::ErrorKind::NotADirectory
    }
    #[cfg(not(windows))]
    {
        e.kind() == io::ErrorKind::NotADirectory
    }
}
fn directory_retry(e: &io::Error) -> bool {
    eperm(e)
        || matches!(
            e.kind(),
            io::ErrorKind::DirectoryNotEmpty | io::ErrorKind::AlreadyExists
        )
}

enum Work {
    Entry(PathBuf),
    Directory(PathBuf, Option<io::Error>),
    FinalDirectory(PathBuf),
}

pub(super) async fn remove(
    path: &Path,
    recursive: bool,
    force: bool,
    ops: &dyn Operations,
) -> Result<(), FsError> {
    match ops.lstat(path).await {
        Ok(m) if m.is_dir() && !m.file_type().is_symlink() && !recursive => {
            return Err(FsError::new(
                FsErrorCode::Unknown,
                "Path is a directory: rm requires recursive",
            ));
        }
        Ok(_) => (),
        Err(e) if force && missing(&e) => (),
        Err(e) => return Err(native_error(e, path)),
    }
    let mut stack = vec![Work::Entry(path.to_owned())];
    while let Some(work) = stack.pop() {
        match work {
            Work::Entry(p) => {
                let classification = ops.lstat(&p).await;
                match classification {
                    Ok(m) if m.is_dir() && !m.file_type().is_symlink() => {
                        stack.push(Work::Directory(p, None));
                        continue;
                    }
                    Err(e) if missing(&e) => continue,
                    Err(e) if cfg!(windows) && eperm(&e) => {
                        recover(&p, e, ops, &mut stack).await?;
                        continue;
                    }
                    _ => (),
                }
                match ops.unlink(&p).await {
                    Ok(()) => (),
                    Err(e) if missing(&e) => (),
                    Err(e) if cfg!(windows) && eperm(&e) => recover(&p, e, ops, &mut stack).await?,
                    Err(e)
                        if e.kind() == io::ErrorKind::IsADirectory
                            || (!cfg!(windows) && eperm(&e)) =>
                    {
                        stack.push(Work::Directory(p, Some(e)))
                    }
                    Err(e) => return Err(native_error(e, &p)),
                }
            }
            Work::Directory(p, original) => match ops.rmdir(&p).await {
                Ok(()) => (),
                Err(e) if missing(&e) => (),
                Err(e) if enotdir(&e) => {
                    if let Some(e) = original {
                        return Err(native_error(e, &p));
                    }
                }
                Err(e) if directory_retry(&e) => {
                    let children = match ops.readdir(&p).await {
                        Ok(children) => children,
                        Err(e) if missing(&e) => continue,
                        Err(e) => return Err(native_error(e, &p)),
                    };
                    stack.push(Work::FinalDirectory(p));
                    // Reverse push, not reverse processing: native enumeration order survives.
                    stack.extend(children.into_iter().rev().map(Work::Entry));
                }
                Err(e) => return Err(native_error(e, &p)),
            },
            Work::FinalDirectory(p) => match ops.rmdir(&p).await {
                Ok(()) => (),
                Err(e) if missing(&e) => (),
                Err(e) => return Err(native_error(e, &p)),
            },
        }
    }
    Ok(())
}

async fn recover(
    p: &Path,
    original: io::Error,
    ops: &dyn Operations,
    stack: &mut Vec<Work>,
) -> Result<(), FsError> {
    if let Err(e) = ops.chmod(p).await {
        return if missing(&e) {
            Ok(())
        } else {
            Err(native_error(original, p))
        };
    }
    match ops.stat(p).await {
        Ok(m) if m.is_dir() => {
            stack.push(Work::Directory(p.to_owned(), Some(original)));
            Ok(())
        }
        Ok(_) => match ops.unlink(p).await {
            Ok(()) => Ok(()),
            Err(e) if missing(&e) => Ok(()),
            Err(e) => Err(native_error(e, p)),
        },
        Err(e) if missing(&e) => Ok(()),
        Err(_) => Err(native_error(original, p)),
    }
}

#[cfg(test)]
mod tests;
