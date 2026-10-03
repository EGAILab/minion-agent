mod environment;
mod error;
mod filesystem;
pub(crate) mod path;
mod service;
mod shell;
mod signal;
mod subprocess;
mod world;

pub use environment::{EnvEntries, EnvSnapshot, EnvValue, Platform, pinned_windows_uppercase};
pub use error::{
    FsError, FsErrorCode, ShellError, ShellErrorCode, SubprocessError, SubprocessErrorCode,
};
pub use filesystem::{
    DirEntryProbe, DirEntryProbeKind, FileInfo, FileKind, FileSystem, FsTarget, LocalFileSystem,
    TargetKey, file_url_to_path,
};
pub use path::{FsPath, file_url_to_js_path};
pub use service::{FileSystemService, LocalExecutionProviders, ShellService, SubprocessService};
pub use shell::{LocalShell, Shell, ShellExecOptions, ShellOutput, StreamCallback};
pub use signal::{AbortSignal, CancellationController, CancellationSignal};
pub use subprocess::{
    ExitStatus, LocalProcess, LocalSubprocess, Process, ReadableStream, SpawnOptions, StdioMode,
    Subprocess, WritableStream,
};
pub use world::{
    ExecutionWorldError, ExecutionWorldIdentity, IncompatiblePair, compatible,
    validate_execution_worlds,
};
