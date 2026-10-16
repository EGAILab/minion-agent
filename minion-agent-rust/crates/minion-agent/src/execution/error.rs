use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! execution_error {
    ($name:ident, $code:ident) => {
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct $name {
            pub code: $code,
            pub message: String,
        }

        impl $name {
            pub fn new(code: $code, message: impl Into<String>) -> Self {
                Self {
                    code,
                    message: message.into(),
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}", self.message)
            }
        }

        impl std::error::Error for $name {}
    };
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FsErrorCode {
    Aborted,
    NotFound,
    PermissionDenied,
    NotDirectory,
    IsDirectory,
    Invalid,
    NotSupported,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FsError {
    pub code: FsErrorCode,
    pub message: String,
    /// Actual failing OS-call path, or the logical operation fallback. A failed
    /// process-cwd read instead carries the caller's raw logical argument.
    pub path: Option<super::FsPath>,
    // Origin is private: code/text/path cannot distinguish a failed cwd read
    // from a target lstat/realpath failure with exactly the same OS error.
    pub(super) process_cwd_read: bool,
}

impl FsError {
    pub fn new(code: FsErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path: None,
            process_cwd_read: false,
        }
    }

    pub fn with_path(mut self, path: impl Into<super::FsPath>) -> Self {
        self.path = Some(path.into());
        self
    }
}

impl fmt::Display for FsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}
impl std::error::Error for FsError {}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellErrorCode {
    Aborted,
    Timeout,
    ShellUnavailable,
    SpawnError,
    CallbackError,
    Unknown,
}

execution_error!(ShellError, ShellErrorCode);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SubprocessErrorCode {
    Aborted,
    SpawnError,
    PipeError,
    Unknown,
}

execution_error!(SubprocessError, SubprocessErrorCode);
