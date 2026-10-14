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
    /// Actual failing OS-call path, or the logical operation fallback.
    pub path: Option<super::FsPath>,
}

impl FsError {
    pub fn new(code: FsErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path: None,
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

/// Legacy scalar diagnostics remain available in `message`; EXEC-013 diagnostics
/// additionally retain their exact JavaScript string, including a split surrogate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubprocessError {
    pub code: SubprocessErrorCode,
    pub message: String,
    nul_diagnostic: Option<crate::llm::ResultString>,
}

impl SubprocessError {
    pub fn new(code: SubprocessErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            nul_diagnostic: None,
        }
    }

    pub(super) fn nul_input(message: crate::llm::ResultString) -> Self {
        Self {
            code: SubprocessErrorCode::SpawnError,
            message: String::from_utf16_lossy(message.code_units()),
            nul_diagnostic: Some(message),
        }
    }

    /// Typed rejection identity, never inferred from an OS diagnostic's text.
    pub fn nul_diagnostic(&self) -> Option<&crate::llm::ResultString> {
        self.nul_diagnostic.as_ref()
    }
}

impl fmt::Display for SubprocessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}
impl std::error::Error for SubprocessError {}
