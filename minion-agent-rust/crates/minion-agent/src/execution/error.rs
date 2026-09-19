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

execution_error!(FsError, FsErrorCode);

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
