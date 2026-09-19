use std::{fmt, sync::Arc};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct ExecutionWorldIdentity(Arc<str>);

impl ExecutionWorldIdentity {
    pub fn fresh() -> Self {
        Self(Arc::from(Uuid::new_v4().to_string()))
    }

    pub fn local() -> Self {
        Self(Arc::from("minion.local"))
    }
}

impl fmt::Debug for ExecutionWorldIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ExecutionWorldIdentity(<opaque>)")
    }
}

pub fn compatible(left: &ExecutionWorldIdentity, right: &ExecutionWorldIdentity) -> bool {
    left == right
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IncompatiblePair {
    pub left: String,
    pub right: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionWorldError {
    pub incompatible_pairs: Vec<IncompatiblePair>,
}

impl fmt::Display for ExecutionWorldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("execution capabilities belong to incompatible worlds")
    }
}

impl std::error::Error for ExecutionWorldError {}

pub fn validate_execution_worlds(
    providers: &[(&str, &ExecutionWorldIdentity)],
) -> Result<(), ExecutionWorldError> {
    debug_assert!(
        providers
            .iter()
            .enumerate()
            .all(|(index, (name, _))| providers[..index].iter().all(|(seen, _)| seen != name)),
        "execution-world labels must be unique"
    );
    let mut incompatible_pairs = Vec::new();
    for (left_index, (left_name, left_identity)) in providers.iter().enumerate() {
        for (right_name, right_identity) in &providers[left_index + 1..] {
            if !compatible(left_identity, right_identity) {
                incompatible_pairs.push(IncompatiblePair {
                    left: (*left_name).to_owned(),
                    right: (*right_name).to_owned(),
                });
            }
        }
    }
    if incompatible_pairs.is_empty() {
        Ok(())
    } else {
        Err(ExecutionWorldError { incompatible_pairs })
    }
}
