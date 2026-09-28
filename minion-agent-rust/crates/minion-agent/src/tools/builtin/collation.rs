//! The pinned ICU binding is isolated because the core crate forbids unsafe code.

use crate::tools::ToolCapabilityError;

pub(super) fn sort_names(names: Vec<String>) -> Result<Vec<String>, ToolCapabilityError> {
    minion_agent_pinned_icu::sort_names(names).map_err(ToolCapabilityError::new)
}
