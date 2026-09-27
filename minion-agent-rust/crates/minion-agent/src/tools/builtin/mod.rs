//! Pinned Pi `read` and `ls` filesystem query tools (WP-13.1).
//!
//! These compose the certified Layer-12 filesystem and Layer-06 tool result seams. Neither
//! message history nor filesystem registration is duplicated here.

mod collation;
mod image;
mod ls;
mod mime;
mod numeric;
mod paths;
mod photon;
mod read;
mod text;
mod truncate;

pub use ls::create_ls_tool;
pub use read::{ReadToolOptions, create_read_tool};

async fn race_abort(
    mut worker: tokio::task::JoinHandle<
        Result<crate::tools::AgentToolResult, crate::tools::ToolCapabilityError>,
    >,
    signal: Option<std::sync::Arc<dyn crate::tools::ToolExecutionSignal>>,
) -> Result<crate::tools::AgentToolResult, crate::tools::ToolCapabilityError> {
    loop {
        // The worker owns its filesystem future. Returning here detaches, never cancels it.
        if worker.is_finished() {
            return worker
                .await
                .map_err(|error| crate::tools::ToolCapabilityError::new(error.to_string()))?;
        }
        if signal.as_ref().is_some_and(|signal| signal.is_cancelled()) {
            return Err(crate::tools::ToolCapabilityError::new(
                paths::OPERATION_ABORTED,
            ));
        }
        tokio::select! {
            result = &mut worker => return result.map_err(|error| crate::tools::ToolCapabilityError::new(error.to_string()))?,
            () = tokio::time::sleep(std::time::Duration::from_millis(1)) => {},
        }
    }
}
