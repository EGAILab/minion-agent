//! Agent tool definitions and the Runtime-scoped registry.
//!
//! Layer 05 represents capabilities and metadata but never invokes tool
//! preparation or execution; invocation belongs to Layer 06.

mod definition;
mod registry;

pub use definition::*;
pub use registry::ToolRegistry;
