//! Agent tool definitions and the Runtime-scoped registry.
//!
//! Layer 05 represents capabilities and metadata but never invokes tool
//! preparation or execution; invocation belongs to Layer 06.

pub mod builtin;
mod definition;
mod execution;
mod prepared;
mod prepared_validation;
mod registry;
mod schema;

pub use definition::*;
pub use execution::*;
pub use prepared::{NonJsonPreparedValue, PreparedNumber, PreparedString, PreparedValue};
pub use registry::ToolRegistry;
pub use schema::{RuntimeSchemaError, RuntimeSchemaObject};
