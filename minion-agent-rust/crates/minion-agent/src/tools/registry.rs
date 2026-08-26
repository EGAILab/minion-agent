use std::{collections::HashSet, sync::Arc};

use crate::{
    RegistrationHandle, RuntimeError, ScopeHandle, ScopeTree, ScopedRegistry, llm::ToolSchema,
};

use super::ToolDefinition;

/// The authoritative scope-aware tool surface.
///
/// Scope ancestry, liveness, registration order, and disposal remain owned by
/// the certified Runtime primitives. This wrapper only composes visible tools
/// by their string names.
#[derive(Clone)]
pub struct ToolRegistry {
    entries: Arc<ScopedRegistry<ToolDefinition>>,
}

impl ToolRegistry {
    pub fn new(tree: impl Into<ScopeTree>) -> Self {
        Self {
            entries: Arc::new(ScopedRegistry::new(tree)),
        }
    }

    pub fn register_for_scope(
        &self,
        owner: Option<&ScopeHandle>,
        tool: ToolDefinition,
    ) -> Result<RegistrationHandle, RuntimeError> {
        self.entries.register(owner, tool)
    }

    pub fn visible(&self, request: Option<&ScopeHandle>) -> Vec<Arc<ToolDefinition>> {
        let mut names = HashSet::new();
        self.entries
            .visible_from_scope(request)
            .into_iter()
            .filter(|tool| names.insert(tool.name().to_owned()))
            .collect()
    }

    pub fn resolve(
        &self,
        name: &str,
        request: Option<&ScopeHandle>,
    ) -> Option<Arc<ToolDefinition>> {
        self.visible(request)
            .into_iter()
            .find(|tool| tool.name() == name)
    }

    pub fn schemas(&self, request: Option<&ScopeHandle>) -> Vec<ToolSchema> {
        self.visible(request)
            .into_iter()
            .map(|tool| tool.schema())
            .collect()
    }
}
