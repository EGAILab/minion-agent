use futures::executor::block_on;
use minion_agent::{
    ScopeTree,
    tools::{ToolDefinition, ToolExecutionRequest, ToolRegistry},
};
use serde_json::json;

fn tool(name: &str, label: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        format!("{name} description"),
        serde_json::from_value(json!({"type": "object", "properties": {}})).unwrap(),
        label,
        |_request: ToolExecutionRequest| Box::pin(async { unreachable!() }),
    )
}

fn names(registry: &ToolRegistry, scope: Option<&minion_agent::ScopeHandle>) -> Vec<String> {
    registry
        .visible(scope)
        .into_iter()
        .map(|tool| tool.name().to_owned())
        .collect()
}

#[test]
fn visibility_is_nearest_first_without_descendant_or_sibling_leaks() {
    let tree = ScopeTree::new();
    let parent = tree.create_root();
    let child = tree.create_child(&parent).unwrap();
    let sibling = tree.create_child(&parent).unwrap();
    let registry = ToolRegistry::new(tree);
    registry
        .register_for_scope(None, tool("global", "Global"))
        .unwrap();
    registry
        .register_for_scope(Some(&parent), tool("parent", "Parent"))
        .unwrap();
    registry
        .register_for_scope(Some(&child), tool("child", "Child"))
        .unwrap();
    registry
        .register_for_scope(Some(&sibling), tool("sibling", "Sibling"))
        .unwrap();

    assert_eq!(names(&registry, None), ["global"]);
    assert_eq!(names(&registry, Some(&parent)), ["parent", "global"]);
    assert_eq!(
        names(&registry, Some(&child)),
        ["child", "parent", "global"]
    );
    assert_eq!(
        names(&registry, Some(&sibling)),
        ["sibling", "parent", "global"]
    );
}

#[test]
fn nearest_and_earliest_registration_wins_with_live_fallback() {
    let tree = ScopeTree::new();
    let parent = tree.create_root();
    let child = tree.create_child(&parent).unwrap();
    let registry = ToolRegistry::new(tree);
    registry
        .register_for_scope(Some(&parent), tool("read", "Parent"))
        .unwrap();
    let first = registry
        .register_for_scope(Some(&child), tool("read", "Child first"))
        .unwrap();
    registry
        .register_for_scope(Some(&child), tool("read", "Child second"))
        .unwrap();

    assert_eq!(
        registry.resolve("read", Some(&child)).unwrap().label(),
        "Child first"
    );
    assert_eq!(names(&registry, Some(&child)), ["read"]);
    block_on(first.dispose()).unwrap();
    assert_eq!(
        registry.resolve("read", Some(&child)).unwrap().label(),
        "Child second"
    );
}

#[test]
fn unknown_and_disposed_requesters_observe_nothing() {
    let tree = ScopeTree::new();
    let scope = tree.create_root();
    let registry = ToolRegistry::new(tree);
    registry
        .register_for_scope(None, tool("global", "Global"))
        .unwrap();
    registry
        .register_for_scope(Some(&scope), tool("local", "Local"))
        .unwrap();

    assert!(registry.resolve("missing", Some(&scope)).is_none());
    block_on(scope.dispose()).unwrap();
    assert!(registry.visible(Some(&scope)).is_empty());
    assert!(registry.resolve("global", Some(&scope)).is_none());
}

#[test]
fn withdrawal_is_idempotent_and_restores_farther_registration() {
    let tree = ScopeTree::new();
    let scope = tree.create_root();
    let registry = ToolRegistry::new(tree);
    registry
        .register_for_scope(None, tool("read", "Global"))
        .unwrap();
    let local = registry
        .register_for_scope(Some(&scope), tool("read", "Local"))
        .unwrap();

    block_on(local.dispose()).unwrap();
    block_on(local.dispose()).unwrap();
    assert_eq!(
        registry.resolve("read", Some(&scope)).unwrap().label(),
        "Global"
    );
}
