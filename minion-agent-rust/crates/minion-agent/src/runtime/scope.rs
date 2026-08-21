use std::{
    cmp::Reverse,
    collections::HashMap,
    sync::{Arc, Weak},
};

use futures::future::{BoxFuture, FutureExt, Shared};
use parking_lot::Mutex;

use super::{DisposeErrors, EffectStore, RuntimeError};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScopeId(u64);

impl ScopeId {
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone)]
pub struct ScopeTree {
    state: Arc<Mutex<ScopeTreeState>>,
}

struct ScopeTreeState {
    next_id: Option<u64>,
    scopes: HashMap<ScopeId, ScopeNode>,
    in_flight_disposals: HashMap<ScopeId, InFlightScopeDisposal>,
    next_disposal_generation: u64,
}

type ScopeUnwind = Shared<BoxFuture<'static, Result<(), DisposeErrors>>>;

struct InFlightScopeDisposal {
    generation: u64,
    unwind: ScopeUnwind,
}

struct ScopeNode {
    parent: Option<ScopeId>,
    children: Vec<ScopeId>,
    depth: usize,
    active: bool,
    effects: Arc<EffectStore>,
}

#[derive(Clone)]
pub struct ScopeHandle {
    state: Arc<Mutex<ScopeTreeState>>,
    id: ScopeId,
    effects: Arc<EffectStore>,
}

impl ScopeTree {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(ScopeTreeState {
                next_id: Some(0),
                scopes: HashMap::new(),
                in_flight_disposals: HashMap::new(),
                next_disposal_generation: 0,
            })),
        }
    }

    pub fn create_root(&self) -> ScopeHandle {
        self.create_scope(None)
            .expect("a root scope never has an inactive parent")
    }

    pub fn create_child(&self, parent: &ScopeHandle) -> Result<ScopeHandle, RuntimeError> {
        if !Arc::ptr_eq(&self.state, &parent.state) {
            return Err(RuntimeError::InactiveOwner {
                owner: "scope from another tree".to_owned(),
            });
        }
        self.create_scope(Some(parent.id))
    }

    pub fn create_scope(&self, parent: Option<ScopeId>) -> Result<ScopeHandle, RuntimeError> {
        let effects = Arc::new(EffectStore::new());
        let mut state = self.state.lock();
        let depth = match parent {
            Some(parent) => state
                .scopes
                .get(&parent)
                .filter(|scope| scope.active)
                .map(|scope| scope.depth + 1)
                .ok_or_else(|| RuntimeError::InactiveOwner {
                    owner: format!("scope {}", parent.as_u64()),
                })?,
            None => 0,
        };
        let id = ScopeId(
            state
                .next_id
                .expect("scope ID space exhausted; IDs must never be reused"),
        );
        state.next_id = id.0.checked_add(1);
        state.scopes.insert(
            id,
            ScopeNode {
                parent,
                children: Vec::new(),
                depth,
                active: true,
                effects: Arc::clone(&effects),
            },
        );
        if let Some(parent) = parent {
            state
                .scopes
                .get_mut(&parent)
                .expect("the active parent was checked before insertion")
                .children
                .push(id);
        }
        Ok(ScopeHandle {
            state: Arc::clone(&self.state),
            id,
            effects,
        })
    }

    pub fn is_ancestor(&self, ancestor: ScopeId, descendant: ScopeId) -> bool {
        let state = self.state.lock();
        let mut current = state.scopes.get(&descendant).and_then(|scope| scope.parent);
        while let Some(scope) = current {
            if scope == ancestor {
                return true;
            }
            current = state.scopes.get(&scope).and_then(|node| node.parent);
        }
        false
    }

    pub(crate) fn active_ancestor_chain(&self, request: ScopeId) -> Option<Vec<ScopeId>> {
        let state = self.state.lock();
        let mut chain = Vec::new();
        let mut current = Some(request);
        while let Some(scope_id) = current {
            let scope = state.scopes.get(&scope_id)?;
            if !scope.active {
                return None;
            }
            chain.push(scope_id);
            current = scope.parent;
        }
        Some(chain)
    }
}

impl Default for ScopeTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ScopeHandle {
    pub fn id(&self) -> ScopeId {
        self.id
    }

    pub fn effects(&self) -> &EffectStore {
        &self.effects
    }

    pub fn is_active(&self) -> bool {
        self.state
            .lock()
            .scopes
            .get(&self.id)
            .is_some_and(|scope| scope.active)
    }

    pub async fn dispose(&self) -> Result<(), DisposeErrors> {
        let Some(unwind) = ({
            let mut state = self.state.lock();
            if let Some(disposal) = state.in_flight_disposals.get(&self.id) {
                Some(disposal.unwind.clone())
            } else if !state.scopes.get(&self.id).is_some_and(|scope| scope.active) {
                None
            } else {
                let scopes = collect_subtree(&state, self.id);
                for (_, _, effects) in &scopes {
                    effects.close();
                }
                for (id, _, _) in &scopes {
                    state
                        .scopes
                        .get_mut(id)
                        .expect("the scope was collected from this tree")
                        .active = false;
                }
                let generation = state.next_disposal_generation;
                state.next_disposal_generation = state
                    .next_disposal_generation
                    .checked_add(1)
                    .expect("scope disposal generation exhausted");
                let unwind = dispose_scope_generation(
                    scopes,
                    Arc::downgrade(&self.state),
                    self.id,
                    generation,
                )
                .boxed()
                .shared();
                state.in_flight_disposals.insert(
                    self.id,
                    InFlightScopeDisposal {
                        generation,
                        unwind: unwind.clone(),
                    },
                );
                Some(unwind)
            }
        }) else {
            return Ok(());
        };

        unwind.await
    }

    pub(crate) fn belongs_to(&self, tree: &ScopeTree) -> bool {
        Arc::ptr_eq(&self.state, &tree.state)
    }
}

fn collect_subtree(
    state: &ScopeTreeState,
    root: ScopeId,
) -> Vec<(ScopeId, usize, Arc<EffectStore>)> {
    let mut pending = vec![root];
    let mut scopes = Vec::new();
    while let Some(id) = pending.pop() {
        let Some(scope) = state.scopes.get(&id) else {
            continue;
        };
        pending.extend(scope.children.iter().copied());
        scopes.push((id, scope.depth, Arc::clone(&scope.effects)));
    }
    scopes
}

async fn dispose_scope_generation(
    mut scopes: Vec<(ScopeId, usize, Arc<EffectStore>)>,
    state: Weak<Mutex<ScopeTreeState>>,
    root: ScopeId,
    generation: u64,
) -> Result<(), DisposeErrors> {
    scopes.sort_by_key(|(_, depth, _)| Reverse(*depth));
    let mut errors = Vec::new();
    for (_, _, effects) in scopes {
        if let Err(scope_errors) = effects.close_and_dispose().await {
            errors.extend(scope_errors.into_inner());
        }
    }
    let result = if errors.is_empty() {
        Ok(())
    } else {
        Err(DisposeErrors::from_inner(errors))
    };
    if let Some(state) = state.upgrade() {
        let mut state = state.lock();
        if state
            .in_flight_disposals
            .get(&root)
            .is_some_and(|disposal| disposal.generation == generation)
        {
            state.in_flight_disposals.remove(&root);
        }
    }
    result
}
