//! Shared argument containers. Cloning a handle keeps object/array identity.
//!
//! Reads return snapshots of handles, not borrowed values held under a lock.
//! Consequently no container lock crosses a tool callback or an async boundary.

use std::sync::Arc;

use parking_lot::RwLock;

use crate::argument_object::{ArgumentKey, ArgumentObject};

#[derive(Debug)]
pub struct ArgumentObjectRef<K: ArgumentKey, V>(Arc<RwLock<ArgumentObject<K, V>>>);

impl<K: ArgumentKey, V> Clone for ArgumentObjectRef<K, V> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<K: ArgumentKey, V> Default for ArgumentObjectRef<K, V> {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(ArgumentObject::new())))
    }
}

impl<K: ArgumentKey, V: Clone> ArgumentObjectRef<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn same_identity(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub fn len(&self) -> usize {
        self.0.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.read().is_empty()
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.0.read().get(key).cloned()
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.0.read().contains_key(key)
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        self.0.write().insert(key, value)
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        self.0.write().remove(key)
    }

    pub fn iter(&self) -> std::vec::IntoIter<(K, V)> {
        self.0
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn keys(&self) -> std::vec::IntoIter<K> {
        self.0
            .read()
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn values(&self) -> std::vec::IntoIter<V> {
        self.0
            .read()
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<K: ArgumentKey, V: Clone> FromIterator<(K, V)> for ArgumentObjectRef<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self(Arc::new(RwLock::new(iter.into_iter().collect())))
    }
}

impl<K: ArgumentKey, V: Clone, const N: usize> From<[(K, V); N]> for ArgumentObjectRef<K, V> {
    fn from(entries: [(K, V); N]) -> Self {
        entries.into_iter().collect()
    }
}

impl<K: ArgumentKey, V: Clone> From<ArgumentObject<K, V>> for ArgumentObjectRef<K, V> {
    fn from(object: ArgumentObject<K, V>) -> Self {
        Self(Arc::new(RwLock::new(object)))
    }
}

impl<K: ArgumentKey, V: Clone> IntoIterator for ArgumentObjectRef<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: ArgumentKey, V: Clone> IntoIterator for &ArgumentObjectRef<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: ArgumentKey, V: Clone + PartialEq> PartialEq for ArgumentObjectRef<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.same_identity(other)
            || (self.len() == other.len()
                && self
                    .iter()
                    .all(|(key, value)| other.get(&key).is_some_and(|other| value == other)))
    }
}

#[derive(Debug)]
pub struct ArgumentArray<V>(Arc<RwLock<Vec<V>>>);

impl<V> Clone for ArgumentArray<V> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<V> Default for ArgumentArray<V> {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(Vec::new())))
    }
}

impl<V: Clone> ArgumentArray<V> {
    pub fn same_identity(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub fn len(&self) -> usize {
        self.0.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.read().is_empty()
    }

    pub fn get(&self, index: usize) -> Option<V> {
        self.0.read().get(index).cloned()
    }

    pub fn push(&self, value: V) {
        self.0.write().push(value);
    }

    pub fn insert(&self, index: usize, value: V) {
        self.0.write().insert(index, value);
    }

    pub fn remove(&self, index: usize) -> V {
        self.0.write().remove(index)
    }

    pub fn set(&self, index: usize, value: V) -> Option<V> {
        let mut values = self.0.write();
        values
            .get_mut(index)
            .map(|slot| std::mem::replace(slot, value))
    }

    pub fn extend(&self, values: impl IntoIterator<Item = V>) {
        self.0.write().extend(values);
    }

    pub fn iter(&self) -> std::vec::IntoIter<V> {
        self.to_vec().into_iter()
    }

    pub fn to_vec(&self) -> Vec<V> {
        self.0.read().clone()
    }
}

impl<V> From<Vec<V>> for ArgumentArray<V> {
    fn from(values: Vec<V>) -> Self {
        Self(Arc::new(RwLock::new(values)))
    }
}

impl<V> FromIterator<V> for ArgumentArray<V> {
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl<V: Clone> IntoIterator for ArgumentArray<V> {
    type Item = V;
    type IntoIter = std::vec::IntoIter<V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<V: Clone + PartialEq> PartialEq for ArgumentArray<V> {
    fn eq(&self, other: &Self) -> bool {
        self.same_identity(other) || self.iter().eq(other.iter())
    }
}
