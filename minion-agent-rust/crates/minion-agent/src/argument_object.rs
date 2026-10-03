//! K1 argument-object storage. Mutation and iteration share one ordering rule.
//! This is deliberately not used for tool-result details or provider metadata.

use std::{hash::Hash, ops::Index};

use indexmap::IndexMap;

/// Lossless UTF-16 keys used by the certified raw and prepared domains.
pub trait ArgumentKey: Clone + Eq + Hash {
    fn code_units(&self) -> &[u16];
}

impl ArgumentKey for crate::javascript::JsString {
    fn code_units(&self) -> &[u16] {
        self.code_units()
    }
}

impl ArgumentKey for crate::tools::PreparedString {
    fn code_units(&self) -> &[u16] {
        self.code_units()
    }
}

/// ECMAScript array-index recognition, total even for arbitrarily long keys.
pub fn array_index(units: &[u16]) -> Option<u32> {
    if units.is_empty() || units.len() > 10 || (units.len() > 1 && units[0] == 48) {
        return None;
    }
    let mut number = 0_u64;
    for &unit in units {
        if !(48..=57).contains(&unit) {
            return None;
        }
        number = number * 10 + u64::from(unit - 48);
    }
    (number < u64::from(u32::MAX)).then_some(number as u32)
}

/// An object whose own keys always enumerate index-first, then insertion-order.
/// Replacing an existing property retains its position; deletion and reinsertion
/// give an ordinary property a new position. No mutable map escape hatch exists.
#[derive(Clone, Debug, PartialEq)]
pub struct ArgumentObject<K: ArgumentKey, V>(IndexMap<K, V>);

impl<K: ArgumentKey, V> Default for ArgumentObject<K, V> {
    fn default() -> Self {
        Self(IndexMap::new())
    }
}

impl<K: ArgumentKey, V> ArgumentObject<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let previous = self.0.insert(key, value);
        // Stable sorting keeps every non-index property's insertion position.
        self.0.sort_by(|a, _, b, _| {
            match (array_index(a.code_units()), array_index(b.code_units())) {
                (Some(a), Some(b)) => a.cmp(&b),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });
        previous
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.0.get(key)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.0.get_mut(key)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.0.contains_key(key)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.0.shift_remove(key)
    }

    pub fn entry_or_insert(&mut self, key: K, value: V) -> &mut V {
        if !self.0.contains_key(&key) {
            self.insert(key.clone(), value);
        }
        self.0.get_mut(&key).expect("existing property")
    }

    pub fn iter(&self) -> indexmap::map::Iter<'_, K, V> {
        self.0.iter()
    }

    pub fn keys(&self) -> indexmap::map::Keys<'_, K, V> {
        self.0.keys()
    }

    pub fn values(&self) -> indexmap::map::Values<'_, K, V> {
        self.0.values()
    }
}

impl<K: ArgumentKey, V> FromIterator<(K, V)> for ArgumentObject<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut object = Self::new();
        for (key, value) in iter {
            object.insert(key, value);
        }
        object
    }
}

impl<K: ArgumentKey, V, const N: usize> From<[(K, V); N]> for ArgumentObject<K, V> {
    fn from(entries: [(K, V); N]) -> Self {
        entries.into_iter().collect()
    }
}

impl<K: ArgumentKey, V> IntoIterator for ArgumentObject<K, V> {
    type Item = (K, V);
    type IntoIter = indexmap::map::IntoIter<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, K: ArgumentKey, V> IntoIterator for &'a ArgumentObject<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = indexmap::map::Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: ArgumentKey, V> Index<&K> for ArgumentObject<K, V> {
    type Output = V;
    fn index(&self, key: &K) -> &V {
        &self.0[key]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::javascript::JsString;

    #[test]
    fn recognition_is_canonical_ascii_and_total() {
        for (text, expected) in [
            ("0", Some(0)),
            ("4294967294", Some(4294967294)),
            ("01", None),
            ("4294967295", None),
            ("-0", None),
            ("١", None),
            ("", None),
        ] {
            assert_eq!(
                array_index(&text.encode_utf16().collect::<Vec<_>>()),
                expected
            );
        }
        assert_eq!(array_index(&vec![49; 100_000]), None);
    }

    #[test]
    fn every_mutation_keeps_index_first_and_ordinary_insertion_order() {
        let mut object = ArgumentObject::<JsString, i32>::from([
            ("z".into(), 1),
            ("2".into(), 2),
            ("1".into(), 3),
            ("a".into(), 4),
        ]);
        object.insert("0".into(), 5);
        object.insert("z".into(), 6);
        object.remove(&"a".into());
        object.insert("b".into(), 7);
        object.insert("a".into(), 8);
        assert_eq!(
            object
                .keys()
                .map(|k| k.to_string().unwrap())
                .collect::<Vec<_>>(),
            ["0", "1", "2", "z", "b", "a"]
        );
        assert_eq!(object.get(&"z".into()), Some(&6));
    }
}
