//! K1 argument-object storage. Mutation and iteration share one ordering rule.
//! This is deliberately not used for tool-result details or provider metadata.

use std::{collections::BTreeMap, hash::Hash, ops::Index};

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
pub struct ArgumentObject<K: ArgumentKey, V> {
    // Index insertion is O(log n), including reverse-order input. Inserting into
    // a contiguous index prefix would still shift O(n) entries per insertion.
    indices: BTreeMap<u32, (K, V)>,
    ordinary: IndexMap<K, V>,
}

impl<K: ArgumentKey, V> Default for ArgumentObject<K, V> {
    fn default() -> Self {
        Self {
            indices: BTreeMap::new(),
            ordinary: IndexMap::new(),
        }
    }
}

impl<K: ArgumentKey, V> ArgumentObject<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.indices.len() + self.ordinary.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.ordinary.is_empty()
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(index) = array_index(key.code_units()) {
            match self.indices.entry(index) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((key, value));
                    None
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    Some(std::mem::replace(&mut entry.get_mut().1, value))
                }
            }
        } else {
            self.ordinary.insert(key, value)
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(index) = array_index(key.code_units()) {
            self.indices.get(&index).map(|(_, value)| value)
        } else {
            self.ordinary.get(key)
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if let Some(index) = array_index(key.code_units()) {
            self.indices.get_mut(&index).map(|(_, value)| value)
        } else {
            self.ordinary.get_mut(key)
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(index) = array_index(key.code_units()) {
            self.indices.remove(&index).map(|(_, value)| value)
        } else {
            self.ordinary.shift_remove(key)
        }
    }

    pub fn entry_or_insert(&mut self, key: K, value: V) -> &mut V {
        if !self.contains_key(&key) {
            self.insert(key.clone(), value);
        }
        self.get_mut(&key).expect("existing property")
    }

    pub fn iter(&self) -> ArgumentObjectIter<'_, K, V> {
        ArgumentObjectIter {
            indices: self.indices.values(),
            ordinary: self.ordinary.iter(),
        }
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> + ExactSizeIterator {
        self.iter().map(|(key, _)| key)
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> + ExactSizeIterator {
        self.iter().map(|(_, value)| value)
    }
}

/// Borrowed iteration is allocation-free and supports the same forward/reverse
/// order and exact length as the former single-map iterator.
pub struct ArgumentObjectIter<'a, K, V> {
    indices: std::collections::btree_map::Values<'a, u32, (K, V)>,
    ordinary: indexmap::map::Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for ArgumentObjectIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.indices
            .next()
            .map(|(key, value)| (key, value))
            .or_else(|| self.ordinary.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.indices.len() + self.ordinary.len();
        (len, Some(len))
    }
}

impl<K, V> DoubleEndedIterator for ArgumentObjectIter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.ordinary
            .next_back()
            .or_else(|| self.indices.next_back().map(|(key, value)| (key, value)))
    }
}

impl<K, V> ExactSizeIterator for ArgumentObjectIter<'_, K, V> {}
impl<K, V> std::iter::FusedIterator for ArgumentObjectIter<'_, K, V> {}

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
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.indices
            .into_values()
            .chain(self.ordinary)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a, K: ArgumentKey, V> IntoIterator for &'a ArgumentObject<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = ArgumentObjectIter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: ArgumentKey, V> Index<&K> for ArgumentObject<K, V> {
    type Output = V;
    fn index(&self, key: &K) -> &V {
        self.get(key).expect("existing property")
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

    #[test]
    fn both_partitions_support_lookup_mutable_entry_and_removal_without_reordering() {
        let mut object = ArgumentObject::<JsString, i32>::from([
            ("z".into(), 1),
            ("01".into(), 2),
            ("1".into(), 3),
            ("0".into(), 4),
            ("a".into(), 5),
        ]);
        *object.get_mut(&"1".into()).unwrap() = 30;
        *object.get_mut(&"01".into()).unwrap() = 20;
        assert_eq!(object.entry_or_insert("1".into(), 300), &30);
        assert_eq!(object.entry_or_insert("01".into(), 200), &20);
        *object.entry_or_insert("2".into(), 6) = 60;
        *object.entry_or_insert("b".into(), 7) = 70;
        assert!(object.contains_key(&"2".into()));
        assert!(!object.contains_key(&"3".into()));
        assert_eq!(object.remove(&"missing".into()), None);
        assert_eq!(object.remove(&"0".into()), Some(4));
        assert_eq!(object.remove(&"z".into()), Some(1));
        assert_eq!(object.len(), 5);
        assert_eq!(
            object
                .iter()
                .map(|(k, v)| (k.to_string().unwrap(), *v))
                .collect::<Vec<_>>(),
            [
                ("1".into(), 30),
                ("2".into(), 60),
                ("01".into(), 20),
                ("a".into(), 5),
                ("b".into(), 70)
            ]
        );
        assert!(!object.is_empty());
        for key in ["1", "2", "01", "a", "b"] {
            object.remove(&key.into());
        }
        assert!(object.is_empty());
    }
}
