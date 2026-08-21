use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

use super::RuntimeError;

macro_rules! normative_name {
    ($name:ident, $valid:ident) => {
        #[derive(Clone, Debug)]
        pub struct $name(Arc<str>);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, RuntimeError> {
                let value = value.as_ref();
                if !$valid(value) {
                    return Err(RuntimeError::InvalidName(value.to_owned()));
                }
                Ok(Self(Arc::from(value)))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.as_str() == other.as_str()
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                self.as_str().cmp(other.as_str())
            }
        }

        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.as_str().hash(state);
            }
        }
    };
}

normative_name!(ServiceName, is_simple_name);
normative_name!(EventName, is_event_name);

fn is_simple_name(value: &str) -> bool {
    let mut characters = value.bytes();
    matches!(characters.next(), Some(b'a'..=b'z'))
        && characters.all(|character| matches!(character, b'a'..=b'z' | b'0'..=b'9' | b'_'))
}

fn is_event_name(value: &str) -> bool {
    let mut segments = value.split('/');
    matches!(segments.next(), Some(first) if is_simple_name(first))
        && segments.all(is_qualified_event_segment)
}

fn is_qualified_event_segment(value: &str) -> bool {
    let mut characters = value.bytes();
    matches!(characters.next(), Some(b'a'..=b'z'))
        && characters.all(|character| matches!(character, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-'))
}
