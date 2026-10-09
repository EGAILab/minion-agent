//! ignore@7.0.5 matching, eager DIV-006 compilation, iterative parents.
use regress::Regex;
use std::collections::HashMap;
struct Rule {
    negative: bool,
    regex: Result<Regex, String>,
}
pub(super) struct Ignore {
    rules: Vec<Rule>,
    cache: HashMap<Vec<u16>, bool>,
    canonical: Vec<u16>,
}
impl Ignore {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            rules: Vec::new(),
            cache: HashMap::new(),
            canonical: minion_agent_pinned_icu::ignore_canonicalize_table()?,
        })
    }
    pub fn add_units(&mut self, pattern: &[u16]) -> Result<bool, regress::Error> {
        use super::ignore_units::{canonical, expression, legacy_hex, points};
        if pattern.is_empty()
            || pattern.iter().all(|&u| {
                char::from_u32(u32::from(u))
                    .is_some_and(|c| crate::javascript::js_trim(&c.to_string()).is_empty())
            })
            || pattern.first() == Some(&35)
            || Regex::new(r"(?:[^\\]|^)\\$")
                .expect("constant regex")
                .find_from_utf16(pattern, 0)
                .next()
                .is_some()
        {
            return Ok(false);
        }
        let negative = pattern.first() == Some(&33);
        let mut body = if negative { &pattern[1..] } else { pattern };
        if body.starts_with(&[92, 33]) || body.starts_with(&[92, 35]) {
            body = &body[1..];
        }
        let expression = legacy_hex(&expression(body));
        Regex::from_unicode(points(&expression), regress::Flags::default())?;
        let canonical = canonical(&expression, &self.canonical);
        let regex = Regex::from_unicode(points(&canonical), regress::Flags::default())?;
        self.rules.push(Rule {
            negative,
            regex: Ok(regex),
        });
        self.cache.clear();
        Ok(true)
    }
    fn one(&self, path: &[u16]) -> Result<bool, ()> {
        let mut ignored = false;
        let mut unignored = false;
        let canonical: Vec<_> = path
            .iter()
            .map(|unit| self.canonical[usize::from(*unit)])
            .collect();
        for rule in &self.rules {
            if (unignored == rule.negative && ignored != unignored)
                || (rule.negative && !ignored && !unignored)
            {
                continue;
            }
            if rule
                .regex
                .as_ref()
                .map_err(|_| ())?
                .find_from_utf16(&canonical, 0)
                .next()
                .is_some()
            {
                ignored = !rule.negative;
                unignored = rule.negative;
            }
        }
        Ok(ignored)
    }
    pub fn ignores(&mut self, path: &[u16]) -> Result<bool, ()> {
        if path.is_empty()
            || path.starts_with(&[47])
            || path.starts_with(&[46, 47])
            || path.starts_with(&[46, 46, 47])
            || path == [46]
            || path == [46, 46]
        {
            return Err(());
        }
        let mut current = path.to_vec();
        let mut pending = Vec::new();
        while !self.cache.contains_key(&current) {
            pending.push(current.clone());
            let mut parts: Vec<_> = current
                .split(|&c| c == 47)
                .filter(|s| !s.is_empty())
                .collect();
            parts.pop();
            if parts.is_empty() {
                break;
            }
            let mut parent = Vec::new();
            for part in parts {
                parent.extend_from_slice(part);
                parent.push(47);
            }
            current = parent;
        }
        let mut ignored = self.cache.get(&current).copied().unwrap_or(false);
        while let Some(current) = pending.pop() {
            if !ignored {
                ignored = self.one(&current)?;
            }
            self.cache.insert(current, ignored);
        }
        Ok(*self.cache.get(path).expect("evaluated addressed path"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_ignore_corpus() {
        let corpus: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../minion-agent-python/tests/skills/data/ignore-corpus.json"
        ))
        .unwrap();
        let mut matcher = Ignore::new().unwrap();
        for (index, case) in corpus["cases"].as_array().unwrap().iter().enumerate() {
            matcher.rules.clear();
            matcher.cache.clear();
            for pattern in case["patterns"].as_array().unwrap() {
                let pattern = pattern.as_str().unwrap();
                // Raw corpus contains Pi's lazy SyntaxErrors; discovery drops
                // precisely these patterns eagerly, per DIV-006.
                if let Err(error) = matcher.add_units(&pattern.encode_utf16().collect::<Vec<_>>()) {
                    matcher.rules.push(Rule {
                        negative: pattern.starts_with('!'),
                        regex: Err(error.to_string()),
                    });
                    matcher.cache.clear();
                }
            }
            for result in case["results"].as_array().unwrap() {
                let path = result["path"].as_str().unwrap();
                let observed = matcher.ignores(&path.encode_utf16().collect::<Vec<_>>());
                if result.get("error").is_some() {
                    assert!(
                        observed.is_err(),
                        "case {index}: {} {path:?}",
                        case["patterns"]
                    );
                } else {
                    assert_eq!(
                        observed.unwrap_or_else(|_| panic!(
                            "unexpected error case {index}: {} {path:?}",
                            case["patterns"]
                        )),
                        result["ignored"].as_bool().unwrap(),
                        "case {index}: {} {path:?}",
                        case["patterns"]
                    );
                }
            }
        }
    }

    #[test]
    fn utf16_prefix_and_nonunicode_case_folding_are_lossless() {
        let mut matcher = Ignore::new().unwrap();
        let pattern = [0xd800, 47, 120];
        matcher.add_units(&pattern).unwrap();
        assert!(matcher.ignores(&pattern).unwrap());
        assert!(!matcher.ignores(&[0xfffd, 47, 120]).unwrap());
        matcher
            .add_units(&"k".encode_utf16().collect::<Vec<_>>())
            .unwrap();
        assert!(matcher.ignores(&[75]).unwrap());
        assert!(!matcher.ignores(&[0x212a]).unwrap());
        matcher.add_units(&[115]).unwrap();
        assert!(matcher.ignores(&[83]).unwrap());
        // U+017F uppercases to ASCII S, but non-Unicode /i must not
        // introduce a non-ASCII -> ASCII equivalence (Kelvin's uppercase
        // remains Kelvin, so it alone cannot discriminate this guard).
        assert!(!matcher.ignores(&[0x017f]).unwrap());
        matcher
            .add_units(&"[Z-a]".encode_utf16().collect::<Vec<_>>())
            .unwrap();
        assert!(matcher.ignores(&[65]).unwrap()); // 'a' maps to 'A' without changing the range's validity
        assert!(matcher.ignores(&[91]).unwrap()); // original range still includes punctuation
    }

    #[test]
    fn ignore_parent_chain_is_stack_independent_and_cache_is_invalidated() {
        let mut matcher = Ignore::new().unwrap();
        matcher
            .add_units(&"ignored".encode_utf16().collect::<Vec<_>>())
            .unwrap();
        let path = format!("{}leaf", "a/".repeat(3000));
        assert!(
            !matcher
                .ignores(&path.encode_utf16().collect::<Vec<_>>())
                .unwrap()
        );
        matcher
            .add_units(&"leaf".encode_utf16().collect::<Vec<_>>())
            .unwrap();
        assert!(
            matcher
                .ignores(&path.encode_utf16().collect::<Vec<_>>())
                .unwrap()
        );
        matcher
            .add_units(&"!ignored/kept".encode_utf16().collect::<Vec<_>>())
            .unwrap();
        assert!(
            matcher
                .ignores(&"ignored/kept".encode_utf16().collect::<Vec<_>>())
                .unwrap()
        );
    }
}
