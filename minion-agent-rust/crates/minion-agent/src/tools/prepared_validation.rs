//! Extends the existing JSON Schema engine's instance domain without changing
//! prepared arguments. The engine owns traversal, references and composition.
//!
//! Its private instance carrier encodes extra number categories and non-scalar
//! UTF-16 strings/keys with collision-free tags. Observing keywords decode them;
//! no tag is ever handed to a tool or hook.
//! The JSON-only path uses the original validator without any extension.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use jsonschema::{
    Keyword, ValidationError, Validator,
    paths::{LazyLocation, Location},
};
use serde_json::{Map, Number, Value};

use super::{PreparedNumber, PreparedString, PreparedValue};

#[derive(Debug)]
pub(super) enum PreparedValidationError {
    Schema(String),
    Instance(String),
}

#[derive(Clone)]
struct NumericCarrier {
    tags: [Number; 3],
    strings: BTreeMap<String, PreparedString>,
}

impl NumericCarrier {
    fn new(schema: &PreparedValue, value: &PreparedValue) -> Self {
        fn collect_runtime(value: &PreparedValue, used: &mut BTreeSet<u64>) {
            match value {
                PreparedValue::Number(PreparedNumber::Finite(n)) => {
                    used.insert(n.as_f64().unwrap().to_bits());
                }
                PreparedValue::Array(a) => a.iter().for_each(|v| collect_runtime(v, used)),
                PreparedValue::Object(o) => o.values().for_each(|v| collect_runtime(v, used)),
                _ => {}
            }
        }
        let mut used = BTreeSet::new();
        collect_runtime(schema, &mut used);
        collect_runtime(value, &mut used);
        let mut bits = f64::MAX.to_bits();
        let tags = std::array::from_fn(|_| {
            while used.contains(&bits) {
                bits -= 1;
            }
            let n = Number::from_f64(f64::from_bits(bits)).unwrap();
            bits -= 1;
            n
        });
        fn runtime_strings(
            value: &PreparedValue,
            used: &mut BTreeSet<String>,
            special: &mut BTreeSet<PreparedString>,
        ) {
            fn string(
                s: &PreparedString,
                used: &mut BTreeSet<String>,
                special: &mut BTreeSet<PreparedString>,
            ) {
                if let Some(s) = s.as_str() {
                    used.insert(s.to_owned());
                } else {
                    special.insert(s.clone());
                }
            }
            match value {
                PreparedValue::String(s) => string(s, used, special),
                PreparedValue::Array(a) => a.iter().for_each(|v| runtime_strings(v, used, special)),
                PreparedValue::Object(o) => {
                    for (k, v) in o {
                        string(k, used, special);
                        runtime_strings(v, used, special);
                    }
                }
                _ => {}
            }
        }
        let mut used_strings = BTreeSet::new();
        let mut special_strings = BTreeSet::new();
        runtime_strings(schema, &mut used_strings, &mut special_strings);
        runtime_strings(value, &mut used_strings, &mut special_strings);
        fn reference_segments(
            value: &PreparedValue,
            used: &mut BTreeSet<String>,
            special: &mut BTreeSet<PreparedString>,
        ) {
            match value {
                PreparedValue::String(s) if s.code_units().starts_with(&[35, 47]) => {
                    for segment in s.code_units()[2..].split(|unit| *unit == 47) {
                        let key = PreparedValue::String(PreparedString::from_code_units(
                            decode_pointer_segment(segment),
                        ));
                        runtime_strings(&key, used, special);
                    }
                }
                PreparedValue::Array(a) => {
                    a.iter().for_each(|v| reference_segments(v, used, special))
                }
                PreparedValue::Object(o) => o
                    .values()
                    .for_each(|v| reference_segments(v, used, special)),
                _ => {}
            }
        }
        reference_segments(schema, &mut used_strings, &mut special_strings);
        let mut strings = BTreeMap::new();
        let mut index = 0;
        for value in special_strings {
            let tag = loop {
                let tag = format!("__minion_private_utf16_{index}__");
                index += 1;
                if used_strings.insert(tag.clone()) {
                    break tag;
                }
            };
            strings.insert(tag, value);
        }
        Self { tags, strings }
    }

    fn encode_string(&self, value: &PreparedString) -> String {
        value.as_str().map(str::to_owned).unwrap_or_else(|| {
            self.strings
                .iter()
                .find(|(_, original)| *original == value)
                .expect("carrier collected every runtime string")
                .0
                .clone()
        })
    }

    fn string(&self, value: &str) -> PreparedString {
        self.strings
            .get(value)
            .cloned()
            .unwrap_or_else(|| value.into())
    }

    fn regex_source(&self, value: &str) -> String {
        use std::fmt::Write;
        let original = self.string(value);
        let mut source = String::new();
        for ch in char::decode_utf16(original.code_units().iter().copied()) {
            match ch {
                Ok(ch) => source.push(ch),
                Err(ch) => write!(source, "\\u{:04x}", ch.unpaired_surrogate()).unwrap(),
            }
        }
        source
    }

    fn special(&self, value: &Value) -> Option<f64> {
        let bits = value.as_f64()?.to_bits();
        self.tags
            .iter()
            .position(|n| n.as_f64().unwrap().to_bits() == bits)
            .map(|i| [f64::INFINITY, f64::NEG_INFINITY, f64::NAN][i])
    }

    fn contains_special(&self, value: &Value) -> bool {
        self.special(value).is_some()
            || match value {
                Value::Array(a) => a.iter().any(|v| self.contains_special(v)),
                Value::Object(o) => o.values().any(|v| self.contains_special(v)),
                _ => false,
            }
    }

    fn encode(&self, value: &PreparedValue) -> Value {
        match value {
            PreparedValue::Null => Value::Null,
            PreparedValue::Bool(v) => Value::Bool(*v),
            PreparedValue::String(v) => Value::String(self.encode_string(v)),
            PreparedValue::Number(n) => Value::Number(match n {
                PreparedNumber::Finite(n) => n.clone(),
                PreparedNumber::PositiveInfinity => self.tags[0].clone(),
                PreparedNumber::NegativeInfinity => self.tags[1].clone(),
                PreparedNumber::NaN => self.tags[2].clone(),
            }),
            PreparedValue::Array(a) => Value::Array(a.iter().map(|v| self.encode(v)).collect()),
            PreparedValue::Object(o) => Value::Object(
                o.iter()
                    .map(|(k, v)| (self.encode_string(k), self.encode(v)))
                    .collect(),
            ),
        }
    }

    // Local JSON pointers address schema keys, not their private carrier tags.
    // Rewrite only reference locations, never literal const/enum contents.
    fn encode_schema(&self, value: &PreparedValue) -> Value {
        match value {
            PreparedValue::Array(a) => {
                Value::Array(a.iter().map(|v| self.encode_schema(v)).collect())
            }
            PreparedValue::Object(o) => Value::Object(
                o.iter()
                    .map(|(k, v)| {
                        let encoded = if matches!(k.as_str(), Some("$ref" | "$dynamicRef"))
                            && let PreparedValue::String(s) = v
                            && s.code_units().starts_with(&[35, 47])
                        {
                            let mut pointer = String::from("#");
                            for segment in s.code_units()[2..].split(|unit| *unit == 47) {
                                let key = self.encode_string(&PreparedString::from_code_units(
                                    decode_pointer_segment(segment),
                                ));
                                pointer.push('/');
                                pointer.push_str(&key.replace('~', "~0").replace('/', "~1"));
                            }
                            Value::String(pointer)
                        } else if matches!(
                            k.as_str(),
                            Some(
                                "properties"
                                    | "patternProperties"
                                    | "$defs"
                                    | "definitions"
                                    | "dependentSchemas"
                            )
                        ) && let PreparedValue::Object(entries) = v
                        {
                            Value::Object(
                                entries
                                    .iter()
                                    .map(|(name, subschema)| {
                                        (self.encode_string(name), self.encode_schema(subschema))
                                    })
                                    .collect(),
                            )
                        } else if matches!(
                            k.as_str(),
                            Some("const" | "enum" | "default" | "examples")
                        ) {
                            self.encode(v)
                        } else {
                            self.encode_schema(v)
                        };
                        (self.encode_string(k), encoded)
                    })
                    .collect(),
            ),
            _ => self.encode(value),
        }
    }
}

fn decode_pointer_segment(segment: &[u16]) -> Vec<u16> {
    let mut decoded = Vec::new();
    let mut i = 0;
    while i < segment.len() {
        if segment[i] == 126 && i + 1 < segment.len() && matches!(segment[i + 1], 48 | 49) {
            decoded.push(if segment[i + 1] == 48 { 126 } else { 47 });
            i += 2;
        } else {
            decoded.push(segment[i]);
            i += 1;
        }
    }
    decoded
}

struct RuntimeKeyword {
    name: &'static str,
    base: Validator,
    carrier: Arc<NumericCarrier>,
    path: Location,
    active: bool,
    constraint: Value,
    pattern: Option<regress::Regex>,
}

impl RuntimeKeyword {
    fn valid(&self, instance: &Value) -> bool {
        if !self.active {
            return true;
        }
        if matches!(self.name, "minLength" | "maxLength" | "pattern") {
            let Some(s) = instance.as_str() else {
                return true;
            };
            let original = self.carrier.string(s);
            return match self.name {
                "minLength" => {
                    original.code_point_len() as f64 >= self.constraint.as_f64().unwrap()
                }
                "maxLength" => {
                    original.code_point_len() as f64 <= self.constraint.as_f64().unwrap()
                }
                "pattern" => self
                    .pattern
                    .as_ref()
                    .unwrap()
                    .find_from_utf16(original.code_units(), 0)
                    .next()
                    .is_some(),
                _ => unreachable!(),
            };
        }
        if self.name == "const" || self.name == "enum" {
            return !self.carrier.contains_special(instance) && self.base.is_valid(instance);
        }
        if self.name == "uniqueItems" {
            // Tags are collision-free and preserve each non-finite category's
            // distinct identity, so the existing engine's deep equality remains
            // applicable. No numeric range operation participates in equality.
            return self.base.is_valid(instance);
        }
        if self.carrier.special(instance).is_none() {
            return self.base.is_valid(instance);
        }
        match self.name {
            "type" => false, // all explicit JSON types exclude non-finite numbers
            // TOOL-041 / RC001: numeric keywords apply only to finite numbers.
            // Composition still belongs to the engine: an inapplicable bound
            // succeeds, so two such oneOf branches fail, and not reverses it.
            "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" | "multipleOf" => true,
            _ => unreachable!(),
        }
    }
}

impl Keyword for RuntimeKeyword {
    fn is_valid(&self, instance: &Value) -> bool {
        self.valid(instance)
    }

    fn validate<'i>(
        &self,
        instance: &'i Value,
        location: &LazyLocation,
    ) -> Result<(), ValidationError<'i>> {
        if self.valid(instance) {
            Ok(())
        } else {
            Err(ValidationError::custom(
                self.path.clone(),
                location.into(),
                instance,
                format!("prepared runtime value fails {}", self.name),
            ))
        }
    }
}

// The jsonschema custom-keyword factory fixes this unboxed error signature.
#[allow(clippy::result_large_err)]
#[cfg(test)]
pub(super) fn validate_prepared(
    schema: &Value,
    value: &PreparedValue,
) -> Result<(), PreparedValidationError> {
    validate_runtime_schema(&PreparedValue::from(schema.clone()), value)
}

pub(super) fn validate_runtime_schema(
    schema: &PreparedValue,
    value: &PreparedValue,
) -> Result<(), PreparedValidationError> {
    if let (Ok(schema), Ok(json)) = (schema.try_to_json(), value.try_to_json()) {
        let base = jsonschema::validator_for(&schema)
            .map_err(|e| PreparedValidationError::Schema(e.to_string()))?;
        return base
            .validate(&json)
            .map_err(|e| PreparedValidationError::Instance(e.to_string()));
    }
    let carrier = Arc::new(NumericCarrier::new(schema, value));
    let schema = carrier.encode_schema(schema);
    let validator =
        runtime_validator(&schema, carrier.clone()).map_err(PreparedValidationError::Schema)?;
    validator.validate(&carrier.encode(value)).map_err(|e| {
        PreparedValidationError::Instance(format!(
            "prepared runtime arguments fail schema at {}",
            e.schema_path
        ))
    })
}

fn runtime_validator(schema: &Value, carrier: Arc<NumericCarrier>) -> Result<Validator, String> {
    runtime_validator_in(schema, carrier, Arc::new(schema.clone()))
}

#[allow(clippy::result_large_err)] // custom-keyword factory fixes the unboxed error signature
fn runtime_validator_in(
    schema: &Value,
    carrier: Arc<NumericCarrier>,
    root: Arc<Value>,
) -> Result<Validator, String> {
    let mut options = jsonschema::options();
    let uri = root
        .get("$id")
        .and_then(Value::as_str)
        .unwrap_or("urn:minion:prepared-runtime");
    options = options.with_base_uri(uri).with_resource(
        uri,
        jsonschema::Resource::from_contents((*root).clone()).map_err(|e| e.to_string())?,
    );
    for name in [
        "type",
        "const",
        "enum",
        "uniqueItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
    ] {
        let carrier = carrier.clone();
        let dialect = root.get("$schema").cloned();
        options = options.with_keyword(
            name,
            move |parent: &Map<String, Value>, constraint: &Value, path: Location| {
                let mut s = Map::new();
                s.insert(name.into(), constraint.clone());
                if let Some(uri) = parent.get("$schema").or(dialect.as_ref()) {
                    s.insert("$schema".into(), uri.clone());
                }
                // Draft 4 makes exclusivity a boolean modifier of the bound,
                // unlike later drafts' independent numeric keyword.
                let modifier = match name {
                    "minimum" => Some("exclusiveMinimum"),
                    "maximum" => Some("exclusiveMaximum"),
                    _ => None,
                };
                if let Some(key) = modifier
                    && let Some(value @ Value::Bool(_)) = parent.get(key)
                {
                    s.insert(key.into(), value.clone());
                }
                if constraint.is_boolean() {
                    let bound = match name {
                        "exclusiveMinimum" => Some("minimum"),
                        "exclusiveMaximum" => Some("maximum"),
                        _ => None,
                    };
                    if let Some(key) = bound
                        && let Some(value) = parent.get(key)
                    {
                        s.insert(key.into(), value.clone());
                    }
                }
                let single_schema = Value::Object(s);
                let active = name != "const"
                    || jsonschema::Draft::default().detect(&single_schema).ok()
                        != Some(jsonschema::Draft::Draft4);
                let validator = jsonschema::validator_for(if name == "pattern" {
                    &Value::Bool(true)
                } else {
                    &single_schema
                })
                .map_err(|e| {
                    ValidationError::custom(
                        path.clone(),
                        Location::new(),
                        constraint,
                        e.to_string(),
                    )
                })?;
                Ok(Box::new(RuntimeKeyword {
                    name,
                    base: validator,
                    carrier: carrier.clone(),
                    path: path.clone(),
                    active,
                    constraint: constraint.clone(),
                    pattern: if name == "pattern" {
                        Some(
                            regress::Regex::with_flags(
                                &carrier.regex_source(constraint.as_str().unwrap()),
                                "u",
                            )
                            .map_err(|e| {
                                ValidationError::custom(
                                    path.clone(),
                                    Location::new(),
                                    constraint,
                                    e.to_string(),
                                )
                            })?,
                        )
                    } else {
                        None
                    },
                }) as Box<dyn Keyword>)
            },
        );
    }
    if !carrier.strings.is_empty() {
        for name in ["patternProperties", "additionalProperties"] {
            let carrier = carrier.clone();
            let root = root.clone();
            options = options.with_keyword(
                name,
                move |parent: &Map<String, Value>, constraint: &Value, path: Location| {
                    let patterns = parent
                        .get("patternProperties")
                        .and_then(Value::as_object)
                        .map(|p| {
                            p.iter()
                                .map(|(pattern, schema)| {
                                    regress::Regex::with_flags(&carrier.regex_source(pattern), "u")
                                        .map(|re| (re, schema.clone()))
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()
                        .map_err(|e| {
                            ValidationError::custom(
                                path.clone(),
                                Location::new(),
                                constraint,
                                e.to_string(),
                            )
                        })?
                        .unwrap_or_default();
                    let properties = parent
                        .get("properties")
                        .and_then(Value::as_object)
                        .map(|p| p.keys().cloned().collect())
                        .unwrap_or_default();
                    Ok(Box::new(RuntimeObjectKeyword {
                        name,
                        constraint: constraint.clone(),
                        patterns,
                        properties,
                        carrier: carrier.clone(),
                        root: root.clone(),
                        path,
                    }) as Box<dyn Keyword>)
                },
            );
        }
    }
    if schema != root.as_ref() {
        fn absolute_fragments(value: &mut Value, base: &str) {
            match value {
                Value::Array(a) => a.iter_mut().for_each(|v| absolute_fragments(v, base)),
                Value::Object(o) => {
                    let scoped_base = o.get("$id").and_then(Value::as_str).and_then(|id| {
                        url::Url::parse(id)
                            .or_else(|_| url::Url::parse(base)?.join(id))
                            .ok()
                    });
                    let base = scoped_base.as_ref().map(url::Url::as_str).unwrap_or(base);
                    for (key, value) in o {
                        if matches!(key.as_str(), "$ref" | "$dynamicRef")
                            && value.as_str().is_some_and(|s| s.starts_with('#'))
                        {
                            *value = Value::String(format!("{base}{}", value.as_str().unwrap()));
                        } else {
                            absolute_fragments(value, base);
                        }
                    }
                }
                _ => {}
            }
        }
        let mut nested = schema.clone();
        absolute_fragments(&mut nested, uri);
        options
            .with_base_uri("urn:minion:prepared-runtime:subschema")
            .build(&nested)
            .map_err(|e| e.to_string())
    } else {
        options.build(schema).map_err(|e| e.to_string())
    }
}

struct RuntimeObjectKeyword {
    name: &'static str,
    constraint: Value,
    patterns: Vec<(regress::Regex, Value)>,
    properties: BTreeSet<String>,
    carrier: Arc<NumericCarrier>,
    root: Arc<Value>,
    path: Location,
}

impl RuntimeObjectKeyword {
    fn valid(&self, instance: &Value) -> bool {
        let Some(object) = instance.as_object() else {
            return true;
        };
        for (key, value) in object {
            let original = self.carrier.string(key);
            let matches = self
                .patterns
                .iter()
                .filter(|(re, _)| {
                    re.find_from_utf16(original.code_units(), 0)
                        .next()
                        .is_some()
                })
                .collect::<Vec<_>>();
            if self.name == "patternProperties" {
                for (_, schema) in matches {
                    if !runtime_validator_in(schema, self.carrier.clone(), self.root.clone())
                        .is_ok_and(|v| v.is_valid(value))
                    {
                        return false;
                    }
                }
            } else if !self.properties.contains(key)
                && matches.is_empty()
                && !runtime_validator_in(&self.constraint, self.carrier.clone(), self.root.clone())
                    .is_ok_and(|v| v.is_valid(value))
            {
                return false;
            }
        }
        true
    }
}

impl Keyword for RuntimeObjectKeyword {
    fn is_valid(&self, instance: &Value) -> bool {
        self.valid(instance)
    }
    fn validate<'i>(
        &self,
        instance: &'i Value,
        location: &LazyLocation,
    ) -> Result<(), ValidationError<'i>> {
        if self.valid(instance) {
            Ok(())
        } else {
            Err(ValidationError::custom(
                self.path.clone(),
                location.into(),
                instance,
                format!("prepared runtime object fails {}", self.name),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema_literals_names_regexes_and_local_refs_keep_utf16_identity() {
        let runtime = |text: &str| PreparedValue::from(crate::llm::RawValue::decode(text).unwrap());
        let schema = runtime(
            r##"{"$defs":{"\ud800":{"const":"\udfff"}},"properties":{"const":{"$ref":"#/$defs/\ud800"}}}"##,
        );
        assert!(validate_runtime_schema(&schema, &runtime(r#"{"const":"\udfff"}"#)).is_ok());
        assert!(validate_runtime_schema(&schema, &runtime(r#"{"const":"\ufffd"}"#)).is_err());
        let missing =
            runtime(r##"{"$ref":"#/$defs/\ud800","properties":{"x":{"const":"\udfff"}}}"##);
        assert!(matches!(
            validate_runtime_schema(&missing, &runtime("{}")),
            Err(PreparedValidationError::Schema(_))
        ));
        let literal = runtime(r##"{"const":{"$ref":"#/\ud800"}}"##);
        assert!(validate_runtime_schema(&literal, &runtime(r##"{"$ref":"#/\ud800"}"##)).is_ok());
        assert!(validate_runtime_schema(&literal, &runtime(r##"{"$ref":"#/\ufffd"}"##)).is_err());
    }

    fn utf16_object(units: Vec<u16>) -> PreparedValue {
        let mut value = PreparedValue::from(json!({"text":"raw"}));
        value["text"] = PreparedValue::String(PreparedString::from_code_units(units));
        value
    }

    #[test]
    fn runtime_surrogates_validate_without_scalar_normalization() {
        let high = utf16_object(vec![0xd800]);
        let pair = utf16_object(vec![0xd83d, 0xde00]);
        let two = utf16_object(vec![0xd800, 0xd800]);
        for value in [&high, &pair] {
            assert!(
                validate_prepared(
                    &json!({"properties":{"text":{"type":"string","maxLength":1,"pattern":"^.$"}}}),
                    value
                )
                .is_ok()
            );
        }
        assert!(validate_prepared(&json!({"properties":{"text":{"maxLength":1}}}), &two).is_err());
        assert!(
            validate_prepared(
                &json!({"properties":{"text":{"pattern":"^..$","minLength":2}}}),
                &two
            )
            .is_ok()
        );
        assert!(validate_prepared(&json!({"properties":{"text":{"enum":["�"]}}}), &high).is_err());
        assert!(
            validate_prepared(
                &json!({"properties":{"text":{"pattern":"^\\p{Surrogate}$"}}}),
                &high
            )
            .is_ok()
        );
        assert!(
            validate_prepared(
                &json!({"properties":{"text":{"pattern":"^\\p{Letter}$"}}}),
                &high
            )
            .is_err()
        );
    }

    #[test]
    fn runtime_keys_use_real_unicode_patterns_and_nested_reference_validation() {
        let key = PreparedString::from_code_units(vec![65, 0xd800]);
        let object = PreparedValue::Object(BTreeMap::from([(key, PreparedValue::Bool(true))]));
        assert!(validate_prepared(&json!({"patternProperties":{"^A.$":{"type":"boolean"}},"additionalProperties":false}),&object).is_ok());
        assert!(
            validate_prepared(
                &json!({"patternProperties":{"^A.$":{"type":"number"}}}),
                &object
            )
            .is_err()
        );
        assert!(validate_prepared(&json!({"patternProperties":{"^B":{"type":"boolean"}},"additionalProperties":false}),&object).is_err());
        assert!(validate_prepared(&json!({"$defs":{"entry":{"type":"boolean"}},"patternProperties":{"^A.$":{"$ref":"#/$defs/entry"}},"additionalProperties":false}),&object).is_ok());
        assert!(validate_prepared(&json!({"patternProperties":{"^A.$":{"$id":"https://example.test/nested","$defs":{"entry":{"type":"boolean"}},"$ref":"#/$defs/entry"}}}),&object).is_ok());
        assert!(validate_prepared(&json!({"propertyNames":{"pattern":"^B"}}), &object).is_err());
    }

    #[test]
    fn private_string_tags_cannot_collide_with_schema_or_runtime_literals() {
        let mut value = utf16_object(vec![0xd800]);
        value["ordinary"] = PreparedValue::from(json!("__minion_private_utf16_0__"));
        let schema = json!({"properties":{"text":{"const":"__minion_private_utf16_1__"},"ordinary":{"const":"__minion_private_utf16_0__"}}});
        assert!(validate_prepared(&schema, &value).is_err());
        assert_eq!(value["ordinary"], json!("__minion_private_utf16_0__"));
        assert_eq!(
            match &value["text"] {
                PreparedValue::String(s) => s.code_units(),
                _ => panic!(),
            },
            &[0xd800]
        );
    }

    #[test]
    fn integer_valued_length_constraints_and_invalid_schemas_keep_engine_behavior() {
        let value = utf16_object(vec![0xd800]);
        assert!(
            validate_prepared(
                &json!({"properties":{"text":{"minLength":1.0,"maxLength":1.0}}}),
                &value
            )
            .is_ok()
        );
        for schema in [
            json!({"properties":{"text":{"minLength":1.5}}}),
            json!({"patternProperties":{"^A":{"type":"invalid"}}}),
            json!({"additionalProperties":{"type":"invalid"}}),
        ] {
            assert!(matches!(
                validate_prepared(&schema, &value),
                Err(PreparedValidationError::Schema(_))
            ));
        }
    }

    fn special_object(number: f64) -> PreparedValue {
        let mut value = PreparedValue::from(json!({"n": 0}));
        value["n"] = PreparedValue::number(number);
        value
    }

    #[test]
    fn declared_numeric_and_nullable_positions_are_finite_only() {
        for schema in [
            json!({"type":"number"}),
            json!({"type":"integer"}),
            json!({"type":["number","null"]}),
            json!({"anyOf":[{"type":"number"},{"type":"null"}]}),
        ] {
            for number in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                assert!(validate_prepared(&schema, &PreparedValue::number(number)).is_err());
            }
            assert!(validate_prepared(&schema, &PreparedValue::number(-0.0)).is_ok());
        }
    }

    #[test]
    fn nested_constraints_references_and_complete_unconstrained_union_branches() {
        let value = special_object(f64::INFINITY);
        assert!(validate_prepared(&json!({"type":"object","properties":{}}), &value).is_ok());
        let constrained = json!({"type":"object","properties":{"n":{"type":"number"}}});
        assert!(validate_prepared(&constrained, &value).is_err());
        for branches in [
            json!([constrained.clone(), {}]),
            json!([{}, constrained.clone()]),
        ] {
            assert!(validate_prepared(&json!({"anyOf":branches}), &value).is_ok());
        }
        assert!(validate_prepared(&json!({"allOf":[{}, constrained]}), &value).is_err());
        assert!(
            validate_prepared(
                &json!({"$defs":{"n":{"type":"number"}},"properties":{"n":{"$ref":"#/$defs/n"}}}),
                &value
            )
            .is_err()
        );
        let array = PreparedValue::Array(vec![PreparedValue::number(f64::NAN)]);
        assert!(validate_prepared(&json!({"items":{"type":"number"}}), &array).is_err());
        assert!(
            validate_prepared(
                &json!({"anyOf":[{"items":{"type":"number"}},{"items":{}}]}),
                &array
            )
            .is_ok()
        );
    }

    #[test]
    fn non_finite_does_not_impersonate_null_string_const_or_enum() {
        for n in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let value = PreparedValue::number(n);
            for schema in [
                json!({"type":"null"}),
                json!({"type":"string"}),
                json!({"const":null}),
                json!({"enum":[null,"Infinity",0]}),
            ] {
                assert!(validate_prepared(&schema, &value).is_err());
            }
            assert!(validate_prepared(&json!({"not":{"type":"number"}}), &value).is_ok());
        }
    }

    #[test]
    fn tags_do_not_collide_with_real_values_or_schema_constants() {
        let next = f64::from_bits(f64::MAX.to_bits() - 1);
        let mut value = PreparedValue::from(json!({"n":f64::MAX,"next":next}));
        value["extra"] = PreparedValue::number(f64::INFINITY);
        assert!(validate_prepared(&json!({"properties":{"n":{"const":f64::MAX},"next":{"type":"number","maximum":next}}}), &value).is_ok());
        assert!(
            validate_prepared(&json!({"properties":{"extra":{"const":f64::MAX}}}), &value).is_err()
        );
        assert!(value["extra"].as_f64().unwrap().is_infinite());
        assert_eq!(value["n"].as_f64(), Some(f64::MAX));
    }

    #[test]
    fn unique_items_preserves_distinct_categories_and_duplicate_detection() {
        let schema = json!({"uniqueItems":true});
        let distinct = PreparedValue::Array(vec![
            PreparedValue::Null,
            PreparedValue::number(0.0),
            PreparedValue::number(f64::INFINITY),
            PreparedValue::number(f64::NEG_INFINITY),
            PreparedValue::number(f64::NAN),
        ]);
        assert!(validate_prepared(&schema, &distinct).is_ok());
        for n in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let duplicate =
                PreparedValue::Array(vec![PreparedValue::number(n), PreparedValue::number(n)]);
            assert!(validate_prepared(&schema, &duplicate).is_err());
        }
        assert!(
            validate_prepared(
                &json!({"uniqueItems":false}),
                &PreparedValue::number(f64::NAN)
            )
            .is_ok()
        );
    }

    #[test]
    fn numeric_keywords_ignore_non_finite_values_but_constrain_finite_values() {
        for (schema, rejected_finite) in [
            (json!({"minimum":0}), -1.0),
            (json!({"maximum":0}), 1.0),
            (json!({"exclusiveMinimum":0}), -0.0),
            (json!({"exclusiveMaximum":0}), 0.0),
            (json!({"multipleOf":2}), 3.0),
        ] {
            for number in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                assert!(
                    validate_prepared(&schema, &PreparedValue::number(number)).is_ok(),
                    "non-finite {number} must be outside {schema}"
                );
            }
            assert!(validate_prepared(&schema, &PreparedValue::number(rejected_finite)).is_err());
            // Exercise the extended engine too, not just the JSON-only fast path.
            let mut value = special_object(rejected_finite);
            value["extra"] = PreparedValue::number(f64::NAN);
            assert!(validate_prepared(&json!({"properties":{"n":schema}}), &value).is_err());
        }
    }

    #[test]
    fn numeric_keyword_applicability_survives_composition_references_and_arrays() {
        for number in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let value = PreparedValue::number(number);
            assert!(
                validate_prepared(&json!({"oneOf":[{"maximum":0},{"minimum":1}]}), &value).is_err()
            );
            assert!(validate_prepared(&json!({"not":{"maximum":0}}), &value).is_err());
            assert!(
                validate_prepared(&json!({"anyOf":[{"maximum":0},{"type":"string"}]}), &value)
                    .is_ok()
            );
            assert!(
                validate_prepared(&json!({"allOf":[{"maximum":0},{"minimum":1}]}), &value).is_ok()
            );
            assert!(validate_prepared(&json!({"type":"number","maximum":0}), &value).is_err());
            assert!(
                validate_prepared(
                    &json!({"$defs":{"bound":{"maximum":0}},"$ref":"#/$defs/bound"}),
                    &value
                )
                .is_ok()
            );
            assert!(
                validate_prepared(
                    &json!({"items":{"maximum":0}}),
                    &PreparedValue::Array(vec![value])
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn older_draft_bound_modifiers_and_finite_siblings_keep_their_rules() {
        let mut value = special_object(0.0);
        value["extra"] = PreparedValue::number(f64::INFINITY);
        let schema = json!({"$schema":"http://json-schema.org/draft-04/schema#", "properties":{"n":{"minimum":0,"exclusiveMinimum":true}}});
        let baseline = jsonschema::validator_for(&schema).unwrap();
        assert!(!baseline.is_valid(&json!({"n":0})));
        assert!(baseline.is_valid(&json!({"n":1})));
        assert!(validate_prepared(&schema, &value).is_err());
        value["n"] = PreparedValue::number(1.0);
        let outcome = validate_prepared(&schema, &value);
        assert!(outcome.is_ok(), "{outcome:?}");
        let schema = json!({"$schema":"http://json-schema.org/draft-04/schema#", "const":null});
        assert!(
            jsonschema::validator_for(&schema)
                .unwrap()
                .is_valid(&json!({"n":1}))
        );
        assert!(
            validate_prepared(&schema, &value).is_ok(),
            "Draft 4 did not adopt const"
        );
    }
}
