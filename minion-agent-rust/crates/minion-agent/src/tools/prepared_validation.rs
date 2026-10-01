//! Extends the existing JSON Schema engine's instance domain without changing
//! prepared arguments. The engine owns traversal, references and composition.
//!
//! Its private instance carrier encodes three extra number categories with
//! collision-free finite tags. Every keyword that observes numeric identity or
//! numeric constraints decodes them; no tag is ever handed to a tool or hook.
//! The JSON-only path uses the original validator without any extension.

use std::{collections::BTreeSet, sync::Arc};

use jsonschema::{
    Keyword, ValidationError, Validator,
    paths::{LazyLocation, Location},
};
use serde_json::{Map, Number, Value};

use super::{PreparedNumber, PreparedValue};

#[derive(Debug)]
pub(super) enum PreparedValidationError {
    Schema(String),
    Instance(String),
}

#[derive(Clone)]
struct NumericCarrier {
    tags: [Number; 3],
}

impl NumericCarrier {
    fn new(schema: &Value, value: &PreparedValue) -> Self {
        fn collect(value: &Value, used: &mut BTreeSet<u64>) {
            match value {
                Value::Number(n) => {
                    used.insert(n.as_f64().unwrap().to_bits());
                }
                Value::Array(a) => a.iter().for_each(|v| collect(v, used)),
                Value::Object(o) => o.values().for_each(|v| collect(v, used)),
                _ => {}
            }
        }
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
        collect(schema, &mut used);
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
        Self { tags }
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
            PreparedValue::String(v) => Value::String(v.clone()),
            PreparedValue::Number(n) => Value::Number(match n {
                PreparedNumber::Finite(n) => n.clone(),
                PreparedNumber::PositiveInfinity => self.tags[0].clone(),
                PreparedNumber::NegativeInfinity => self.tags[1].clone(),
                PreparedNumber::NaN => self.tags[2].clone(),
            }),
            PreparedValue::Array(a) => Value::Array(a.iter().map(|v| self.encode(v)).collect()),
            PreparedValue::Object(o) => {
                Value::Object(o.iter().map(|(k, v)| (k.clone(), self.encode(v))).collect())
            }
        }
    }
}

struct RuntimeKeyword {
    name: &'static str,
    base: Validator,
    carrier: Arc<NumericCarrier>,
    path: Location,
    active: bool,
}

impl RuntimeKeyword {
    fn valid(&self, instance: &Value) -> bool {
        if !self.active {
            return true;
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
pub(super) fn validate_prepared(
    schema: &Value,
    value: &PreparedValue,
) -> Result<(), PreparedValidationError> {
    let base = jsonschema::validator_for(schema)
        .map_err(|e| PreparedValidationError::Schema(e.to_string()))?;
    if let Ok(json) = value.try_to_json() {
        return base
            .validate(&json)
            .map_err(|e| PreparedValidationError::Instance(e.to_string()));
    }
    let carrier = Arc::new(NumericCarrier::new(schema, value));
    let mut options = jsonschema::options();
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
    ] {
        let carrier = carrier.clone();
        let dialect = schema.get("$schema").cloned();
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
                let validator = jsonschema::validator_for(&single_schema).map_err(|e| {
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
                    path,
                    active,
                }) as Box<dyn Keyword>)
            },
        );
    }
    let validator = options
        .build(schema)
        .map_err(|e| PreparedValidationError::Schema(e.to_string()))?;
    validator
        .validate(&carrier.encode(value))
        // Do not expose the private carrier's finite tags in a diagnostic.
        // TOOL-003 owns binding-local text, not Pi's JSON.stringify projection.
        .map_err(|e| {
            PreparedValidationError::Instance(format!(
                "prepared runtime arguments fail schema at {}",
                e.schema_path
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
