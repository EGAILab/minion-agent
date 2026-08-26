use minion_agent::llm::{
    ConstrainedSampling, GrammarVariants, JsonSchemaObject, JsonSchemaStrictness, ToolSchema,
};
use serde_json::{Value, json};

#[test]
fn parameters_accept_and_preserve_every_object_valued_schema_shape() {
    let schemas = [
        json!({"type": "object", "properties": {}}),
        json!({"type": "string"}),
        json!({"oneOf": [{"type": "string"}, {"type": "number"}]}),
        json!({"$comment": "x", "custom-extension": {"a": 1}}),
    ];

    for schema in schemas {
        let parsed: JsonSchemaObject = serde_json::from_value(schema.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), schema);
    }
}

#[test]
fn parameters_reject_every_non_object_container() {
    for invalid in [
        Value::Null,
        Value::Bool(true),
        Value::Bool(false),
        json!([]),
        json!("schema"),
        json!(42),
    ] {
        assert!(serde_json::from_value::<JsonSchemaObject>(invalid).is_err());
    }
}

#[test]
fn constrained_sampling_preserves_absent_false_json_schema_and_grammar() {
    let documents = [
        json!({
            "name": "absent",
            "description": "absent",
            "parameters": {},
        }),
        json!({
            "name": "disabled",
            "description": "disabled",
            "parameters": {},
            "constrained_sampling": false,
        }),
        json!({
            "name": "strict",
            "description": "strict",
            "parameters": {},
            "constrained_sampling": {"type": "json_schema", "strict": "require"},
        }),
        json!({
            "name": "grammar",
            "description": "grammar",
            "parameters": {},
            "constrained_sampling": {
                "type": "grammar",
                "variants": {"openai_lark": "start: WORD+", "openai_regex": "^[a-z]+$"},
            },
        }),
    ];

    let parsed = documents
        .into_iter()
        .map(|document| serde_json::from_value::<ToolSchema>(document).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(parsed[0].constrained_sampling, None);
    assert_eq!(
        parsed[1].constrained_sampling,
        Some(ConstrainedSampling::Disabled)
    );
    assert_eq!(
        parsed[2].constrained_sampling,
        Some(ConstrainedSampling::JsonSchema {
            strict: JsonSchemaStrictness::Require,
        })
    );
    assert_eq!(
        parsed[3].constrained_sampling,
        Some(ConstrainedSampling::Grammar {
            variants: GrammarVariants {
                openai_lark: Some("start: WORD+".into()),
                openai_regex: Some("^[a-z]+$".into()),
            },
        })
    );
    assert_eq!(
        parsed[0].as_json()["constrained_sampling"],
        Value::Null
    );
}

#[test]
fn grammar_variants_accept_only_the_two_independently_optional_pi_keys() {
    for valid in [
        json!({"type": "grammar", "variants": {}}),
        json!({"type": "grammar", "variants": {"openai_lark": "start: WORD+"}}),
        json!({"type": "grammar", "variants": {"openai_regex": "^[a-z]+$"}}),
        json!({
            "type": "grammar",
            "variants": {"openai_lark": "start: WORD+", "openai_regex": "^[a-z]+$"},
        }),
    ] {
        assert!(serde_json::from_value::<ConstrainedSampling>(valid).is_ok());
    }

    assert!(
        serde_json::from_value::<ConstrainedSampling>(json!({
            "type": "grammar",
            "variants": {"unknown": "value"},
        }))
        .is_err()
    );
}

#[test]
fn constrained_sampling_rejects_true_while_projection_null_round_trips_absence() {
    assert!(
        serde_json::from_value::<ToolSchema>(json!({
            "name": "true",
            "description": "true",
            "parameters": {},
            "constrained_sampling": true,
        }))
        .is_err()
    );
    let projected: ToolSchema = serde_json::from_value(json!({
        "name": "null",
        "description": "null",
        "parameters": {},
        "constrained_sampling": null,
    }))
    .unwrap();
    assert_eq!(projected.constrained_sampling, None);
}
