use minion_agent::{
    llm::RawValue,
    tools::{PreparedValue, RuntimeSchemaError, RuntimeSchemaObject, ToolDefinition},
};
use serde_json::json;

#[test]
fn argument_sharing_does_not_introduce_a_runtime_schema_mutation_surface() {
    let input = PreparedValue::from(json!({"const":0}));
    let schema = RuntimeSchemaObject::try_from(input.clone()).unwrap();
    input.set("const", PreparedValue::number(f64::NAN));
    let snapshot = schema.as_value();
    snapshot.set("const", PreparedValue::number(f64::INFINITY));
    assert_eq!(
        schema.try_to_json().unwrap(),
        serde_json::from_value(json!({"const":0})).unwrap()
    );
    assert_eq!(
        schema.clone().as_value().get("const").unwrap().as_f64(),
        Some(0.0)
    );
}

#[test]
fn runtime_schema_preserves_utf16_values_and_keys_and_refuses_lossy_projection() {
    let raw =
        RawValue::decode(r#"{"properties":{"\ud800":{"const":"\udfff"}},"required":["\ud800"]}"#)
            .unwrap();
    let expected = PreparedValue::from(raw);
    let schema = RuntimeSchemaObject::try_from(expected.clone()).unwrap();
    let tool = ToolDefinition::new_with_runtime_schema("probe", "probe", schema, "probe", |_| {
        Box::pin(async { unreachable!("metadata inspection never executes") })
    });
    assert_eq!(tool.parameters().as_value(), expected);
    assert_eq!(tool.schema(), Err(RuntimeSchemaError::NonScalarProjection));
}

#[test]
fn scalar_schema_projection_and_domain_checks_remain_explicit() {
    let scalar = json!({"type":"object","properties":{"x":{"type":"string"}}});
    let schema = RuntimeSchemaObject::try_from(PreparedValue::from(scalar.clone())).unwrap();
    assert_eq!(
        serde_json::Value::from(schema.try_to_json().unwrap()),
        scalar
    );
    assert_eq!(
        RuntimeSchemaObject::try_from(PreparedValue::Null),
        Err(RuntimeSchemaError::InvalidDomain)
    );
    let value = PreparedValue::from(json!({"const":0}));
    value.set("const", PreparedValue::number(f64::NAN));
    assert_eq!(
        RuntimeSchemaObject::try_from(value),
        Err(RuntimeSchemaError::InvalidDomain)
    );
}
