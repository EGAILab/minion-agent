use minion_agent::{
    llm::{
        Message, RawNumber, ResultString, ResultTextBlock, ResultValue, ToolResultContentBlock,
        ToolResultMessage,
    },
    session::Session,
};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn typed_session_preserves_every_result_domain_member_without_json() {
    let high = ResultString::from_code_units(vec![0xD800]);
    let low = ResultString::from_code_units(vec![0xDC00]);
    let details = ResultValue::Object(BTreeMap::from([(
        high.clone(),
        ResultValue::Array(vec![
            ResultValue::String(low.clone()),
            ResultValue::number(-0.0),
            ResultValue::number(f64::INFINITY),
            ResultValue::number(f64::NEG_INFINITY),
            ResultValue::number(f64::NAN),
            ResultValue::Null,
            ResultValue::Bool(false),
        ]),
    )]));
    let mut message = ToolResultMessage::new(
        "call",
        "tool",
        vec![ToolResultContentBlock::Text(ResultTextBlock::new(high))],
        false,
        0.0,
    );
    message.details = Some(details.clone());
    let session = Session::new("room", [] as [&str; 0]).unwrap();
    session
        .append_message(Message::ToolResult(Box::new(message.clone())))
        .unwrap();
    assert_eq!(
        session.derive_messages().unwrap(),
        vec![Message::ToolResult(Box::new(message.clone()))]
    );
    assert!(!session.events().is_empty());
    assert!(
        serde_json::to_value(&message).is_err(),
        "live values must never silently become JSON null or replacement text"
    );
    assert!(details.try_to_json().is_err());
    assert!(
        RawNumber::new(f64::NAN).is_err(),
        "raw domain remains unchanged"
    );
}

#[test]
fn result_json_boundary_is_explicit_and_finite_scalar_roundtrip_is_compatible() {
    for value in [
        json!({"x":[true,false,null,0,1.5,"text"]}),
        json!(""),
        json!([]),
        json!({}),
        json!(false),
    ] {
        let runtime = ResultValue::from(value.clone());
        assert_eq!(runtime.try_to_json().unwrap(), value);
        let encoded = serde_json::to_string(&runtime).unwrap();
        let decoded: ResultValue = serde_json::from_str(&encoded).unwrap();
        assert_eq!(runtime, decoded);
    }
    let negative = ResultValue::number(-0.0);
    assert_ne!(negative, ResultValue::number(0.0));
    let encoded = serde_json::to_string(&negative).unwrap();
    let decoded: ResultValue = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, negative);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(serde_json::to_value(ResultValue::number(value)).is_err());
    }
    assert_eq!(
        ResultString::from_code_units(vec![0xD83D, 0xDE00]),
        ResultString::from("😀")
    );
    assert_ne!(
        ResultString::from_code_units(vec![0xD800]),
        ResultString::from("�")
    );
}
