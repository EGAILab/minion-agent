use minion_agent::tools::{PreparedValue, builtin::prepare_edit_arguments};
use serde_json::json;

#[test]
fn preparation_preserves_json_parse_overflow_and_negative_zero() {
    let raw = json!({"path":"f", "edits": "[{\"oldText\":\"a\",\"newText\":\"b\",\"extra\":1e9999,\"zero\":-0,\"rounded\":9007199254740993}]"});
    let value = prepare_edit_arguments(raw).unwrap();
    let edit = &value["edits"].as_array().unwrap()[0];
    assert_eq!(edit["extra"].as_f64(), Some(f64::INFINITY));
    let zero = edit["zero"].as_f64().unwrap();
    assert_eq!(zero, 0.0);
    assert!(zero.is_sign_negative());
    assert_eq!(edit["rounded"].as_f64(), Some(9007199254740992.0));
}

#[test]
fn preparation_accepts_single_object_and_appends_legacy_fields() {
    let raw = json!({"path":"f", "edits":{"oldText":"a","newText":"b","extra":true},"oldText":"c","newText":"d","kept":17});
    let value = prepare_edit_arguments(raw).unwrap();
    assert_eq!(
        value,
        PreparedValue::from(
            json!({"path":"f","edits":[{"oldText":"a","newText":"b","extra":true},{"oldText":"c","newText":"d"}],"kept":17})
        )
    );
}

#[test]
fn preparation_leaves_invalid_json_and_non_edit_objects_unchanged() {
    for raw in [
        json!({"edits":"[NaN]"}),
        json!({"edits":"{}"}),
        json!({"edits":{"oldText":"a"}}),
        json!(false),
        json!(null),
    ] {
        assert_eq!(
            prepare_edit_arguments(raw.clone()).unwrap(),
            PreparedValue::from(raw)
        );
    }
}
