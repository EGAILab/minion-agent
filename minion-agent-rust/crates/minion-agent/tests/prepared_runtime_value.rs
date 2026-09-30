use minion_agent::tools::{PreparedNumber, PreparedValue};
use serde_json::json;

#[test]
fn prepared_domain_preserves_every_binary64_category() {
    for value in [
        0.0,
        -0.0,
        1e308,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        let prepared = PreparedValue::number(value);
        let observed = prepared.as_f64().unwrap();
        if value.is_nan() {
            assert!(observed.is_nan());
        } else {
            assert_eq!(observed.to_bits(), value.to_bits());
        }
        assert_eq!(prepared.try_to_json().is_ok(), value.is_finite());
    }
    assert_eq!(
        PreparedNumber::from_f64(f64::INFINITY),
        PreparedNumber::PositiveInfinity
    );
    assert_eq!(
        PreparedNumber::from_f64(f64::NEG_INFINITY),
        PreparedNumber::NegativeInfinity
    );
    assert_eq!(PreparedNumber::from_f64(f64::NAN), PreparedNumber::NaN);
}

#[test]
fn json_round_trip_is_exact_and_keeps_missing_null_false_and_empty_distinct() {
    let raw = json!({"null": null, "false": false, "empty": [], "object": {},
                     "string": "", "integer": 9007199254740993_u64, "negativeZero": -0.0});
    let prepared = PreparedValue::from(raw.clone());
    assert_eq!(prepared.try_to_json().unwrap(), raw);
    assert!(prepared.get("missing").is_none());
    assert_eq!(prepared.get("null"), Some(&PreparedValue::Null));
    assert_eq!(prepared.get("false"), Some(&PreparedValue::Bool(false)));
    assert!(
        prepared
            .get("negativeZero")
            .unwrap()
            .as_f64()
            .unwrap()
            .is_sign_negative()
    );
}

#[test]
fn nested_non_finite_values_are_neither_strings_nor_null_and_conversion_is_fallible() {
    let mut prepared = PreparedValue::from(json!({"a/b~c": [0]}));
    prepared.as_object_mut().unwrap().insert(
        "a/b~c".into(),
        PreparedValue::Array(vec![PreparedValue::number(f64::NAN)]),
    );
    let before = prepared.clone();
    let error = prepared.try_to_json().unwrap_err();
    assert_eq!(error.pointer, "/a~1b~0c/0");
    assert_eq!(prepared, before);
    assert!(
        prepared.get("a/b~c").unwrap().as_array().unwrap()[0]
            .as_f64()
            .unwrap()
            .is_nan()
    );
    assert_ne!(PreparedValue::number(f64::NAN), PreparedValue::Null);
    assert_ne!(
        PreparedValue::number(f64::INFINITY),
        PreparedValue::String("Infinity".into())
    );
}

#[test]
fn numbers_are_not_obtained_by_coercing_strings() {
    for text in ["Infinity", "-Infinity", "NaN", "1", "outside"] {
        let prepared = PreparedValue::from(json!(text));
        assert_eq!(prepared.as_f64(), None);
        assert_eq!(prepared.as_str(), Some(text));
    }
}
