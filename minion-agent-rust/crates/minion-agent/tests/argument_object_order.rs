//! Development witnesses for the K1 storage primitive; not the complete corpus gate.
use minion_agent::{
    argument_object::{ArgumentObject, array_index},
    llm::{RawString, RawValue},
    tools::PreparedValue,
};

fn keys(raw: &RawValue) -> Vec<String> {
    let RawValue::Object(object) = raw else {
        panic!("object");
    };
    object.keys().map(|key| key.to_string().unwrap()).collect()
}

#[test]
fn raw_decode_and_serialization_keep_the_same_order_recursively() {
    let text = r#"{"z":{"b":1,"2":2,"1":3},"2":2,"1":1,"a":0}"#;
    let raw = RawValue::decode(text).unwrap();
    assert_eq!(keys(&raw), ["1", "2", "z", "a"]);
    assert_eq!(keys(&raw["z"]), ["1", "2", "b"]);
    let serialized = serde_json::to_string(&raw).unwrap();
    assert_eq!(serialized, r#"{"1":1,"2":2,"z":{"1":3,"2":2,"b":1},"a":0}"#);
    let replay: RawValue = serde_json::from_str(&serialized).unwrap();
    assert_eq!(keys(&replay), keys(&raw));
    assert_eq!(keys(&replay["z"]), keys(&raw["z"]));
}

#[test]
fn prepared_and_replacement_values_do_not_impose_sorted_key_order() {
    let raw = RawValue::decode(r#"{"z":1,"a":2,"1":3}"#).unwrap();
    let mut prepared = PreparedValue::from(raw);
    prepared["0"] = PreparedValue::Bool(true);
    let object = prepared.as_object().unwrap();
    assert_eq!(
        object
            .keys()
            .map(|key| key.as_str().unwrap())
            .collect::<Vec<_>>(),
        ["0", "1", "z", "a"]
    );
    assert_eq!(
        serde_json::to_string(&prepared.try_to_json().unwrap()).unwrap(),
        r#"{"0":true,"1":3,"z":1,"a":2}"#
    );
}

#[test]
fn duplicate_properties_keep_first_position_and_last_value() {
    let raw = RawValue::decode(r#"{"z":1,"a":2,"z":3,"__proto__":4}"#).unwrap();
    assert_eq!(keys(&raw), ["z", "a", "__proto__"]);
    assert_eq!(raw["z"].as_f64(), Some(3.0));
}

#[test]
fn explicit_wrong_storage_controls_are_distinguished() {
    let entries = ["z", "2", "1", "a"];
    let correct: ArgumentObject<RawString, ()> =
        entries.into_iter().map(|key| (key.into(), ())).collect();
    let actual = correct
        .keys()
        .map(|key| key.to_string().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(actual, ["1", "2", "z", "a"]);
    assert_ne!(actual, entries); // insertion-without-index-first mutant
    assert_ne!(actual, ["1", "2", "a", "z"]); // sorted-map mutant
    for wrong in ["01", "4294967295", "999999999999999999999999999999"] {
        assert_eq!(array_index(&wrong.encode_utf16().collect::<Vec<_>>()), None);
    }
}
