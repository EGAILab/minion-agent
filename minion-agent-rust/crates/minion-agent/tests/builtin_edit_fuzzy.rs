use minion_agent::tools::builtin::fuzzy_normalize;

#[test]
fn normalization_is_unicode16_and_uses_javascript_whitespace() {
    assert_eq!(
        fuzzy_normalize("Ａ’\u{feff}\nB\u{85}\nC\u{1c}\n\u{a7f1}\n\u{1ccd6}").unwrap(),
        "A'\nB\u{85}\nC\u{1c}\n\u{a7f1}\nA"
    );
    assert_eq!(fuzzy_normalize("e\u{301} — “x”   ").unwrap(), "é - \"x\"");
}

#[test]
fn pinned_fuzzy_fixture_is_replayed_against_production() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../conformance/agent/fixtures/wp132-fuzzy-normalize/fuzzy_normalize.json");
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
    for case in value["cases"].as_array().expect("fixture cases") {
        assert_eq!(
            fuzzy_normalize(case["text"].as_str().unwrap()).unwrap(),
            case["normalized"].as_str().unwrap()
        );
    }
}
