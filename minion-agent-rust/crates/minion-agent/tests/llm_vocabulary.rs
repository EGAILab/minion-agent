use minion_agent::llm::{ModelIdentity, ModelIdentityError};

#[test]
fn model_identity_requires_all_three_non_empty_components() {
    assert!(ModelIdentity::new("openai", "responses", "gpt-5").is_ok());

    assert_eq!(
        ModelIdentity::new("", "responses", "gpt-5")
            .expect_err("an empty provider must fail")
            .field(),
        "provider"
    );
    assert_eq!(
        ModelIdentity::new("openai", "", "gpt-5")
            .expect_err("an empty api must fail")
            .field(),
        "api"
    );
    assert_eq!(
        ModelIdentity::new("openai", "responses", "")
            .expect_err("an empty model id must fail")
            .field(),
        "model_id"
    );
}

#[test]
fn model_identity_has_value_equality_and_round_trips() {
    let identity = ModelIdentity::new("openai", "responses", "gpt-5")
        .expect("complete model identity must be valid");
    let equal_value = ModelIdentity::new(
        String::from("openai"),
        String::from("responses"),
        String::from("gpt-5"),
    )
    .expect("separately allocated equal strings must be equal identities");

    assert_eq!(identity, equal_value);

    let json = serde_json::to_value(&identity).expect("identity must serialize");
    assert_eq!(
        json,
        serde_json::json!({
            "provider": "openai",
            "api": "responses",
            "model_id": "gpt-5"
        })
    );
    assert_eq!(
        serde_json::from_value::<ModelIdentity>(json).expect("identity must deserialize"),
        identity
    );
}

#[test]
fn model_identity_deserialization_uses_the_same_validation_boundary() {
    let error = serde_json::from_value::<ModelIdentity>(serde_json::json!({
        "provider": "openai",
        "api": "",
        "model_id": "gpt-5"
    }))
    .expect_err("deserialization must not bypass identity validation");

    assert!(error.to_string().contains("api"));

    let direct = ModelIdentity::new("openai", "", "gpt-5")
        .expect_err("direct construction must reject the same value");
    assert_eq!(
        direct,
        ModelIdentityError::MissingComponent { field: "api" }
    );
}
