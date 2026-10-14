use super::*;
use serde_json::Value;
use std::sync::atomic::AtomicUsize;

#[cfg(windows)]
#[test]
fn existing_cwd_normalizer_colon_component_characterization() {
    // Pure computation through the certified resolver; no native operation.
    for tail in ["file:///C:/a%2Fb", "file:///C:/a%2Fb\0"] {
        let actual = super::super::path::resolve(Path::new("C:\\x"), &tail.into());
        println!(
            "legacy cwd resolution: {:?}",
            super::super::filesystem::resolve_local_path(Path::new("C:\\x"), tail)
        );
        let expected_bad = if tail.ends_with('\0') {
            "C:\\x\\file:C:a%2Fb\0"
        } else {
            "C:\\x\\file:C:a%2Fb"
        };
        assert_eq!(
            actual.as_str(),
            Some(expected_bad),
            "unchanged-normalizer-characterization"
        );
    }
}

fn string(value: &Value) -> ResultString {
    if let Some(value) = value.as_str() {
        return value.into();
    }
    ResultString::from_code_units(
        value["utf16"]
            .as_array()
            .expect("UTF16 carrier")
            .iter()
            .map(|u| u.as_u64().unwrap() as u16)
            .collect(),
    )
}
fn selected<'a>(value: &'a Value, platform: &str) -> &'a Value {
    value.get(platform).unwrap_or(value)
}
fn entries(value: &Value) -> Vec<(ResultString, ResultString)> {
    value
        .as_array()
        .map(|a| a.iter().map(|p| (string(&p[0]), string(&p[1]))).collect())
        .unwrap_or_default()
}

#[tokio::test]
async fn spawn_nul_canonical_real_provider() {
    let document: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../conformance/agent/spawn-nul/spawn-nul.json"
    )))
    .unwrap();
    let platform = if cfg!(windows) { "win32" } else { "linux" };
    let rows = document["spawn_nul"]["cases"].as_array().unwrap();
    assert_eq!(rows.len(), 57);
    for row in rows {
        let id = row["id"].as_str().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut provider = LocalSubprocess::new(".")
            .with_logical_cwd(string(selected(&row["provider_cwd"], platform)))
            .with_lossless_base_env(entries(&row["baseline"]));
        provider.native_spawn_calls = Some(calls.clone());
        let argv = row["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(string)
            .collect::<Vec<_>>();
        let controller = super::super::CancellationController::default();
        if row["pre_aborted"] == true {
            controller.abort();
        }
        let options = LosslessSpawnOptions {
            cwd: (!row["cwd"].is_null()).then(|| string(selected(&row["cwd"], platform))),
            env: entries(&row["env"]),
            inherit_env: row["inherit_env"].as_bool().unwrap(),
            signal: Some(Arc::new(controller.signal())),
            ..LosslessSpawnOptions::default()
        };
        let error = provider
            .spawn_lossless(&argv, options)
            .await
            .err()
            .expect("scenario rejects");
        let expected = &row["expect"][platform];
        assert_eq!(
            serde_json::to_value(error.code).unwrap(),
            expected["code"],
            "{id}: code"
        );
        if expected["message"] == "os" {
            assert!(
                error.nul_diagnostic().is_none(),
                "{id}: native error must stay unmarked"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1, "{id}: native boundary");
        } else {
            let actual = error
                .nul_diagnostic()
                .cloned()
                .unwrap_or_else(|| error.message.as_str().into());
            assert_eq!(
                actual,
                string(&expected["message"]),
                "{id}: exact diagnostic units"
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "{id}: no native spawn after rejection"
            );
        }
    }
}

#[tokio::test]
async fn spawn_nul_legacy_and_ordered_carriers_share_validation() {
    let provider = LocalSubprocess::new(".").with_base_env(BTreeMap::new());
    let argv = vec!["missing-executable".to_owned()];
    let legacy = provider
        .spawn(
            &argv,
            SpawnOptions {
                env: BTreeMap::from([("Z".into(), "z\0".into()), ("A".into(), "a\0".into())]),
                inherit_env: false,
                ..SpawnOptions::default()
            },
        )
        .await
        .err()
        .unwrap();
    let ordered = provider
        .spawn_lossless(
            &["missing-executable".into()],
            LosslessSpawnOptions {
                env: vec![("Z".into(), "z\0".into()), ("A".into(), "a\0".into())],
                inherit_env: false,
                ..LosslessSpawnOptions::default()
            },
        )
        .await
        .err()
        .unwrap();
    assert!(
        legacy.message.contains("options.env['A']"),
        "legacy-map-order"
    );
    let first = if cfg!(windows) { "A" } else { "Z" };
    assert!(
        ordered.message.contains(&format!("options.env['{first}']")),
        "ordered-overlay-order"
    );
}

#[test]
fn spawn_nul_unrelated_invalid_input_is_not_a_typed_rejection() {
    let error = map_spawn_error(io::Error::new(
        io::ErrorKind::InvalidInput,
        "ordinary InvalidInput",
    ));
    assert_eq!(error.code, SubprocessErrorCode::SpawnError);
    assert_eq!(error.message, "ordinary InvalidInput");
    assert!(error.nul_diagnostic().is_none());
    let forged = SubprocessError::new(
        SubprocessErrorCode::SpawnError,
        "The argument 'file' must be a string without null bytes. Received 'x\\x00'",
    );
    assert!(forged.nul_diagnostic().is_none());
}

#[tokio::test]
async fn spawn_nul_long_numeric_name_is_ordinary() {
    let provider = LocalSubprocess::new(".").with_base_env(BTreeMap::new());
    let error = provider
        .spawn_lossless(
            &["missing".into()],
            LosslessSpawnOptions {
                env: vec![
                    ("1".repeat(4301).into(), "ok".into()),
                    ("K".into(), "bad\0".into()),
                ],
                inherit_env: false,
                ..LosslessSpawnOptions::default()
            },
        )
        .await
        .err()
        .unwrap();
    assert!(
        error.message.contains("options.env['K']"),
        "long-numeric-name-no-conversion-error"
    );
}
