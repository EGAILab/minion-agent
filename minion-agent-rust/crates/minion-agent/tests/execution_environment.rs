use async_trait::async_trait;
use minion_agent::{
    execution::{
        EnvEntries, EnvSnapshot, EnvValue, ExecutionWorldIdentity, LocalSubprocess, Platform,
        Process, SpawnOptions, Subprocess, SubprocessError, SubprocessErrorCode,
        pinned_windows_uppercase,
    },
    javascript::JsString,
    tools::builtin::environment::{
        compose_spawn_environment, node_environment_view, windows_spawn_environment,
    },
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, sync::Mutex};

fn wide(pairs: &[(&str, &str)]) -> EnvSnapshot {
    EnvSnapshot::new(EnvEntries::Windows(
        pairs
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect(),
    ))
}

fn bytes(pairs: &[(Vec<u8>, Vec<u8>)]) -> EnvSnapshot {
    EnvSnapshot::new(EnvEntries::Posix(pairs.to_vec()))
}

#[test]
fn vendored_native_table_has_exact_shared_bytes() {
    let blob = include_bytes!("../src/execution/windows_upcase.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(blob)),
        "78580c216df002802980880f491c6ef31dc594f8c8ca2cb23f27b0d90f637df8"
    );
    let data: serde_json::Value = serde_json::from_slice(blob).unwrap();
    assert_eq!(data["map"].as_object().unwrap().len(), 973);
    for (lower, upper) in [
        (0x61, 0x41),
        (0xe9, 0xc9),
        (0xdf, 0xdf),
        (0x131, 0x131),
        (0xd800, 0xd800),
    ] {
        assert_eq!(pinned_windows_uppercase(lower), upper);
    }
}

#[cfg(windows)]
#[test]
fn live_native_table_matches_pinned_table_on_this_host() {
    for unit in 0..=u16::MAX {
        assert_eq!(
            minion_agent_native_env::uppercase_unit(unit),
            pinned_windows_uppercase(unit),
            "unit {unit:04x}"
        );
    }
}

#[test]
fn snapshot_native_lookup_counts_units_without_consumer_arbitration() {
    let s = wide(&[
        ("PROGRAMFILES", "D:/world"),
        ("Qé", "acute"),
        ("Qß", "sharp"),
        ("Qss", "ss"),
        ("Qı", "dotless"),
        ("QI", "ascii"),
        ("Q😀a", "astral"),
    ]);
    assert_eq!(s.platform(), Platform::Windows);
    for (name, value) in [
        ("ProgramFiles", "D:/world"),
        ("QÉ", "acute"),
        ("Qß", "sharp"),
        ("QSS", "ss"),
        ("Qı", "dotless"),
        ("qi", "ascii"),
        ("Q😀a", "astral"),
    ] {
        assert_eq!(s.get(name), Some(EnvValue::Windows(&JsString::from(value))));
    }
    assert_eq!(s.get("Q😀b"), None);
    assert!(matches!(s.entries(),EnvEntries::Windows(v) if v.len()==7));
    let lone = JsString::from_code_units(vec![0x51, 0xd800, 0x61]);
    let s = EnvSnapshot::new(EnvEntries::Windows(vec![(lone.clone(), "v".into())]));
    assert_eq!(s.get_windows(&lone), Some(&JsString::from("v")));
    assert_eq!(
        s.get_windows(&JsString::from_code_units(vec![0x51, 0xd800, 0x62])),
        None
    );
}

#[test]
fn snapshots_and_consumer_copies_are_isolated() {
    let mut p = Fake::new(wide(&[("K", "old")]));
    let old = p.base_env();
    let mut copy = old.copy();
    if let EnvEntries::Windows(entries) = &mut copy {
        entries[0].1 = "copy".into();
    }
    *p.snapshot.get_mut().unwrap() = wide(&[("K", "new")]);
    assert_eq!(
        old.get("K"),
        Some(EnvValue::Windows(&JsString::from("old")))
    );
    assert_eq!(
        p.base_env().get("K"),
        Some(EnvValue::Windows(&JsString::from("new")))
    );
}

#[test]
fn posix_lossless_entries_exact_lookup_and_node_decode() {
    let cases: &[(&[u8], &str)] = &[
        (b"a\xffb", "a\u{fffd}b"),
        (b"a\xe1\x80", "a\u{fffd}"),
        (b"a\xe1\x80b", "a\u{fffd}b"),
        (b"a\xf0\x90\x80", "a\u{fffd}"),
        (b"a\xed\xa0\x80b", "a\u{fffd}\u{fffd}\u{fffd}b"),
        (b"a\xc0\xafb", "a\u{fffd}\u{fffd}b"),
        (b"\xe2\x82\xac\xe1\x80\xe2\x82\xac", "€\u{fffd}€"),
        (b"\xef\xbb\xbfa", "\u{feff}a"),
    ];
    for (value, expected) in cases {
        let s = bytes(&[(b"V".to_vec(), value.to_vec())]);
        assert_eq!(s.get("V"), Some(EnvValue::Bytes(value)));
        assert_eq!(node_environment_view(&s)["V"], *expected);
    }
    let s = bytes(&[
        (b"N_\xff".to_vec(), b"x".to_vec()),
        ("N_é".as_bytes().to_vec(), b"y".to_vec()),
        (b"Path".to_vec(), b"a".to_vec()),
        (b"PATH".to_vec(), b"b".to_vec()),
    ]);
    let view = node_environment_view(&s);
    assert!(!view.contains_key("N_�"));
    assert_eq!(view["N_é"], "y");
    assert_eq!(view["Path"], "a");
    assert_eq!(view["PATH"], "b");
    assert_eq!(s.get("path"), None);
}

#[test]
fn windows_pairs_and_lone_units_use_generalized_utf8() {
    for lone in [0xd800, 0xdc80] {
        let s = EnvSnapshot::new(EnvEntries::Windows(vec![
            (
                "V".into(),
                JsString::from_code_units(vec![0x61, lone, 0x62]),
            ),
            (JsString::from_code_units(vec![0x4e, lone]), "drop".into()),
            (
                "W_PAIR".into(),
                JsString::from_code_units(vec![0x61, 0xd83d, 0xde00, 0x62]),
            ),
        ]));
        let view = node_environment_view(&s);
        assert_eq!(view.len(), 2);
        assert_eq!(view["V"], "a���b");
        assert_eq!(view["W_PAIR"], "a😀b");
    }
    assert_eq!(
        node_environment_view(&wide(&[("V", "😀")])),
        node_environment_view(&EnvSnapshot::new(EnvEntries::Windows(vec![(
            "V".into(),
            JsString::from_code_units(vec![0xd83d, 0xde00])
        )])))
    );
}

#[test]
fn windows_arbitration_is_uppercase_and_utf16_first_in_both_orders() {
    for (a, b, winner) in [
        ("Qß", "Qss", "Qss"),
        ("Qı", "QI", "QI"),
        ("XK", "xk", "XK"),
        ("xK", "xk", "xK"),
        ("Xk", "xK", "Xk"),
    ] {
        for reverse in [false, true] {
            let mut pairs = vec![(a.to_string(), "a".into()), (b.to_string(), "b".into())];
            if reverse {
                pairs.reverse();
            }
            let env = pairs.into_iter().collect();
            let output = windows_spawn_environment(&env).unwrap();
            assert_eq!(output.len(), 1);
            assert_eq!(output[winner], if winner == a { "a" } else { "b" });
        }
    }
}

#[test]
fn fake_windows_world_composition_uses_provider_not_host() {
    let provider = Fake::new(wide(&[
        ("PROGRAMFILES", "D:/world"),
        ("PATH", "world-path"),
        ("Minion_Session_Id", "stale"),
        ("MINION_MODEL", "old"),
    ]));
    assert_eq!(provider.platform(), Platform::Windows);
    let snapshot = provider.base_env();
    let removes = ["MINION_SESSION_ID", "MINION_MODEL"];
    let env = compose_spawn_environment(
        &snapshot,
        &removes,
        &BTreeMap::from([("MINION_SESSION_ID".into(), "live".into())]),
    )
    .unwrap();
    assert_eq!(env["PROGRAMFILES"], "D:/world");
    assert_eq!(env["PATH"], "world-path");
    assert_eq!(env["MINION_SESSION_ID"], "live");
    assert!(!env.contains_key("Minion_Session_Id"));
    assert!(!env.contains_key("MINION_MODEL"));
    let env = compose_spawn_environment(&snapshot, &removes, &BTreeMap::new()).unwrap();
    assert_eq!(env["Minion_Session_Id"], "stale");
}

#[test]
fn local_declaration_and_configured_baseline_stay_fixed() {
    let local = LocalSubprocess::new(".")
        .with_base_env(BTreeMap::from([("ONLY_PROVIDER".into(), "value".into())]));
    assert_eq!(local.platform(), Platform::local());
    let s = local.base_env();
    assert_eq!(node_environment_view(&s).len(), 1);
    assert_eq!(
        node_environment_view(&s),
        BTreeMap::from([("ONLY_PROVIDER".into(), "value".into())])
    );
    let second = local.clone().with_base_env(BTreeMap::new());
    assert!(node_environment_view(&second.base_env()).is_empty());
    assert_eq!(
        node_environment_view(&local.base_env())["ONLY_PROVIDER"],
        "value"
    );
    assert_eq!(s, local.base_env());
}

async fn child(
    provider: &dyn Subprocess,
    env: BTreeMap<String, String>,
    inherit_env: bool,
) -> String {
    let node = std::process::Command::new("node")
        .arg("-p")
        .arg("process.execPath")
        .output()
        .unwrap();
    let path = String::from_utf8(node.stdout).unwrap().trim().to_owned();
    let argv = vec![
        path,
        "-e".into(),
        "process.stdout.write(JSON.stringify(Object.entries(process.env).sort()))".into(),
    ];
    let p = provider
        .spawn(
            &argv,
            SpawnOptions {
                env,
                inherit_env,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let mut bytes = Vec::new();
    while let Some(chunk) = p.stdout().unwrap().read_chunk().await.unwrap() {
        bytes.extend(chunk);
    }
    assert_eq!(p.wait().await.unwrap().exit_code, Some(0));
    String::from_utf8(bytes).unwrap()
}

#[tokio::test]
async fn configured_baseline_rebuild_matches_real_inheritance_and_false_is_exact() {
    let mut env: BTreeMap<String, String> = std::env::vars().collect();
    env.insert("WpE4_ProviderOnly".into(), "value".into());
    let provider = LocalSubprocess::new(".").with_base_env(env.clone());
    let snapshot = provider.base_env();
    assert_eq!(
        child(&provider, BTreeMap::new(), true).await,
        child(&provider, node_environment_view(&snapshot), false).await
    );
    let supplied = BTreeMap::from([("ONLY_CALLER".into(), "exact".into())]);
    let output = child(&provider, supplied, false).await;
    let rows: Vec<(String, String)> = serde_json::from_str(&output).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows, vec![("ONLY_CALLER".into(), "exact".into())]);
}

struct Fake {
    snapshot: Mutex<EnvSnapshot>,
    world: ExecutionWorldIdentity,
}
impl Fake {
    fn new(snapshot: EnvSnapshot) -> Self {
        Self {
            snapshot: Mutex::new(snapshot),
            world: ExecutionWorldIdentity::local(),
        }
    }
}
#[async_trait]
impl Subprocess for Fake {
    fn cwd(&self) -> &Path {
        Path::new(".")
    }
    fn execution_world(&self) -> &ExecutionWorldIdentity {
        &self.world
    }
    fn platform(&self) -> Platform {
        Platform::Windows
    }
    fn base_env(&self) -> EnvSnapshot {
        self.snapshot.lock().unwrap().clone()
    }
    async fn spawn(
        &self,
        _: &[String],
        _: SpawnOptions,
    ) -> Result<std::sync::Arc<dyn Process>, SubprocessError> {
        Err(SubprocessError::new(
            SubprocessErrorCode::SpawnError,
            "fake snapshot-only provider",
        ))
    }
}
