use base64::{Engine as _, engine::general_purpose::STANDARD};
use minion_agent::tools::builtin::generate_edit_details;

#[test]
fn every_successful_authority_corpus_diff_and_patch_is_byte_exact() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../conformance/agent/builtin-mutation");
    let mut compared = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if !path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("builtin-edit-corpus-")
        {
            continue;
        }
        let source = std::fs::read_to_string(path).unwrap();
        // Parse independent case blocks: Rust's certified string domain rejects the
        // explicitly flagged lone-surrogate case, not every other case in its document.
        for block in source.split("\n    - id:").skip(1) {
            let mut lines = block.lines();
            let body = format!(
                "id:{}\n{}",
                lines.next().unwrap(),
                lines
                    .map(|l| l.strip_prefix("      ").unwrap_or(l))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            let case: serde_json::Value = match serde_yaml::from_str(&body) {
                Ok(case) => case,
                Err(error) => {
                    assert!(
                        body.contains("unpaired_surrogate_arguments: true"),
                        "unexpected fixture decode failure: {error}"
                    );
                    assert!(serde_json::from_str::<serde_json::Value>(r#""\ud800""#).is_err());
                    continue;
                }
            };
            if case["expect"]["is_error"] != false {
                continue;
            }
            let original = STANDARD
                .decode(case["fixture"][0]["file"]["base64"].as_str().unwrap())
                .unwrap();
            let final_bytes = STANDARD
                .decode(case["expect"]["files_after"][0]["base64"].as_str().unwrap())
                .unwrap();
            let normalize = |bytes: &[u8]| {
                let text = String::from_utf8_lossy(bytes);
                text.strip_prefix('\u{feff}')
                    .unwrap_or(&text)
                    .replace("\r\n", "\n")
                    .replace('\r', "\n")
            };
            assert_eq!(
                generate_edit_details(
                    case["arguments"]["path"].as_str().unwrap(),
                    &normalize(&original),
                    &normalize(&final_bytes)
                ),
                case["expect"]["details"],
                "{}",
                case["id"]
            );
            compared += 1;
        }
    }
    assert!(
        compared > 100,
        "must discover successful corpus cases, got {compared}"
    );
}
