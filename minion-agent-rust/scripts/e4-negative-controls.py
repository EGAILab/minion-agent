"""Emit guarded apply_patch source faults; never writes the implementation itself.

Run from the Rust workspace: python scripts/e4-negative-controls.py NAME [--restore].
Apply the emitted patch, run the named witness, and always apply --restore afterward.
Compilation errors are not successful controls: each must reach an assertion/test failure.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ENV = "crates/minion-agent/src/execution/environment.rs"
COMPOSE = "crates/minion-agent/src/tools/builtin/environment.rs"
SUBPROCESS = "crates/minion-agent/src/execution/subprocess.rs"

FAULTS = {
    "snapshot-public": (ENV, [("    entries: EnvEntries,", "    pub entries: EnvEntries,")], "doc:execution::environment::EnvSnapshot"),
    "platform-public": (SUBPROCESS, [
        ("pub struct LocalSubprocess {", "pub struct LocalSubprocess {\n    pub platform: Platform,"),
        ("            base_env: std::env::vars().collect(),", "            base_env: std::env::vars().collect(),\n            platform: Platform::local(),"),
        ("        Platform::local()", "        self.platform"),
    ], "doc:execution::subprocess::LocalSubprocess"),
    "host-baseline": (SUBPROCESS, [("EnvSnapshot::configured(self.platform(), &self.base_env)", "EnvSnapshot::configured(self.platform(), &std::env::vars().collect())")], "local_declaration_and_configured_baseline_stay_fixed"),
    "ascii-lookup": (ENV, [(".map(native_uppercase)", ".map(|unit| if (0x61..=0x7a).contains(&unit) { unit - 32 } else { unit })")], "snapshot_native_lookup_counts_units_without_consumer_arbitration"),
    "lowercase-arbitration": (COMPOSE, [("    for (name, upper) in names.into_iter().zip(uppers) {", "    for (name, _upper) in names.into_iter().zip(uppers) {\n        let upper = name.to_lowercase();")], "windows_arbitration_is_uppercase_and_utf16_first_in_both_orders"),
    "ascii-arbitration": (COMPOSE, [("    for (name, upper) in names.into_iter().zip(uppers) {", "    for (name, _upper) in names.into_iter().zip(uppers) {\n        let upper = name.to_ascii_uppercase();")], "windows_arbitration_is_uppercase_and_utf16_first_in_both_orders"),
    # This is the exact casefold result for the characterized sharp-S/dotless-I controls.
    "casefold-arbitration": (COMPOSE, [("    for (name, upper) in names.into_iter().zip(uppers) {", "    for (name, _upper) in names.into_iter().zip(uppers) {\n        let upper = name.to_lowercase().replace('ß', \"ss\");")], "windows_arbitration_is_uppercase_and_utf16_first_in_both_orders"),
    "per-surrogate-replacement": (COMPOSE, [("                let unit = error.unpaired_surrogate();", "                let _unit = error.unpaired_surrogate();\n                output.extend([0xef, 0xbf, 0xbd]);\n                continue;\n                #[allow(unreachable_code)]\n                let unit = _unit;")], "windows_pairs_and_lone_units_use_generalized_utf8"),
    "pair-uncombined": (COMPOSE, [("    for decoded in char::decode_utf16(value.code_units().iter().copied()) {", "    for decoded in value.code_units().iter().flat_map(|unit| char::decode_utf16(std::iter::once(*unit))) {")], "windows_pairs_and_lone_units_use_generalized_utf8"),
    "strip-bom": (COMPOSE, [("String::from_utf8_lossy(&v).into_owned()", "String::from_utf8_lossy(&v).trim_start_matches('\\u{feff}').to_owned()")], "posix_lossless_entries_exact_lookup_and_node_decode"),
    "keep-invalid-name": (COMPOSE, [("String::from_utf8(n)\n                .ok()", "Some(String::from_utf8_lossy(&n).into_owned())")], "posix_lossless_entries_exact_lookup_and_node_decode"),
    "last-name-wins": (COMPOSE, [("    let mut chosen = BTreeSet::new();", "    names.reverse();\n    let mut chosen = BTreeSet::new();")], "windows_arbitration_is_uppercase_and_utf16_first_in_both_orders"),
    "case-insensitive-removal": (COMPOSE, [("        env.remove(*name);", "        env.retain(|key, _| !key.eq_ignore_ascii_case(name));")], "fake_windows_world_composition_uses_provider_not_host"),
    "inherit-false-leaks-baseline": (SUBPROCESS, [("        if !options.inherit_env {\n            command.env_clear();", "        if !options.inherit_env {\n            command.env_clear();\n            command.envs(&self.base_env);")], "configured_baseline_rebuild_matches_real_inheritance_and_false_is_exact"),
}

if sys.argv[1] == "--list":
    print(json.dumps({k: {"path": v[0], "witness": v[2]} for k, v in FAULTS.items()}))
else:
    path, replacements, _ = FAULTS[sys.argv[1]]
    restore = "--restore" in sys.argv
    original = Path(path).read_text(encoding="utf-8")
    changed = original
    for good, bad in replacements:
        before, after = (bad, good) if restore else (good, bad)
        if changed.count(before) != 1:
            raise SystemExit(f"control anchor must occur exactly once: {before!r}")
        changed = changed.replace(before, after)
    import difflib
    diff = list(difflib.unified_diff(original.splitlines(), changed.splitlines(), n=3, lineterm=""))
    print("*** Begin Patch\n*** Update File: " + str(Path(path).resolve()).replace("\\", "/"))
    for line in diff[2:]:
        print("@@" if line.startswith("@@") else line)
    print("*** End Patch")
