"""Compiled, fired, named-witness controls. Build/setup failures are INVALID.

Only an explicit disposable control-code tree and a disjoint controls target
may be mutated. Every baseline and restored baseline must be green. No cleanup.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from div009_controls import controls as div009_controls

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cache_safety", HERE / "cargo-cache-safety.py")
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)
ROOT = safety.PROJECT if os.name == "nt" else Path("/tmp")
EXEC = "crates/minion-agent/src/execution/"
SELECTED = "execution::path_resolve::tests::node_frozen_selected_entry_and_logical_rows"
PATHS = "execution::path::tests::"


def controls():
    # name, file, old, replacement, witness, intended diagnostic, native-only
    rows = [
        ("colon-separator-dropped", EXEC + "path_resolve.rs", "if !result.is_empty() {",
         'if { FIRE !result.is_empty() && result.last() != Some(&58) } {',
         "execution::path_resolve::tests::posix_colons_remain_ordinary_components", "file:/C:/a%2Fb", False),
        ("mismatch-always-root", EXEC + "path_resolve.rs",
         "if lower(&units[..units.len().min(2)]) != lower(&device) && units.get(2) == Some(&92) {",
         "if { FIRE lower(&units[..units.len().min(2)]) != lower(&device) } {", SELECTED, "bmp-1", False),
        ("fall-through-selection", EXEC + "path_resolve.rs",
         ".filter(|value| !value.code_units().is_empty())",
         ".filter(|value| { FIRE !value.code_units().is_empty() && lower(&value.code_units()[..value.code_units().len().min(2)]) == lower(&device) })",
         SELECTED, "bmp-1", False),
        ("empty-is-value", EXEC + "path_resolve.rs",
         ".filter(|value| !value.code_units().is_empty())",
         ".filter(|value| { FIRE let _ = value; true })", SELECTED, "empty", False),
        ("scalar-index-not-unit-index", EXEC + "path_resolve.rs", "units.get(2) == Some(&92)",
         "{ FIRE char::decode_utf16(units.iter().copied()).nth(2).and_then(Result::ok) == Some('\\\\') }",
         SELECTED, "astral-backslash", False),
        ("one-replacement-native-view", EXEC + "path.rs", "Err(_) => output.extend_from_slice(&[0xfffd; 3]),",
         "Err(_) => { FIRE output.extend_from_slice(&[0xfffd; 1]); },", SELECTED, "lone-high-backslash", False),
        ("raw-native-entry", EXEC + "path.rs", ".map(|value| inherited_native_view(&native_units(Path::new(&value))))",
         ".map(|value| { FIRE native_units(Path::new(&value)) })", "native", "entry/lone-high-backslash", True),
        ("raw-native-cwd", EXEC + "path.rs",
         'read.map(|cwd| native_cwd_view(&native_units(&cwd), cfg!(windows)))',
         'read.map(|cwd| { FIRE native_units(&cwd) })',
         "native", "cwd/lone-high/absent", True),
        ("logical-projected", EXEC + "path.rs", "let mut expanded = raw.clone();",
         "FIRE let mut expanded = native_view(raw);", PATHS + "lexical_paths_do_not_project_surrogates", "assertion `left == right` failed", False),
        ("entry-preconditions-omitted", EXEC + "path.rs",
         'if cfg!(windows) {\n            for name in payload["absent_entries"].as_array().unwrap() {',
         'if { FIRE false } {\n            for name in payload["absent_entries"].as_array().unwrap() {',
         "native", "refusal/present-empty", True),
        ("search-case-sensitive-regression", "crates/minion-agent/src/tools/builtin/search_paths.rs",
         "crate::execution::node_lower::lower(left) == crate::execution::node_lower::lower(right)",
         "{ FIRE left == right }", "tools::builtin::search_paths::tests::windows_relative_is_case_insensitive_not_prefix_matching",
         "assertion `left == right` failed", False),
        ("host-unicode-lowercase-regression", EXEC + "node_lower.rs", "let points: Vec<u32>",
         "FIRE return String::from_utf16_lossy(units).to_lowercase().encode_utf16().collect(); #[allow(unreachable_code)] let points: Vec<u32>",
         "execution::node_lower::tests::node_unicode16_lowercase_is_not_host_unicode17", "assertion `left == right` failed", False),
        ("logical-provider-cwd-projected", EXEC + "path.rs", "let base = cwd.clone();",
         "let base = { FIRE native_view(cwd) };", PATHS + "logical_provider_cwd_stays_raw", "high", False),
        ("cwd-read-error-panics", EXEC + "path.rs", "read.map(|cwd| native_cwd_view(&native_units(&cwd), cfg!(windows)))",
         '{ FIRE Ok(native_cwd_view(&native_units(&read.expect("CONTROL CWD PANIC")), cfg!(windows))) }',
         PATHS + "native_cwd_read_failure_is_an_error_value", "CONTROL CWD PANIC", False),
        ("cwd-read-error-guesses-root", EXEC + "path_resolve.rs", "None => process_cwd()?,",
         "None => { FIRE process_cwd().unwrap_or_else(|_| { let mut guessed = device.clone(); guessed.push(92); FsPath::from_code_units(guessed) }) },",
         "execution::path_resolve::tests::native_cwd_error_is_returned_without_guessing", "called `Result::unwrap_err()` on an `Ok` value", False),
        # The original 15 run first. CE02 controls change exactly one boundary.
        ("CE02-C1", EXEC + "path.rs", "native_view(&FsPath::from_code_units(units[..end].to_vec()))",
         "{ FIRE native_view(&without_extended_prefix(FsPath::from_code_units(units[..end].to_vec()))) }",
         "ce02/prefix", ["extended/absent/cwd_view", "extended/absent/absolute_path/0", "extended/absent/logical/0", "extended/absent/absolute_path/1", "extended/absent/logical/1", "extended/absent/absolute_path/4", "extended/absent/logical/4"], True),
        ("CE02-C2", EXEC + "path.rs", "let end = if units.last() == Some(&separator) && !keep_root {",
         "let end = if { FIRE let _ = (separator, keep_root); false } {",
         "ce02/roots" if os.name == "nt" else PATHS + "native_cwd_posix_boundary",
         ["extended-root/absent/cwd_view"] if os.name == "nt" else "posix-cwd/trailing", False),
        ("CE02-C3", EXEC + "path.rs", "fn inherited_native_view(raw: &FsPath) -> FsPath {\n    native_view(raw)\n}",
         "fn inherited_native_view(raw: &FsPath) -> FsPath {\n    FIRE native_view(&without_extended_prefix(raw.clone()))\n}",
         "ce02/entry-prefix", ["plain/extended/entry_view", "plain/extended/absolute_path/0", "plain/extended/logical/0"], True),
        ("CE02-C4", EXEC + "path.rs", "fn inherited_native_view(raw: &FsPath) -> FsPath {\n    native_view(raw)\n}",
         "fn inherited_native_view(raw: &FsPath) -> FsPath {\n    FIRE native_cwd_view(raw, true)\n}",
         "ce02/entry-trim", ["plain/matching-trailing/entry_view"], True),
    ]
    return [(name, file, old, new.replace("FIRE", f'eprintln!("CONTROL FIRED {name}");'), witness, reason, native)
            for name, file, old, new, witness, reason, native in rows + div009_controls()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--check-anchors", action="store_true")
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--control-target", type=Path)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    selected = [row for row in controls() if not args.only or row[0] in args.only]
    assert not args.only or len(selected) == len(args.only), "unknown control"
    tree = safety.checked(args.tree, ROOT)
    originals = {file: (tree / file).read_text(encoding="utf-8") for _, file, *_ in controls()}
    original_bytes = {file: (tree / file).read_bytes() for file in originals}
    for name, file, old, *_ in controls():
        assert originals[file].count(old) == 1, f"INVALID ANCHOR {name}: count={originals[file].count(old)}"
    assert len(controls()) == 33, "original 15 plus CE02 four plus DIV009 fourteen"
    print("PATH CONTROL ANCHORS 33/33 MATCH ONCE (original 15, CE02 four, DIV009 fourteen)", flush=True)
    if args.check_anchors:
        return
    assert "control-code" in tree.parts, "explicit disposable control-code tree required"
    assert args.logs and args.control_target
    target = safety.checked(args.control_target, safety.PROJECT).resolve(strict=False) if os.name == "nt" else args.control_target
    gate = Path(os.environ["CARGO_TARGET_DIR"]).resolve(strict=False)
    assert target != gate and not target.is_relative_to(gate) and not gate.is_relative_to(target), "disjoint targets required"
    if os.name == "nt":
        for platform in ("windows", "linux"):
            shared_gate = (safety.PROJECT / ".tmp/cache/cargo-target" / platform).resolve(strict=False)
            assert not target.is_relative_to(shared_gate) and not shared_gate.is_relative_to(target), "shared gate cache cannot be a controls target"
    if os.name != "nt":
        assert target == Path("/control-target") and target.is_dir(), "explicit controls bind required"
    os.environ["CARGO_TARGET_DIR"] = str(target)
    logs = safety.checked(args.logs, ROOT)
    assert not logs.exists(), "NEW logs required"
    safety.checked(logs, ROOT).mkdir()

    def run(name, command):
        path = safety.checked(logs / (name + ".log"), ROOT)
        with safety.checked(path, ROOT).open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=tree, stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, path.read_text(encoding="utf-8", errors="replace")

    def build(name):
        safety.refresh(tree)
        code, output = run(name + "-build", ["cargo", "test", "--locked", "-p", "minion-agent", "--lib", "--no-run", "--message-format=json"])
        assert code == 0, f"INVALID BUILD {name}: exit {code}"
        rows = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        binaries = [row["executable"] for row in rows if row.get("reason") == "compiler-artifact" and row.get("executable") and row.get("target", {}).get("name") == "minion_agent"]
        assert len(binaries) == 1, f"INVALID ARTIFACT {name}"
        return binaries[0]

    def observe(name, binary, witness):
        if witness.startswith("ce02/"):
            return run(name, [sys.executable, str(HERE / "path-resolution-ce02-probe.py"), "--binary", binary,
                              "--sandbox", str(logs / (name + "-fixture")), "--focus", witness.split("/")[1]])
        if witness == "native":
            return run(name, [sys.executable, str(HERE / "path-resolution-probe.py"), "--binary", binary,
                              "--sandbox", str(logs / (name + "-fixture"))])
        return run(name, [binary, witness, "--exact", "--nocapture"])

    results = []
    try:
        for name, file, old, new, witness, reason, native in selected:
            if native and os.name != "nt":
                results.append({"control": name, "status": "NOT_APPLICABLE", "reason": "Windows native entry/cwd/precondition branch"})
                continue
            binary = build(name + "-baseline")
            code, output = observe(name + "-baseline", binary, witness)
            assert code == 0 and (witness == "native" or witness.startswith("ce02/") or "1 passed; 0 failed" in output) and "CONTROL FIRED " not in output, f"INVALID BASELINE {name}"
            source = safety.checked(tree / file, ROOT)
            safety.checked(source, ROOT).write_text(originals[file].replace(old, new), encoding="utf-8")
            binary = build(name)
            code, output = observe(name, binary, witness)
            if witness.startswith("ce02/"):
                record = json.loads(output)
                killed = code == 1 and record.get("status") == "FAIL" and sorted(row["id"] for row in record["failures"]) == sorted(reason) and f"CONTROL FIRED {name}" in record.get("fired", [])
            elif witness == "native":
                record = json.loads(output)
                killed = code == 1 and record.get("status") == "FAIL" and record.get("id") == reason and f"CONTROL FIRED {name}" in record.get("fired", [])
            else:
                killed = code == 101 and "1 failed" in output and witness + " ... FAILED" in output and reason in output and f"CONTROL FIRED {name}" in output
            assert killed, f"SURVIVED OR INVALID {name}: exit {code}; intended witness={reason}"
            results.append({"control": name, "status": "KILLED", "witness": witness, "reason": reason, "fired": True})
            safety.checked(source, ROOT).write_bytes(original_bytes[file])
            print(json.dumps(results[-1]), flush=True)
    finally:
        for file, content in original_bytes.items():
            safety.checked(tree / file, ROOT).write_bytes(content)
    assert all((tree / file).read_bytes() == content for file, content in original_bytes.items()), "restoration must be byte-identical"
    binary = build("restored")
    code, output = run("restored", [binary, "--nocapture"])
    assert code == 0 and "0 failed" in output and "CONTROL FIRED " not in output, "INVALID RESTORED LIB BASELINE"
    code, output = observe("restored-native", binary, "native")
    assert code == 0 and "CONTROL FIRED " not in output, "INVALID RESTORED NATIVE BASELINE"
    if os.name == "nt":
        code, output = run("restored-ce02", [sys.executable, str(HERE / "path-resolution-ce02-probe.py"), "--binary", binary,
            "--sandbox", str(logs / "restored-ce02-fixture")])
        assert code == 0 and json.loads(output)["status"] == "MATCH" and "CONTROL FIRED " not in output, "INVALID RESTORED CE02 BASELINE"
    with safety.checked(logs / "summary.json", ROOT).open("w", encoding="utf-8") as stream:
        json.dump({"controls": results, "restored": "GREEN"}, stream, indent=2)
    print(json.dumps({"killed": sum(row["status"] == "KILLED" for row in results), "not_applicable": sum(row["status"] == "NOT_APPLICABLE" for row in results), "restored": "GREEN"}), flush=True)


if __name__ == "__main__":
    main()
