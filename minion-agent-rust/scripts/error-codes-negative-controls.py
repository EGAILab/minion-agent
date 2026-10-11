"""L12-D007 intended-witness controls, in a disposable containment-checked tree.

One baseline per selected matrix row must pass before mutation. Rust compilation,
setup failures and wrong witnesses are INVALID, never kills. The original source
and an unmutated baseline are restored even after failure. Windows-only recovery
controls are explicitly N/A on Linux, never credited as kills there.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--tree", type=Path, required=True)
p.add_argument("--logs", type=Path, required=True)
a = p.parse_args()
boundary = Path("E:/AI/Projects/OpenMinds/Minions/Minion-Agent") if os.name == "nt" else Path("/tmp/l12d007")
boundary = boundary.resolve(strict=True)

def checked(path):
    path = Path(path)
    if ".." in path.parts or not path.is_absolute() or not path.is_relative_to(boundary) or path == boundary:
        raise RuntimeError(f"unsafe control target: {path}")
    for parent in [*reversed(path.parents), path]:
        if parent == boundary or parent.is_relative_to(boundary):
            if parent.is_symlink():
                raise RuntimeError(f"link in control target: {parent}")
    real = path.resolve(strict=False)
    if not real.is_relative_to(boundary) or real == boundary:
        raise RuntimeError(f"outside control target: {path}")
    return path

tree = checked(a.tree)
if "control-code" not in tree.parts:
    raise RuntimeError("a disposable control-code copy is required")
logs = checked(a.logs)
logs.mkdir(parents=True, exist_ok=True)
rimraf = checked(tree / "crates/minion-agent/src/execution/filesystem/rimraf.rs")
fs = checked(tree / "crates/minion-agent/src/execution/filesystem.rs")
native = checked(tree / "crates/minion-agent-native-fs/src/lib.rs")
originals = {x: x.read_text(encoding="utf-8") for x in (rimraf, fs, native)}

def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"INVALID anchor ({text.count(old)}): {old!r}")
    return text.replace(old, new)

controls = []
def control(name, row, file, old, new, windows=False):
    if windows and os.name != "nt":
        print(f"NOT_APPLICABLE {name}: Windows fixWinEPERM/native IO only", flush=True)
        return
    controls.append((name, row, file, lambda text: replace(text, old, new)))

control("N1-directory-error-caught-as-classification", "E1-child-rmdir-fails", rimraf,
    "                Err(e) => return Err(native_error(e, &p)),\n            },\n            Work::FinalDirectory",
    "                Err(_) => ops.unlink(&p).await.map_err(|e| native_error(e, &p))?,\n            },\n            Work::FinalDirectory")
control("N2-retry-child-rmdir", "E1-child-rmdir-fails", rimraf,
    "                Err(e) => return Err(native_error(e, &p)),\n            },\n            Work::FinalDirectory",
    "                Err(_) => ops.rmdir(&p).await.map_err(|e| native_error(e, &p))?,\n            },\n            Work::FinalDirectory")
control("N3-swallow-child-unlink-failure", "E2-inner-unlink-fails", rimraf,
    "                    Err(e) => return Err(native_error(e, &p)),\n                }\n            }\n            Work::Directory",
    "                    Err(_) => (),\n                }\n            }\n            Work::Directory")
control("N4-ancestor-error-origin", "E2-inner-unlink-fails", rimraf,
    "                    Err(e) => return Err(native_error(e, &p)),\n                }\n            }\n            Work::Directory",
    "                    Err(e) => return Err(native_error(e, path)),\n                }\n            }\n            Work::Directory")
control("N5-enumerate-before-rmdir", "E5-child-rmdir-busy", rimraf,
    "            Work::Directory(p, original) => match ops.rmdir(&p).await {",
    "            Work::Directory(p, original) => match { let _ = ops.readdir(&p).await; ops.rmdir(&p).await } {")
control("N6-classification-eperm-skips-recovery", "L3-chmod-fails", rimraf,
    "                    Err(e) if cfg!(windows) && eperm(&e) => {\n                        recover(&p, e, ops, &mut stack).await?;\n                        continue;\n                    }",
    "                    Err(e) if cfg!(windows) && eperm(&e) => { let _ = e; }", True)
control("N7-correction-error-replaces-original", "U2-chmod-fails", rimraf,
    "            Err(native_error(original, p))\n        };", "            Err(native_error(e, p))\n        };", True)
control("N8-correction-required-to-retry", "U1-file-recovered", rimraf,
    ".map(|_| ())", ".and_then(|changed| if changed { Ok(()) } else { Err(io::Error::from_raw_os_error(5)) })", True)
control("N9-retry-keeps-original", "U4-retry-fails-eio", rimraf,
    "            Err(e) => Err(native_error(e, p)),", "            Err(_) => Err(native_error(original, p)),", True)
def follow_correction(text):
    head, tail = text.split("pub fn clear_readonly_entry(", 1)
    return head + "pub fn clear_readonly_entry(" + replace(tail,
        ".custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)",
        ".custom_flags(FILE_FLAG_BACKUP_SEMANTICS)")
if os.name == "nt":
    controls.append(("N10-correction-follows-link", "U7-link-to-readonly-target", native, follow_correction))
else:
    print("NOT_APPLICABLE N10: Windows own-entry attribute correction", flush=True)
control("N11-nonrecursive-file-bypasses-shared-routing", "T1-single-file-recovered", rimraf,
    "            Work::Entry(p) => {",
    "            Work::Entry(p) => { if !recursive && p == path { ops.unlink(&p).await.map_err(|e| native_error(e, &p))?; continue; }")
control("N12-validation-routed-to-recovery", "V1-validation-eperm-file", rimraf,
    "        Err(e) => return Err(native_error(e, path)),",
    "        Err(e) if eperm(&e) => { ops.chmod(path).await.map_err(|e| native_error(e, path))?; let _ = ops.stat(path).await; },\n        Err(e) => return Err(native_error(e, path)),", True)

def cached_validation(text):
    clone = "validation.as_ref().map(|m| m.clone()).map_err(|e| e.raw_os_error().map(io::Error::from_raw_os_error).unwrap_or_else(|| io::Error::new(e.kind(), e.to_string())))"
    text = replace(text, "    match ops.lstat(path).await {", f"    let validation = ops.lstat(path).await;\n    match {clone} {{")
    return replace(text, "let classification = ops.lstat(&p).await;", f"let classification = {clone};")
controls.append(("N13-reuse-validation-no-classification", "V2-classification-eperm-file", rimraf, cached_validation))
control("N14-force-validation-returns-early", "V9-validation-enoent-force", rimraf,
    "        Err(e) if force && missing(&e) => (),", "        Err(e) if force && missing(&e) => return Ok(()),")
control("child-vanish-windows-only", "@execution::filesystem::rimraf::tests::a_child_vanishing_between_enumeration_and_lstat_is_removed_on_every_platform", rimraf,
    "                    Err(e) if missing(&e) => continue,\n                    Err(e) if cfg!(windows)",
    "                    Err(e) if missing(&e) => { if cfg!(windows) { continue; } return Err(native_error(e, &p)); },\n                    Err(e) if cfg!(windows)") if os.name != "nt" else None
control("handle-denial-keeps-access-code", "@execution::filesystem::rimraf::tests::actual_provider_handle_failures_are_unknown_before_eof_and_keep_the_logical_path", fs,
    "if error.raw_os_error() == Some(5)", "if false && error.raw_os_error() == Some(5)", True)
control("raw-code-collapses-to-error-kind", "@execution::filesystem::l12d007_handle_tests::literal_win32_mapping_is_not_the_std_error_kind_mapping", fs,
    "    if let Some(raw) = error.raw_os_error() {\n        return FsError::new(win32_pi_code(raw), error.to_string());\n    }",
    "    if let Some(raw) = error.raw_os_error() {\n        let _ = win32_pi_code(raw);\n    }", True)
control("depth-limited-removal-walk", "@execution::filesystem::rimraf::tests::removal_walk_uses_constant_interpreter_stack_for_deep_trees", rimraf,
    "    while let Some(work) = stack.pop() {",
    "    let mut depth_budget = 1024usize;\n    while let Some(work) = stack.pop() {\n        depth_budget -= 1;\n        if depth_budget == 0 { return Err(FsError::new(FsErrorCode::Unknown, \"mutant depth bound\")); }")
def std_classification(text):
    text = replace(text, "    if directory && attributes & FILE_ATTRIBUTE_DIRECTORY == 0 {",
        "    if directory && !metadata.is_dir() {")
    text = replace(text,
        "    if !directory\n        && attributes & FILE_ATTRIBUTE_DIRECTORY != 0\n        && attributes & FILE_ATTRIBUTE_REPARSE_POINT == 0\n    {",
        "    if !directory && metadata.is_dir() && !metadata.file_type().is_symlink() {")
    return replace(text, "    let attributes = metadata.file_attributes();", "    let _attributes = metadata.file_attributes();")
if os.name == "nt":
    for link_row in ("U8-dirlink-unlink-eperm", "L10-dirlink-lstat-eperm", "U9-junction-unlink-eperm", "L11-junction-lstat-eperm"):
        controls.append(("std-classification-" + link_row, link_row, native, std_classification))
else:
    print("NOT_APPLICABLE std-classification: Windows raw-attribute classification", flush=True)
control("readdir-missing-is-error", "@execution::filesystem::rimraf::tests::disappearance_at_readdir_is_accepted", rimraf,
    "                        Err(e) if missing(&e) => continue,",
    "                        Err(e) if missing(&e) => return Err(native_error(e, &p)),")
control("first-rmdir-missing-is-error", "@execution::filesystem::rimraf::tests::disappearance_at_first_rmdir_is_accepted", rimraf,
    "                Err(e) if missing(&e) => (),\n                Err(e) if enotdir(&e) => {",
    "                Err(e) if missing(&e) => return Err(native_error(e, &p)),\n                Err(e) if enotdir(&e) => {")
control("final-rmdir-missing-is-error", "@execution::filesystem::rimraf::tests::disappearance_at_final_rmdir_is_accepted", rimraf,
    "            Work::FinalDirectory(p) => match ops.rmdir(&p).await {\n                Ok(()) => (),\n                Err(e) if missing(&e) => (),",
    "            Work::FinalDirectory(p) => match ops.rmdir(&p).await {\n                Ok(()) => (),\n                Err(e) if missing(&e) => return Err(native_error(e, &p)),")
control("retry-unlink-missing-is-error", "@execution::filesystem::rimraf::tests::disappearance_at_retry_unlink_is_accepted", rimraf,
    "            Err(e) if missing(&e) => Ok(()),\n            Err(e) => Err(native_error(e, p)),",
    "            Err(e) if missing(&e) => Err(native_error(e, p)),\n            Err(e) => Err(native_error(e, p)),", True)
control("reparse-any-tag-is-link", "@native:tests::directory_reparse_payloads_keep_libuv_readlink_acceptance", native,
    "        _ => false,\n    }\n}", "        _ => true,\n    }\n}")
matrix_test = "execution::filesystem::rimraf::tests::l12d007_all_41_pi_routing_rows_match_and_injections_fire"
link_test = "execution::filesystem::rimraf::tests::directory_link_recovery_matches_pinned_pi"

def run(name, row, mutant=False):
    package = "minion-agent-native-fs" if row.startswith("@native:") else "minion-agent"
    witness = row[len("@native:"):] if row.startswith("@native:") else row[1:] if row.startswith("@") else link_test if "dirlink" in row or "junction" in row else matrix_test
    env = dict(os.environ)
    env.pop("L12D007_CASE", None)
    if witness in (matrix_test, link_test): env["L12D007_CASE"] = row
    command = ["cargo", "test", "-p", package, "--all-features", "--lib", witness, "--", "--exact", "--nocapture"]
    result = subprocess.run(command, cwd=tree, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    checked(logs / f"{name}.log").write_text(result.stdout, encoding="utf-8")
    valid = result.returncode == (101 if mutant else 0)
    valid &= f"test {witness} ... {'FAILED' if mutant else 'ok'}" in result.stdout
    valid &= ("1 failed" if mutant else "1 passed") in result.stdout
    if witness in (matrix_test, link_test): valid &= f'"id":"{row}"' in result.stdout
    if mutant:
        if witness == link_test:
            valid &= f"Pi directory-link recovery: \"{row}\"" in result.stdout
            valid &= '"unfired":[]' in result.stdout
        elif witness == matrix_test:
            allowed_unfired = name.startswith(("N6-", "N8-", "N11-", "N13-"))
            valid &= (f"Pi routing: \"{row}\"" in result.stdout or
                (allowed_unfired and f"binding-only UNFIRED: \"{row}\"" in result.stdout))
        else:
            valid &= "panicked at" in result.stdout
            signatures = {
                "handle-denial-keeps-access-code": ("handle failure is not EOF or an open error", "PermissionDenied", "Unknown"),
                "raw-code-collapses-to-error-kind": ("left: Unknown", "right: IsDirectory"),
                "depth-limited-removal-walk": ("mutant depth bound", "code: Unknown"),
                "child-vanish-windows-only": ("code: NotFound", "tree/child"),
                "readdir-missing-is-error": ("concurrent disappearance must succeed", "not_found"),
                "first-rmdir-missing-is-error": ("concurrent disappearance must succeed", "not_found"),
                "final-rmdir-missing-is-error": ("concurrent disappearance must succeed", "not_found"),
                "retry-unlink-missing-is-error": ("concurrent disappearance must succeed", "not_found"),
                "reparse-any-tag-is-link": ("assertion failed: !valid_directory_link", "0x9000001a"),
            }
            valid &= all(part in result.stdout for part in signatures[name])
        valid &= "error[E" not in result.stdout and "thread '" in result.stdout
    if not valid:
        raise RuntimeError(f"INVALID {name}: exit {result.returncode}; {logs / (name + '.log')}")

try:
    # Force a source rebuild of the baseline, rather than accepting an artifact
    # whose old include_str! dependency happened to survive in a shared target.
    for file, original in originals.items():
        checked(file).write_text(original, encoding="utf-8")
    # Check *every* intended row, not just a green generic test invocation.
    for name, row, _, _ in controls:
        run(name + "-baseline", row)
    print(f"BASELINE: {len(controls)} intended witnesses selected and passed", flush=True)
    for name, row, file, mutation in controls:
        mutated = mutation(originals[file])
        if mutated == originals[file]:
            raise RuntimeError(f"INVALID no-op {name}")
        checked(file).write_text(mutated, encoding="utf-8")
        try:
            run(name, row, True)
            print(f"KILLED {name} by {row}", flush=True)
        finally:
            checked(file).write_text(originals[file], encoding="utf-8")
finally:
    for file, original in originals.items():
        checked(file).write_text(original, encoding="utf-8")
    run("restored-baseline", "C0-no-error")
