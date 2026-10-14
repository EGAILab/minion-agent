"""EXEC-013 controls: observed Rust panic type, source origin and exact witness.

Run in a disposable control-code copy, never the candidate checkout. Source every
platform's project environment first; the controls target is never a gate target.
No target or fixture is cleaned. JSON records come from PanicHookInfo, not scraped
exception-looking text. A red aggregate corpus is not a control baseline.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

spec = importlib.util.spec_from_file_location('cache_safety', Path(__file__).with_name('cargo-cache-safety.py'))
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)

SOURCE = 'crates/minion-agent/src/execution/spawn_nul.rs'
PROVIDER = 'crates/minion-agent/src/execution/subprocess.rs'
WITNESSES = 'crates/minion-agent/src/execution/spawn_nul_tests.rs'
BASH = 'crates/minion-agent/src/tools/builtin/bash.rs'
BASH_TESTS = 'crates/minion-agent/src/tools/builtin/bash_tests.rs'
TOOL_TESTS = 'crates/minion-agent/tests/spawn_nul_conformance.rs'
TOOL_UNIT_WITNESS = 'spawn_nul_tools_preserve_lone_surrogate_and_cut_units'
PREFIX = 'execution::subprocess::spawn_nul_tests::'

# id, source, unique old/new anchors, witness, intended assertion, platform
CONTROLS = [
    ('skip-argv', SOURCE, 'argv.iter().enumerate()', 'argv.iter().take(0).enumerate()', 'witness_file', 'nul-control-exact-diagnostic', 'all'),
    ('wrong-argument-index', SOURCE, 'format!("args[{}]", i - 1)', 'format!("args[{}]", i)', 'witness_argument_index', 'nul-control-exact-diagnostic', 'all'),
    ('skip-effective-cwd', SOURCE, 'check(cwd, &"options.cwd".into(), true)?;', '// skipped cwd', 'witness_effective_cwd', 'nul-control-exact-diagnostic', 'all'),
    ('env-value-before-name', SOURCE, 'check(name, &key, false)?;\n        check(value, &key, false)?;', 'check(value, &key, false)?;\n        check(name, &key, false)?;', 'witness_env_name_before_value', 'nul-control-exact-diagnostic', 'all'),
    ('drop-windows-collision', SOURCE, 'return values.into_iter().collect();', 'let mut seen = std::collections::BTreeSet::new();\n        return values.into_iter().filter(|(n, _)| seen.insert(String::from_utf16_lossy(n.code_units()).to_uppercase())).collect();', 'witness_raw_collision', 'nul-control-exact-diagnostic', 'windows'),
    ('posix-sort-overlay', SOURCE, 'let mut indexed: Vec<_> = values', 'return values.into_iter().collect();\n    let mut indexed: Vec<_> = values', 'witness_overlay_owner', 'nul-control-exact-diagnostic', 'posix'),
    ('indices-not-indices', SOURCE, 'let mut number = 0u64;', 'return None;\n    let mut number = 0u64;', 'witness_integer_indices', 'nul-control-exact-diagnostic', 'posix'),
    ('accept-leading-zero-index', SOURCE, '(units.len() > 1 && units[0] == 48)', 'false', 'witness_non_index_names', 'nul-control-exact-diagnostic', 'posix'),
    ('control-escape-lower-hex', SOURCE, 'format!("\\\\x{unit:02X}")', 'format!("\\\\x{unit:02x}")', 'witness_control_escape', 'nul-control-exact-diagnostic', 'all'),
    ('replace-lone-surrogate', SOURCE, 'Some(format!("\\\\u{unit:04x}"))', 'Some("\ufffd".to_owned())', 'witness_lone_surrogate', 'nul-control-exact-diagnostic', 'all'),
    ('truncate-too-early', SOURCE, 'received.truncate(128);', 'received.truncate(127);', 'witness_split_surrogate_diagnostic', 'nul-control-exact-diagnostic', 'all'),
    ('disable-newline-split', SOURCE, 'if kept.len() > 76 {', 'if kept.len() > usize::MAX {', 'witness_newline_split', 'nul-control-exact-diagnostic', 'all'),
    ('ignore-preabort', PROVIDER, 'if options.signal.as_ref().is_some_and(|s| s.aborted()) {', 'if false {', 'witness_preabort', 'nul-control-error-code', 'all'),
    ('wrap-nul-tool-error', BASH, 'ToolCapabilityError::new(message.clone())', 'ToolCapabilityError::new("incorrect wrapper")', 'tools::builtin::bash::tests::spawn_nul_bash_diagnostic_passes_without_wrapper', 'nul-tool-passthrough-exact', 'all'),
    ('message-text-is-mark', BASH, 'if let Some(message) = error.nul_diagnostic() {\n                ToolCapabilityError::new(message.clone())', 'if error.message.contains("null bytes") {\n                ToolCapabilityError::new(error.message.clone())', 'tools::builtin::bash::tests::spawn_nul_bash_ordinary_error_text_is_not_a_mark', 'ordinary-error-keeps-wrapper', 'all'),
    ('find-projects-before-validation', 'crates/minion-agent/src/tools/builtin/find.rs', 'crate::llm::ResultString::from_code_units(s.code_units().to_vec())', 'crate::llm::ResultString::from(s.to_utf8_lossy())', TOOL_UNIT_WITNESS, 'nul-tool-lossless-unit-message', 'all'),
    ('grep-projects-before-validation', 'crates/minion-agent/src/tools/builtin/grep.rs', 'crate::llm::ResultString::from_code_units(s.code_units().to_vec())', 'crate::llm::ResultString::from(s.to_utf8_lossy())', TOOL_UNIT_WITNESS, 'nul-tool-lossless-unit-message', 'all'),
    ('windows-native-byte-order', SOURCE, 'return values.into_iter().collect();', 'let mut pairs: Vec<_> = values.into_iter().collect();\n        pairs.sort_by_key(|(n, _)| String::from_utf16_lossy(n.code_units()));\n        return pairs;', 'witness_utf16_order', 'nul-control-exact-diagnostic', 'windows'),
]

def test_name(witness):
    if witness == TOOL_UNIT_WITNESS:
        return witness
    return witness if '::' in witness else PREFIX + witness

def origin_span(text, marker):
    index = text.index('"' + marker + '"')
    begin = max(text.rfind('assert_eq!(', 0, index), text.rfind('assert!(', 0, index))
    if begin < 0:
        raise RuntimeError('missing intended assertion')
    end = text.index(');', index)
    return text[:begin].count('\n') + 1, text[:end].count('\n') + 1

def credit(code, output, test, token, filename, span):
    if code != 101 or re.search(r'error\[E\d+\]:|error: could not compile|could not compile|running 0 tests', output):
        return 'INVALID'
    if f'test {test} ... FAILED' not in output or '1 failed' not in output:
        return 'INVALID'
    records = []
    for line in output.splitlines():
        if line.startswith('MINION_NUL_PANIC '):
            try:
                records.append(json.loads(line.removeprefix('MINION_NUL_PANIC ')))
            except (ValueError, TypeError):
                return 'INVALID'
    if len(records) != 1:
        return 'INVALID'
    record = records[0]
    if not isinstance(record, dict):
        return 'INVALID'
    if set(record) != {'token', 'thread', 'payload_type', 'message', 'file', 'line', 'column'}:
        return 'INVALID'
    if not isinstance(record['message'], str) or type(record['column']) is not int or record['column'] < 1:
        return 'INVALID'
    location = record.get('file')
    line = record.get('line')
    if (record.get('token') != token or record.get('thread') != test
            or record.get('payload_type') not in ('String', 'str')
            or not isinstance(location, str) or location.replace('\\', '/') != filename
            or type(line) is not int or not span[0] <= line <= span[1]):
        return 'INVALID'
    return 'KILLED'

def self_test():
    test, token, filename, span = 'test-a', 'run-token', 'site.rs', (10, 12)
    base = {'token': token, 'thread': test, 'payload_type': 'String', 'message': 'assertion failed', 'file': filename, 'line': 11, 'column': 1}
    def output(record):
        return f'test {test} ... FAILED\n1 failed\nMINION_NUL_PANIC ' + json.dumps(record)
    assert credit(101, output(base), test, token, filename, span) == 'KILLED'
    for field, value in [('token', 'old'), ('thread', 'other-test'), ('payload_type', 'other'),
                         ('file', 'setup.rs'), ('line', 9), ('line', True), ('line', None),
                         ('message', None), ('column', True), ('column', 0)]:
        assert credit(101, output(dict(base, **{field: value})), test, token, filename, span) == 'INVALID'
    for code in [0, 1, 2, 64]:
        assert credit(code, output(base), test, token, filename, span) == 'INVALID'
    assert credit(101, f'test {test} ... FAILED\n1 failed\nNameError: intended assertion', test, token, filename, span) == 'INVALID'
    assert credit(101, output(base) + '\nerror[E0425]: compile error', test, token, filename, span) == 'INVALID'
    assert credit(101, output(base) + '\nMINION_NUL_PANIC {}', test, token, filename, span) == 'INVALID'
    for key in base:
        missing = dict(base)
        del missing[key]
        assert credit(101, output(missing), test, token, filename, span) == 'INVALID'
    assert credit(101, output(dict(base, unknown=True)), test, token, filename, span) == 'INVALID'
    print('DRIVER SELF-TEST 26/26 PASS')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path)
    parser.add_argument('--logs', type=Path)
    parser.add_argument('--target', type=Path, help='dedicated contained controls target; default per-platform controls target')
    parser.add_argument('--check-anchors', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--control', action='append', help='explicit development subset; never reported as the full batch')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.tree is None:
        parser.error('--tree is required')
    boundary = safety.PROJECT if os.name == 'nt' else Path('/tmp')
    tree = safety.checked(args.tree, boundary)
    original = {path: (tree / path).read_bytes() for path in {c[1] for c in CONTROLS}}
    for name, path, old, new, witness, assertion, platform in CONTROLS:
        if original[path].decode().replace('\r\n', '\n').count(old) != 1:
            raise RuntimeError(f'INVALID anchor {name}')
    if args.check_anchors:
        print(f'ANCHORS {len(CONTROLS)}/{len(CONTROLS)} PASS')
        return
    if args.logs is None or 'control-code' not in tree.parts:
        parser.error('new contained --logs and disposable control-code tree required')
    logs = safety.checked(args.logs, boundary)
    safety.checked(logs, boundary).mkdir()
    if args.target is None:
        default_target = safety.PROJECT / '.tmp/cache/cargo-target/windows-controls' if os.name == 'nt' else Path('/control-target')
        os.environ.setdefault('MINION_CONTROL_CARGO_TARGET', str(default_target))
        target = Path(safety.control_target())
    elif os.name == 'nt':
        target = safety.checked(args.target, safety.PROJECT)
        gate_targets = [safety.PROJECT / '.tmp/cache/cargo-target/windows', safety.PROJECT / '.tmp/cache/cargo-target/linux',
                        safety.PROJECT / '.tmp/claude-rust-target', safety.PROJECT / '.tmp/claude-linux-target']
        if any(target == gate or target.is_relative_to(gate) or gate.is_relative_to(target) for gate in gate_targets):
            raise RuntimeError('dedicated controls target cannot overlap a gate target')
        if not target.exists():
            safety.checked(target, safety.PROJECT).mkdir()
    else:
        if args.target != Path('/control-target'):
            raise RuntimeError('contained dedicated /control-target bind required')
        target = Path(safety.control_target())
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    if env.get('CARGO_BUILD_JOBS') != '4':
        raise RuntimeError('four Cargo jobs required')
    platform = 'windows' if os.name == 'nt' else 'posix'
    selected = [c for c in CONTROLS if c[6] in ('all', platform)]
    total_applicable = len(selected)
    if args.control:
        if set(args.control) - {c[0] for c in selected}:
            parser.error('unknown or inapplicable control')
        selected = [c for c in selected if c[0] in args.control]
    results = []
    def run(name, witness):
        token = uuid.uuid4().hex
        safety.refresh(tree)
        target = ['--features', 'conformance', '--test', 'spawn_nul_conformance'] if witness == TOOL_UNIT_WITNESS else ['--lib']
        command = ['cargo', 'test', '-p', 'minion-agent', *target, test_name(witness), '--', '--exact', '--nocapture', '--test-threads=1']
        result = subprocess.run(command, cwd=tree, env=dict(env, MINION_NUL_EVIDENCE_TOKEN=token), capture_output=True, text=True, encoding='utf-8', errors='replace')
        output = result.stdout + result.stderr
        safety.checked(logs / (name + '.log'), boundary).write_text(output, encoding='utf-8')
        return result.returncode, output, token
    # Every selected witness must pass before the first mutation.
    for witness in dict.fromkeys(c[4] for c in selected):
        code, output, _ = run('baseline-' + witness, witness)
        if code or f'test {test_name(witness)} ... ok' not in output or '1 passed' not in output:
            raise RuntimeError(f'INVALID baseline {witness}, exit {code}')
    for name, path, old, new, witness, assertion, _ in selected:
        target = safety.checked(tree / path, boundary)
        try:
            safety.checked(target, boundary).write_text(original[path].decode().replace('\r\n', '\n').replace(old, new), encoding='utf-8')
            code, output, token = run(name, witness)
            witness_file = TOOL_TESTS if witness == TOOL_UNIT_WITNESS else BASH_TESTS if witness.startswith('tools::') else WITNESSES
            witness_text = (tree / witness_file).read_text().replace('\r\n', '\n')
            status = credit(code, output, test_name(witness), token, witness_file, origin_span(witness_text, assertion))
            results.append({'control': name, 'witness': test_name(witness), 'exit': code, 'status': status})
            safety.checked(logs / 'results.json', boundary).write_text(json.dumps(results, indent=2), encoding='utf-8')
            if status != 'KILLED':
                raise RuntimeError(f'{name} {status}; stop and inspect, no kill credited')
        finally:
            safety.checked(target, boundary).write_bytes(original[path])
    for witness in dict.fromkeys(c[4] for c in selected):
        code, output, _ = run('restored-' + witness, witness)
        if code or f'test {test_name(witness)} ... ok' not in output:
            raise RuntimeError(f'INVALID restored baseline {witness}')
    print(f'CONTROLS {len(results)}/{len(selected)} KILLED, restored witnesses GREEN; selected {len(selected)} of {total_applicable} applicable')

if __name__ == '__main__':
    main()
