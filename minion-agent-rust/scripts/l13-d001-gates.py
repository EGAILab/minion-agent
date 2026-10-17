"""L13-D001 targeted Rust witness/canonical gates and compiled wrong-fallback controls.

Uses the existing contained source-copy/freshness helpers, disjoint platform
control targets, fresh exclusive logs, no cleanup, timeout, kill or retry.
Linux callers supply the existing E:-backed /target and /control-target binds,
private /tmp, ICU and artifact environment. This is not a full workspace G3.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
TREE = Path(__file__).resolve().parents[1]
PROJECT = Path('E:/AI/Projects/OpenMinds/Minions/Minion-Agent')
WINDOWS = os.name == 'nt'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load('l13_guard', TREE / 'scripts/path-resolution-gates.py')
safety = load('l13_safety', TREE / 'scripts/cargo-cache-safety.py')
parser = argparse.ArgumentParser()
parser.add_argument('--logs', type=Path, required=True)
args = parser.parse_args()
boundary = PROJECT if WINDOWS else Path('/logs')
logs = guard.checked(args.logs, boundary)
assert not logs.exists(), 'fresh log directory required'
guard.mkdir(logs, boundary)
assert os.environ['CARGO_BUILD_JOBS'] == '4'
if WINDOWS:
    os.environ.update({
        'RUST_ICU_MAJOR_VERSION_NUMBER': '78',
        'RUSTFLAGS': '-L native=' + str(PROJECT / '.toolchain/icu-78.3-src/icu/lib64').replace('\\', '/'),
        'MINION_AGENT_ICU_IDENTITY': str(PROJECT / '.toolchain/icu-78.3-src/pinned-icu-identity.txt'),
        'MINION_SEARCH_ENGINE_ARTIFACTS': str(PROJECT / '.toolchain/search-engines/dl'),
        'PATH': str(PROJECT / '.toolchain/icu-78.3-src/icu/bin64') + os.pathsep + os.environ['PATH'],
    })
results = {}


def run(label, command, tree=TREE, env=None):
    safety.refresh(tree)
    file = guard.checked(logs / (label + '.log'), boundary)
    print('START', label, flush=True)
    with file.open('x', encoding='utf-8') as output:
        rc = subprocess.run(command, cwd=tree, env=env, stdout=output, stderr=subprocess.STDOUT).returncode
    with guard.checked(logs / (label + '.exit'), boundary).open('x', encoding='ascii') as output:
        output.write(str(rc) + '\n')
    results[label] = rc
    print('END', label, rc, flush=True)
    return rc, file.read_text(encoding='utf-8', errors='replace')


for label, command in [
    ('fmt', ['cargo', 'fmt', '--all', '--', '--check']),
    ('witness-and-canonical', ['cargo', 'test', '--locked', '--offline', '-p', 'minion-agent',
                              '--features', 'conformance', '--test', 'builtin_tool_conformance', '--', '--nocapture']),
    ('clippy-targeted', ['cargo', 'clippy', '--locked', '--offline', '-p', 'minion-agent', '--all-features',
                         '--test', 'builtin_tool_conformance', '--', '-D', 'warnings']),
]:
    rc, _ = run(label, command)
    if rc:
        raise SystemExit(rc)

copy = logs / 'control-code' if WINDOWS else Path('/tmp/l13-d001-control-code')
guard.checked(copy)
assert not copy.exists(), 'fresh control copy required'
guard.workspace_copy(TREE.parent, copy)
control_tree = copy / 'minion-agent-rust'
control_env = dict(os.environ, CARGO_TARGET_DIR=safety.control_target())
command = ['cargo', 'test', '--locked', '--offline', '-p', 'minion-agent', '--features', 'conformance',
           '--test', 'builtin_tool_conformance', 'ls_s13_', '--', '--nocapture']


def green(label):
    rc, text = run(label, command, control_tree, control_env)
    assert rc == 0 and '3 passed; 0 failed' in text, label + ': named baseline/restored not green'


green('controls-baseline')
source = guard.checked(control_tree / 'crates/minion-agent/src/tools/builtin/ls.rs')
original = source.read_bytes()
anchor = b'.unwrap_or_else(|_| working.clone())'
assert original.count(anchor) == 1, 'unique production fallback anchor'
mutants = [
    ('K1-raw', b'.unwrap_or_else(|_| path.clone())', 'Path not found: @no'),
    ('K2-fabricated', b'.unwrap_or_else(|_| FsPath::from(format!("/{}", working.as_str().unwrap())))',
     'Path not found: /no such'),
]
for name, replacement, observation in mutants:
    with guard.checked(source).open('wb') as output:
        output.write(original.replace(anchor, replacement))
    rc, text = run(name, command, control_tree, control_env)
    intended = (rc == 101 and 'L13-S13: fallback message' in text and observation in text
                and '1 passed; 2 failed' in text
                and 'error[E' not in text and 'could not compile' not in text)
    results[name + '-intended-kill'] = intended
    # Restore even an invalid observation; never certify from an unrelated failure.
    with guard.checked(source).open('wb') as output:
        output.write(original)
    green(name + '-restored')
    if not intended:
        raise RuntimeError(name + ': NOT_KILLED / invalid observation; inspect retained logs')
with guard.checked(logs / 'summary.json', boundary).open('x', encoding='utf-8') as output:
    json.dump(results, output, indent=2)
print('L13-D001 targeted gates GREEN; K1/K2 intended kills; restored GREEN', flush=True)
