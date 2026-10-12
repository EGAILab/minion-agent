"""RR002: real Cargo old-mtime mutant reuse is RED; isolated fresh gate is GREEN.

Run only after the full intended-witness control batch. Never clean either cache.
The deliberate RED build uses ONLY the dedicated control cache. All source bytes
are restored even on failure; G3/control launchers subsequently refresh sources.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location('cargo_cache_safety', Path(__file__).with_name('cargo-cache-safety.py'))
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)
parser = argparse.ArgumentParser()
parser.add_argument('--tree', type=Path, required=True)
parser.add_argument('--control-tree', type=Path, required=True)
parser.add_argument('--logs', type=Path, required=True)
args = parser.parse_args()
boundary = safety.PROJECT if os.name == 'nt' else Path('/tmp')
tree = safety.checked(args.tree, boundary)
control = safety.checked(args.control_tree, boundary)
if 'control-code' not in control.parts:
    raise RuntimeError('disposable control-code tree required')
logs = safety.checked(args.logs, boundary)
safety.checked(logs, boundary).mkdir()
gate_target = Path(os.environ['CARGO_TARGET_DIR'])
expected = safety.PROJECT / '.tmp/cache/cargo-target/windows' if os.name == 'nt' else Path('/target')
if gate_target != expected:
    raise RuntimeError('regression must start with the unmutated gate target')
mutant_target = safety.control_target()
if str(gate_target) == mutant_target:
    raise RuntimeError('gate and mutant caches must differ')
witness = 'tests::directory_reparse_payloads_keep_libuv_readlink_acceptance'
source = safety.checked(control / 'crates/minion-agent-native-fs/src/lib.rs', boundary)
original = source.read_bytes()
text = original.decode('utf-8').replace('\r\n', '\n')
old = '        _ => false,\n    }\n}'
if text.count(old) != 1:
    raise RuntimeError('INVALID mutant anchor')


def run(name, cwd, target, bad=False):
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    with safety.checked(logs / (name + '.log'), boundary).open('w', encoding='utf-8') as log:
        result = subprocess.run(['cargo', 'test', '-p', 'minion-agent-native-fs', '--all-features', '--lib', witness, '--', '--exact', '--nocapture'], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    output = (logs / (name + '.log')).read_text(encoding='utf-8')
    required = [f'test {witness} ... ' + ('FAILED' if bad else 'ok'), '1 failed' if bad else '1 passed']
    if bad:
        required += ['assertion failed: !valid_directory_link', '0x9000001a']
    if result.returncode != (101 if bad else 0) or not all(x in output for x in required) or 'error[E' in output:
        raise RuntimeError(f'INVALID {name}: exit {result.returncode}, inspect {logs}')
    print(name + (' EXPECTED FAIL' if bad else ' PASS'), flush=True)
    return output


try:
    safety.refresh(control)
    run('baseline', control, mutant_target)
    safety.checked(source, boundary).write_text(text.replace(old, '        _ => true,\n    }\n}'), encoding='utf-8', newline='\n')
    run('mutant', control, mutant_target, True)
    safety.checked(source, boundary).write_bytes(original)
    # Simulate an older checkout: same restored bytes, mtime before the build.
    os.utime(safety.checked(source, boundary), ns=(1_000_000_000, 1_000_000_000))
    bad = run('known-bad-shared-target-old-mtime', control, mutant_target, True)
    if 'Compiling minion-agent-native-fs' in bad:
        raise RuntimeError('INVALID: old-mtime witness rebuilt instead of reusing mutant')
    print('KNOWN-BAD mutant reused after byte restoration without source freshness', flush=True)
    # The actual gate checkout is also made old, then refreshed by the SAME
    # helper used before G3. No content changes and no recursion into targets.
    for path in tree.rglob('*.rs'):
        os.utime(safety.checked(path, boundary), ns=(1_000_000_000, 1_000_000_000))
    safety.refresh(tree)
    green = run('candidate-isolated-fresh-gate', tree, gate_target)
    if 'Compiling minion-agent-native-fs' not in green:
        raise RuntimeError('INVALID: fresh gate package did not rebuild')
    print(f'RR002 CACHE REGRESSION PASS: control={mutant_target}, gate={gate_target}; older checkout runs no mutant', flush=True)
finally:
    safety.checked(source, boundary).write_bytes(original)
