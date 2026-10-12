"""WP133-D001 controls: real intended witnesses, isolated per-platform target.

All baselines green before mutation; source restored in finally. Setup/compiler
errors, absent witnesses and wrong assertions are INVALID, never a kill.
"""
import argparse
import os
from pathlib import Path
import stat
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--tree', type=Path, required=True)
parser.add_argument('--logs', type=Path, required=True)
parser.add_argument('--target-dir', type=Path, required=True)
args = parser.parse_args()
boundary = Path('E:/AI/Projects/OpenMinds/Minions/Minion-Agent') if os.name == 'nt' else Path('/tmp/wp133d')


def checked(path, root=boundary):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path == root or not path.is_relative_to(root):
        raise RuntimeError(f'outside control sandbox: {path}')
    for entry in [*reversed(path.parents), path]:
        try:
            metadata = entry.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & 1024:
            raise RuntimeError(f'link/reparse in control target: {entry}')
    if not path.resolve(strict=False).is_relative_to(root.resolve(strict=True)):
        raise RuntimeError('outside real control target')
    return path


tree = checked(args.tree)
if 'control-code' not in tree.parts:
    raise RuntimeError('disposable control-code tree required')
target = args.target_dir
if os.name == 'nt':
    target = checked(target)
    if '.tmp' not in target.parts or not target.name.endswith('-controls'):
        raise RuntimeError('dedicated project-local controls target required')
else:
    if target != Path('/control-target') or not stat.S_ISDIR(target.lstat().st_mode) or target.is_symlink():
        raise RuntimeError('contained E: controls cache bind at /control-target required')
if target.resolve() == Path(os.environ['CARGO_TARGET_DIR']).resolve():
    raise RuntimeError('control and gate targets must differ')
env = dict(os.environ, CARGO_TARGET_DIR=str(target))
logs = checked(args.logs)
checked(logs).mkdir()
source = checked(tree / 'crates/minion-agent/tests/support/bash_abort_on_output.rs')
original = source.read_bytes()
text = original.decode().replace('\r\n', '\n')
prefix = 'tools::builtin::bash::tests::'
split = prefix + 'abort_output_fixture_split_marker_is_delivered_before_one_abort_and_close_delegates'


def replace(old, new):
    if text.count(old) != 1:
        raise RuntimeError('INVALID anchor: ' + old)
    return text.replace(old, new)


start = text.index('            tokio::spawn(async move {')
end = text.index('            });', start) + len('            });')
immediate = text[:start] + '            cancel();' + text[end:]
controls = [
    ('all-spawns', prefix + 'abort_output_fixture_real_lookup_never_aborts_the_command',
     replace('let command = options.signal.is_some();', 'let command = true;'), 'lookup must not abort the call'),
    ('no-cross-chunk-marker', split,
     replace('bytes.extend_from_slice(chunk);', 'bytes.clear();\n        bytes.extend_from_slice(chunk);'), 'split output marker must abort'),
    ('abort-before-delivery', split, immediate, 'completing output must be delivered before abort'),
    ('stderr-not-watched', prefix + 'abort_output_fixture_watches_stderr_and_combines_delivery_order',
     replace('stderr: wrap(inner.stderr()),', 'stderr: inner.stderr(),'), 'stderr marker must abort'),
    ('premature-utf8-replacement', prefix + 'abort_output_fixture_utf8_suffix_does_not_trigger_a_premature_replacement',
     replace('String::from_utf8_lossy(&bytes[..end])', '{ let _ = end; String::from_utf8_lossy(&bytes) }'), 'incomplete UTF8 is not replacement output yet'),
]
for name, _, mutated, _ in controls:
    if mutated == text:
        raise RuntimeError('INVALID no-op: ' + name)

# Fresh workspace-member sources before ANY baseline, not just the mutated file.
for directory, dirs, files in os.walk(tree, followlinks=False):
    for name in list(dirs):
        if name in ('.git', 'target'): dirs.remove(name)
        else: checked(Path(directory) / name)
    for name in files:
        if name.endswith('.rs') or name in ('Cargo.toml', 'Cargo.lock'):
            os.utime(checked(Path(directory) / name), None)


def run(name, witness, bad=False, reason=''):
    with checked(logs / (name + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(['cargo', 'test', '-p', 'minion-agent', '--all-features', '--lib', witness, '--', '--exact', '--nocapture'], cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT)
    output = (logs / (name + '.log')).read_text(encoding='utf-8')
    required = [f'test {witness} ... ' + ('FAILED' if bad else 'ok'), '1 failed' if bad else '1 passed']
    if bad: required += ['panicked at', reason]
    if result.returncode != (101 if bad else 0) or not all(x in output for x in required) or 'error[E' in output:
        raise RuntimeError(f'INVALID {name}: exit {result.returncode}')


try:
    for name, witness, _, _ in controls:
        run(name + '-baseline', witness)
    print('BASELINE 5/5 intended witnesses selected and passed', flush=True)
    for name, witness, mutated, reason in controls:
        checked(source).write_text(mutated, encoding='utf-8', newline='\n')
        run(name, witness, True, reason)
        print('KILLED ' + name + ': ' + reason, flush=True)
        checked(source).write_bytes(original)
finally:
    checked(source).write_bytes(original)
run('restored-baseline', controls[0][1])
print('WP133D001 CONTROLS PASS 5/5; restored baseline green', flush=True)
