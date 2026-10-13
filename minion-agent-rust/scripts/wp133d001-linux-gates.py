"""WP133-D001: read-only source; fresh private fixture/copy tree; isolated mutants.

Launcher proves every host bind source is inside the project before docker run.
Run in a read-only container with /tmp tmpfs, /source ro, /build ro, /node ro,
/search-artifacts ro, /cargo-home, /target, /control-target and /logs E: binds.
"""
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT = Path('/tmp/wp133d')


def checked(path, boundary=ROOT):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path == boundary or not path.is_relative_to(boundary):
        raise RuntimeError('outside gate sandbox: ' + str(path))
    for ancestor in [*reversed(path.parents), path]:
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError('link in gate mutation: ' + str(ancestor))
    if not path.resolve(strict=False).is_relative_to(boundary.resolve(strict=True)):
        raise RuntimeError('outside real gate sandbox')
    return path


def mkdir(path, boundary=ROOT):
    checked(path, boundary)
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        checked(directory, boundary).mkdir()


def copy_tree(source, destination):
    for directory, dirs, files in os.walk(source, followlinks=False):
        relative = Path(directory).relative_to(source)
        for name in list(dirs):
            if name in ('.git', 'target'):
                dirs.remove(name)
            elif not stat.S_ISDIR((Path(directory) / name).lstat().st_mode):
                raise RuntimeError('non-directory copy entry')
        target = destination / relative
        mkdir(target)
        for name in files:
            src = Path(directory) / name
            if not stat.S_ISREG(src.lstat().st_mode):
                raise RuntimeError('non-regular copy source')
            dst = checked(target / name)
            # copyfile, never copy2/cp -p: source mtimes must be fresh.
            shutil.copyfile(src, dst)
            if dst.suffix == '.sh':
                checked(dst).write_bytes(dst.read_bytes().replace(b'\r\n', b'\n'))


def refresh(tree):
    count = 0
    for directory, dirs, files in os.walk(tree, followlinks=False):
        for name in dirs:
            checked(Path(directory) / name)
        for name in files:
            if name.endswith('.rs') or name in ('Cargo.toml', 'Cargo.lock'):
                os.utime(checked(Path(directory) / name), None)
                count += 1
    print('FRESH WORKSPACE SOURCES', count, flush=True)


if os.name != 'posix' or ROOT.exists():
    raise RuntimeError('fresh private Linux container sandbox required')
mkdir(ROOT, Path('/tmp'))
work = ROOT / 'work'
fixtures = ROOT / 'fixtures'
mkdir(work)
mkdir(fixtures)
os.chmod(checked(fixtures), 0o777)
copy_tree(Path('/source/minion-agent-rust'), work / 'minion-agent-rust')
copy_tree(Path('/source/conformance'), work / 'conformance')
shutil.copyfile('/source/pi-parity-manifest.yaml', checked(work / 'pi-parity-manifest.yaml'))
for relative in (
    'minion-agent-python/tests/skills/data/frontmatter-corpus.json',
    'minion-agent-python/tests/skills/data/ignore-corpus.json',
    'minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt',
):
    destination = checked(work / relative)
    mkdir(destination.parent)
    shutil.copyfile(Path('/source') / relative, checked(destination))
bindir = ROOT / 'bin'
mkdir(bindir)
shutil.copyfile('/node', checked(bindir / 'node'))
os.chmod(checked(bindir / 'node'), 0o755)
env = dict(os.environ)
env.update({
    'CARGO_HOME': '/cargo-home', 'CARGO_TARGET_DIR': '/target', 'CARGO_BUILD_JOBS': '4',
    'RUSTUP_HOME': '/cargo-home/rustup', 'RUSTUP_TOOLCHAIN': '1.97.1',
    'RUST_ICU_MAJOR_VERSION_NUMBER': '78', 'RUSTFLAGS': '-L native=/build/icu/lib',
    'RUSTDOCFLAGS': '-D warnings -L native=/build/icu/lib',
    'MINION_AGENT_ICU_IDENTITY': '/build/icu-identity.txt', 'LD_LIBRARY_PATH': '/build/icu/lib',
    'MINION_SEARCH_ENGINE_ARTIFACTS': '/search-artifacts', 'MINION_FIXTURE_ROOT': str(fixtures),
    'TMP': str(fixtures), 'TEMP': str(fixtures), 'TMPDIR': str(fixtures),
    'PYTHONPYCACHEPREFIX': '/cargo-home/pycache', 'PYTHONDONTWRITEBYTECODE': '1',
    'XDG_CACHE_HOME': '/cargo-home/xdg', 'NODE_COMPILE_CACHE': '/cargo-home/node-cache',
    'npm_config_cache': '/cargo-home/npm', 'PIP_CACHE_DIR': '/cargo-home/pip', 'UV_CACHE_DIR': '/cargo-home/uv',
    'PATH': str(bindir) + ':/cargo-home/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:' + os.environ['PATH'],
})
rust = work / 'minion-agent-rust'


def run(name, argv, cwd=rust):
    refresh(rust)
    print('START', name, flush=True)
    with checked(Path('/logs') / (name + '.log'), Path('/logs')).open('w') as log:
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    checked(Path('/logs') / (name + '.exit'), Path('/logs')).write_text(str(result.returncode) + '\n')
    print('END', name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run('versions', ['sh', '-ec', 'rustc --version; cargo --version; node --version; python3 --version'])
run('anchors', ['python3', 'scripts/bash-abort-output-controls.py', '--tree', str(rust), '--check-anchors'])
run('fixture-witnesses', ['cargo', 'test', '--locked', '-p', 'minion-agent', '--all-features', '--lib', 'abort_output_fixture_', '--', '--nocapture'])
run('real-bash-conformance', ['cargo', 'test', '--locked', '-p', 'minion-agent', '--all-features', '--test', 'builtin_bash_conformance', '--', '--nocapture'])
run('fmt', ['cargo', 'fmt', '--all', '--', '--check'])
run('clippy', ['cargo', 'clippy', '--workspace', '--all-targets', '--all-features', '--', '-D', 'warnings'])
run('tests', ['cargo', 'test', '--workspace', '--all-features'])
run('doc', ['cargo', 'doc', '--workspace', '--no-deps'])
run('conformance-verify', ['cargo', 'run', '-p', 'xtask', '--', 'conformance', 'verify'])
run('uid-bash-build', ['cargo', 'test', '-p', 'minion-agent', '--all-features', '--test', 'builtin_bash_conformance', '--no-run', '--message-format=json'])
rows = [json.loads(line) for line in Path('/logs/uid-bash-build.log').read_text().splitlines() if line.startswith('{')]
executables = [r['executable'] for r in rows if r.get('executable') and r.get('target', {}).get('name') == 'builtin_bash_conformance']
if len(executables) != 1:
    raise RuntimeError('one built bash adapter binary required')
run('uid1000-bash-conformance', ['setpriv', '--reuid', '1000', '--regid', '1000', '--clear-groups', executables[0], '--nocapture'])
control = ROOT / 'control-code'
mkdir(control)
copy_tree(rust, control / 'minion-agent-rust')
copy_tree(work / 'conformance', control / 'conformance')
for relative in (
    'minion-agent-python/tests/skills/data/frontmatter-corpus.json',
    'minion-agent-python/tests/skills/data/ignore-corpus.json',
    'minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt',
):
    destination = checked(control / relative)
    mkdir(destination.parent)
    shutil.copyfile(work / relative, checked(destination))
run('controls', ['python3', 'scripts/bash-abort-output-controls.py', '--tree', str(control / 'minion-agent-rust'),
    '--logs', str(ROOT / 'controls'), '--target-dir', '/control-target'])
for path in (ROOT / 'controls').iterdir():
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('unexpected control evidence entry')
    shutil.copyfile(path, checked(Path('/logs') / ('control-' + path.name), Path('/logs')))
checked(Path('/logs/complete.exit'), Path('/logs')).write_text('0\n')
print('WP133D001 LINUX COMPLETE', flush=True)
