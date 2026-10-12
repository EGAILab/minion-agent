#!/bin/sh
# Source/data are read-only E: binds; target/home/logs are project E: binds.
# All fixtures and disposable copies are in the container's private tmpfs.
set -eu
export TMP=/tmp TEMP=/tmp TMPDIR=/tmp
export CARGO_HOME=/cargo-home CARGO_TARGET_DIR=/target CARGO_BUILD_JOBS=4
export MINION_CONTROL_CARGO_TARGET=/control-target
export RUSTUP_HOME=/cargo-home/rustup RUSTUP_TOOLCHAIN=1.97.1
export RUST_ICU_MAJOR_VERSION_NUMBER=78
export RUSTFLAGS='-L native=/build/icu/lib'
export RUSTDOCFLAGS='-D warnings -L native=/build/icu/lib'
export MINION_AGENT_ICU_IDENTITY=/build/icu-identity.txt
export LD_LIBRARY_PATH=/build/icu/lib MINION_SEARCH_ENGINE_ARTIFACTS=/search-artifacts
export PYTHONPYCACHEPREFIX=/cargo-home/pycache XDG_CACHE_HOME=/cargo-home/xdg
export NODE_COMPILE_CACHE=/cargo-home/node-cache npm_config_cache=/cargo-home/npm
export PIP_CACHE_DIR=/cargo-home/pip UV_CACHE_DIR=/cargo-home/uv
mkdir -p /tmp/l12d008/work /tmp/l12d008/fixtures /tmp/l12d008/uid-logs /tmp/bin
chmod 777 /tmp/l12d008/fixtures /tmp/l12d008/uid-logs
export MINION_FIXTURE_ROOT=/tmp/l12d008/fixtures
export TMP="$MINION_FIXTURE_ROOT" TEMP="$MINION_FIXTURE_ROOT" TMPDIR="$MINION_FIXTURE_ROOT"
cp -r /source/. /tmp/l12d008/work/
cp /node /tmp/bin/node
chmod +x /tmp/bin/node
export PATH=/tmp/bin:/cargo-home/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:$PATH
find /tmp/l12d008/work -name '*.sh' -exec sed -i 's/\r$//' {} +
cd /tmp/l12d008/work/minion-agent-rust
rustc --version
node --version
python3 scripts/cargo-cache-safety.py --tree "$PWD" > /logs/source-freshness.log 2>&1
python3 scripts/cargo-cache-safety-check.py > /logs/cache-safety-check.log 2>&1
# Refactors must not strand previously certified exact-source controls.
mkdir -p /tmp/l12d007
python3 scripts/error-codes-negative-controls.py --tree "$PWD" --check-anchors > /logs/error-codes-anchor-check.log 2>&1
cargo check -p minion-agent-native-fs > /logs/native-fs-check.log 2>&1
cargo fmt --all -- --check > /logs/fmt.log 2>&1
cargo clippy --workspace --all-targets --all-features -- -D warnings > /logs/clippy.log 2>&1
cargo test --workspace --all-features > /logs/test.log 2>&1
cargo doc --workspace --no-deps > /logs/doc.log 2>&1
cargo run -p xtask -- conformance verify > /logs/xtask.log 2>&1
cargo test -p minion-agent --lib execution::filesystem::rimraf::tests -- --nocapture > /logs/rimraf.log 2>&1
# Build the exact test binary as root; run permission evidence without Cargo as uid 1000.
cargo test -p minion-agent --lib --no-run --message-format=json > /logs/lib-build.json 2>/logs/lib-build.log
python3 - <<'PY'
import json, subprocess
rows = [row for row in map(json.loads, open('/logs/lib-build.json')) if row.get('executable') and row.get('target', {}).get('name') == 'minion_agent']
assert len(rows) == 1, rows
subprocess.run(['setpriv', '--reuid', '1000', '--regid', '1000', '--clear-groups', 'python3',
    'scripts/div008-probe.py', '--tree', '/tmp/l12d008/work/minion-agent-rust', '--data', '/data',
    '--logs', '/tmp/l12d008/uid-logs', '--natural', '--binary', rows[0]['executable']], check=True)
PY
cp -r /tmp/l12d008/uid-logs /logs/probe-uid1000
# Existing permission-sensitive canonical corpora also run unprivileged.
cargo test -p minion-agent --test fs_path_domain_conformance --test fs_remove_readonly_conformance --test fs_path_domain --no-run --message-format=json > /logs/corpus-build.json 2>/logs/corpus-build.log
python3 - <<'PY'
import json, subprocess
for row in map(json.loads, open('/logs/corpus-build.json')):
    if row.get('executable') and row.get('target', {}).get('name') in ('fs_path_domain_conformance', 'fs_remove_readonly_conformance', 'fs_path_domain'):
        name = row['target']['name']
        with open('/logs/' + name + '-uid1000.log', 'w') as log:
            extra = ['--exact', 'recursive_remove_single_failure_reports_the_inner_native_call', '--ignored'] if name == 'fs_path_domain' else []
            subprocess.run(['setpriv', '--reuid', '1000', '--regid', '1000', '--clear-groups', row['executable'], *extra, '--nocapture'], stdout=log, stderr=subprocess.STDOUT, check=True)
PY
mkdir /tmp/l12d008/control-code
cp -r /tmp/l12d008/work/minion-agent-rust /tmp/l12d008/control-code/
cp -r /tmp/l12d008/work/conformance /tmp/l12d008/control-code/
mkdir -p /tmp/l12d008/control-code/minion-agent-python/tests/skills/data /tmp/l12d008/control-code/minion-agent-python/tests/execution/data/r002_ada_oracle
cp /tmp/l12d008/work/minion-agent-python/tests/skills/data/frontmatter-corpus.json /tmp/l12d008/work/minion-agent-python/tests/skills/data/ignore-corpus.json /tmp/l12d008/control-code/minion-agent-python/tests/skills/data/
cp /tmp/l12d008/work/minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt /tmp/l12d008/control-code/minion-agent-python/tests/execution/data/r002_ada_oracle/
set +e
python3 scripts/div008-negative-controls.py --tree /tmp/l12d008/control-code/minion-agent-rust --logs /tmp/l12d008/controls > /logs/controls.log 2>&1
control_status=$?
set -e
cp -r /tmp/l12d008/controls /logs/control-details
if [ "$control_status" -ne 0 ]; then exit "$control_status"; fi
# L12-D007 controls have their own strictly contained /tmp/l12d007 sandbox.
mkdir /tmp/l12d007/control-code
cp -r /tmp/l12d008/work/minion-agent-rust /tmp/l12d007/control-code/
cp -r /tmp/l12d008/work/conformance /tmp/l12d007/control-code/
mkdir -p /tmp/l12d007/control-code/minion-agent-python/tests/skills/data /tmp/l12d007/control-code/minion-agent-python/tests/execution/data/r002_ada_oracle
cp /tmp/l12d008/work/minion-agent-python/tests/skills/data/frontmatter-corpus.json /tmp/l12d008/work/minion-agent-python/tests/skills/data/ignore-corpus.json /tmp/l12d007/control-code/minion-agent-python/tests/skills/data/
cp /tmp/l12d008/work/minion-agent-python/tests/execution/data/r002_ada_oracle/systematic_ada292.txt /tmp/l12d007/control-code/minion-agent-python/tests/execution/data/r002_ada_oracle/
set +e
python3 scripts/error-codes-negative-controls.py --tree /tmp/l12d007/control-code/minion-agent-rust --logs /tmp/l12d007/controls > /logs/error-codes-controls.log 2>&1
error_codes_status=$?
set -e
cp -r /tmp/l12d007/controls /logs/error-codes-control-details
if [ "$error_codes_status" -ne 0 ]; then exit "$error_codes_status"; fi
python3 scripts/cargo-cache-regression.py --tree "$PWD" --control-tree /tmp/l12d007/control-code/minion-agent-rust --logs /tmp/l12d007/cache-regression > /logs/cache-regression.log 2>&1
cp -r /tmp/l12d007/cache-regression /logs/cache-regression-details
echo 'L12-D008 LINUX G3, UNPRIVILEGED PROBE/CORPORA AND CONTROLS PASS'
